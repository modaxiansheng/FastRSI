"""Align CURRENT completion supervision with the unchanged typed request.

The consumer, not this head's target, requires consecutive confirmations.
Reuse exact previously encoded features; no new model/robot inference. This
is a retrospective label repair, not another independent validation set.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import traceback


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def run(project, prior, out):
    tick = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    result = dict(status='VALIDATING', new_visual_model_calls=0, new_robot_episodes=0,
        new_gpu_seconds=0, controller_changed=False, threshold=.95,
        scope='Retrospective current-fact label alignment on fixed cached features. '
              'No fresh probe, policy gain, calibrated confidence or cyclic distillation.')

    def save():
        result['wall_seconds'] = time.perf_counter() - tick
        (out / 'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False))

    save()
    try:
        previous = read(prior / 'model_job/learning/RESULT.json')
        feature_path = prior / 'model_job/learning/FEATURES.npz'
        if (previous['status'] != 'COMPLETED' or sha(feature_path) != previous['feature_sha256']
                or sha(prior / 'inputs/INPUTS.json') != previous['input_artifact_sha256']['INPUTS.json']):
            raise ValueError('Unbound cached visual features')
        rows = read(prior / 'inputs/INPUTS.json')
        by_id = {row['id']: row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError('Duplicate visual record')
        facts = {}
        source_bindings = {}
        for source in read(prior / 'REGISTRATION.json')['sources']:
            replay_path = prior / 'replays' / ('init' + str(source['init'])) / 'REPLAY.json'
            replay = read(replay_path)
            if (replay['status'] != 'EXACT_IMAGE_REPLAY'
                    or replay['source_trace_sha256'] != source['trace_sha256']
                    or replay['source_result_sha256'] != source['result_sha256']):
                raise ValueError('Unbound source fact replay')
            source_bindings[str(replay_path.relative_to(project))] = sha(replay_path)
            for frame in replay['rows']:
                for subject, fact in frame['offline_supervision'].items():
                    identity = f'i{source["init"]}-s{frame["step"]}-{subject}'
                    if identity not in by_id:
                        continue
                    row = by_id[identity]
                    if row['role'] != source['role'] or row['subject'] != subject:
                        raise ValueError('Grouped role mismatch')
                    value = bool(fact['native_subgoal'] and not fact['native_grasp']
                                 and not fact['native_gripper_contact'])
                    if value != fact['released_support_proxy']:
                        raise ValueError('Native current fact reconstruction differs')
                    facts[identity] = dict(id=identity, group=row['group'], role=row['role'],
                        subject=subject, current_complete_proxy=value,
                        previously_confirmed_proxy=fact['eligible_to_skip'])
        if set(facts) != set(by_id):
            raise ValueError('Current supervision does not cover cached inputs')
        training = [facts[row['id']] for row in rows if row['role'] != 'probe']
        probes = [facts[row['id']] for row in rows if row['role'] == 'probe']
        (out / 'CURRENT_SUPERVISION.json').write_text(json.dumps(training, indent=2))
        (out / 'CURRENT_PROBE_SUPERVISION.json').write_text(json.dumps(probes, indent=2))
        result['label_repairs'] = [fact for fact in facts.values()
            if fact['current_complete_proxy'] != fact['previously_confirmed_proxy']]
        result['cached_feature_sha256'] = sha(feature_path)
        result['input_sha256'] = sha(prior / 'inputs/INPUTS.json')
        result['prior_result_sha256'] = sha(prior / 'model_job/learning/RESULT.json')
        result['source_bindings'] = source_bindings
        import numpy as np
        import torch
        torch.set_num_threads(4)
        archive = np.load(feature_path, allow_pickle=False)
        matrix = archive['features']
        if (archive['ids'].tolist() != [row['id'] for row in rows]
                or matrix.shape != (len(rows), 1024) or not np.isfinite(matrix).all()):
            raise ValueError('Cached feature schema/order mismatch')
        vector = torch.nn.functional.normalize(torch.from_numpy(matrix).float(), dim=-1)
        result['status'] = 'FITTING'; save()
        versions, predictions = {}, {}
        for name, roles in (('v1_current_seed', {'seed'}),
                            ('v2_current_updated', {'seed', 'update'})):
            indices = [i for i, row in enumerate(rows) if row['role'] in roles]
            selected = [facts[rows[i]['id']] for i in indices]
            counts = Counter((r['group'], r['current_complete_proxy']) for r in selected)
            classes = {r['current_complete_proxy'] for r in selected}
            class_groups = {v: len({g for g, c in counts if c == v}) for v in classes}
            weights = torch.tensor([1 / (len(classes) * class_groups[r['current_complete_proxy']]
                * counts[(r['group'], r['current_complete_proxy'])]) for r in selected])
            target = torch.tensor([r['current_complete_proxy'] for r in selected], dtype=torch.float32)
            torch.manual_seed(17)
            head = torch.nn.Linear(1024, 1)
            torch.nn.init.zeros_(head.weight); torch.nn.init.zeros_(head.bias)
            opt = torch.optim.AdamW(head.parameters(), lr=.02, weight_decay=.01)
            started = time.perf_counter()
            for _ in range(400):
                opt.zero_grad()
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                    head(vector[indices])[:, 0], target, reduction='none') * weights).sum()
                loss.backward(); opt.step()
            fit_seconds = time.perf_counter() - started
            with torch.no_grad():
                probabilities = head(vector)[:, 0].sigmoid().tolist()
            if not all(np.isfinite(probabilities)):
                raise ValueError('Nonfinite corrected head probabilities')
            checkpoint = out / (name + '_head.pt')
            torch.save(head.state_dict(), checkpoint)
            versions[name] = dict(parameters=1025, fit_seconds=fit_seconds,
                final_loss=float(loss.detach()),
                train_groups=sorted({r['group'] for r in selected}),
                class_counts=dict(Counter(str(r['current_complete_proxy']) for r in selected)),
                checkpoint_sha256=sha(checkpoint), checkpoint_file=checkpoint.name)
            predictions[name] = probabilities
        predicted = [dict(id=row['id'], group=row['group'], subject=row['subject'],
            probabilities={v: p[i] for v, p in predictions.items()},
            positive={v: p[i] >= .95 for v, p in predictions.items()})
            for i, row in enumerate(rows) if row['role'] == 'probe']
        (out / 'PROBE_PREDICTIONS.json').write_text(json.dumps(predicted, indent=2))
        metrics = {}
        for version in versions:
            scored = [dict(id=p['id'], actual_current_complete=facts[p['id']]['current_complete_proxy'],
                           positive=p['positive'][version]) for p in predicted]
            metrics[version] = dict(samples=len(scored), episode_groups=1,
                proposed_complete=sum(r['positive'] for r in scored),
                false_complete=sum(r['positive'] and not r['actual_current_complete'] for r in scored),
                missed_complete=sum(not r['positive'] and r['actual_current_complete'] for r in scored),
                rows=scored)
        upstream = []
        for c in previous['calls']:
            if c['role'] == 'probe':
                p=c['upstream_probabilities']; choice=max(p, key=p.get)
                upstream.append(dict(id=c['id'], choice=choice,
                    actual_current_complete=facts[c['id']]['current_complete_proxy']))
        result.update(status='COMPLETED', versions=versions, metrics=metrics,
            upstream_diagnostic=dict(samples=len(upstream), rows=upstream),
            independent_label_blinding=False, deployment_approved=False,
            learning_net_gain_established=False,
            temporal_rule='Current fact target; private consumer requires two actual consecutive estimates separately.')
        save()
        print(json.dumps({k: result[k] for k in ('status','metrics','wall_seconds')}, indent=2))
    except BaseException:
        result.update(status='FAILED', error=traceback.format_exc()); save(); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.prior, args.output)

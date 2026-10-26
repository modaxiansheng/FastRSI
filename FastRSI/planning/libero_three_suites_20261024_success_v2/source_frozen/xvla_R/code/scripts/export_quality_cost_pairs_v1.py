"""Separate factual completion and conditional cost from eight executed branch pairs.

These are one-sample default-pi continuations, not expected success probabilities
or labels for arbitrary learned-router futures. Metadata identifiers are excluded
from model inputs. This exporter does not update or select any model.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np


def main(project, output):
    codec_path = project/'third_party/PhyAgentOS-core/PhyAgentOS/runtime/policy/msgpack_numpy.py'
    spec = importlib.util.spec_from_file_location('pair_codec', codec_path)
    codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)
    def resolve(relative):
        path = project/relative
        if not path.exists() and relative.startswith('runs/'):
            path = project/'planning'/relative[len('runs/'):]
        assert path.is_file(), path
        return path
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    previous = project/'planning/native_paired_q_update_v1_20261010/training/PAIRED_LABELS.json'
    old = json.loads(previous.read_text())
    audit_path = project/'planning/native_first_choice_audit_v2_20261010/RESULT.json'
    audit = json.loads(audit_path.read_text())
    assert audit['status'] == 'AUDITED_FIRST_LEARNED_CHANGE_DIAGNOSTICS'
    pairs = []
    arrays = {}
    def add(initial, step, request, response, outcomes, bindings):
        req = codec.unpackb(request.read_bytes())
        reply = codec.unpackb(response.read_bytes())
        candidates = np.asarray(reply['candidates'], dtype=np.float32)
        assert candidates.shape == (2, 10, 7)
        feature = np.asarray(reply['prefix_feature'], dtype=np.float32)
        state = np.asarray(req['observation/state'], dtype=np.float32)
        assert feature.shape == (4096,) and state.shape == (8,)
        assert np.isfinite(candidates).all() and np.isfinite(feature).all()
        success = [int(x['native_success']) for x in outcomes]
        remaining = [int(x['remaining_actions']) for x in outcomes]
        assert all(0 < n <= 400-step for n in remaining)
        quality_preference = None if success[0] == success[1] else int(success[1] > success[0])
        cost_preference = (int(remaining[1] < remaining[0])
            if all(success) and remaining[0] != remaining[1] else None)
        index = len(pairs)
        arrays[f'feature_{index}'] = feature
        arrays[f'robot_{index}'] = state
        arrays[f'candidates_{index}'] = candidates
        arrays[f'budget_{index}'] = np.array([400-step], dtype=np.float32)
        pairs.append(dict(pair=index, development_group=initial, step=step,
            continuation='unchanged default pi, explicitly matched future noise',
            request=str(request.relative_to(project)), request_sha256=sha(request),
            response=str(response.relative_to(project)), response_sha256=sha(response),
            observed_native_success=success, observed_remaining_actions=remaining,
            quality_preference=quality_preference,
            cost_preference_given_both_success=cost_preference,
            action_count_is_cost_proxy_not_wall_time=True,
            both_failed_does_not_create_preferred_action=not any(success),
            factual_outcomes=outcomes, evidence_bindings=bindings,
            success_labels_not_calibrated_probabilities=True,
            not_labels_for_arbitrary_router_continuation=True))
    for pair in old:
        req = resolve(pair['request'])
        resp = resolve(pair['response'])
        assert sha(req) == pair['request_sha256']
        assert sha(resp) == pair['response_sha256']
        add(pair['initial'], pair['step'], req, resp, pair['outcomes'],
            dict(previous_audited_labels_sha256=sha(previous)))
    assert len(pairs) == 6
    for slot, row in enumerate(audit['cases']):
        result_path = project/f'planning/native_first_choice_gpu{slot}_v1_20261010/RESULT.json'
        result = json.loads(result_path.read_text())
        assert sha(result_path) == row['result_sha256']
        plan = result['plan']
        default_path = resolve(plan['default_outcome'])
        default = json.loads(default_path.read_text())['cases'][0]['sessions'][0]['result']
        assert default['success'] == row['default_success']
        outcomes = [dict(native_success=default['success'], remaining_actions=default['num_steps']-plan['fork_step'],
                        reference=plan['default_outcome'], source_sha256=sha(default_path)),
                    dict(native_success=result['native_success'], remaining_actions=result['remaining_actions'],
                        reference=str(result_path.relative_to(project)), source_sha256=sha(result_path))]
        request = resolve(plan['source_request'])
        response = resolve(plan['source_response'])
        assert sha(request) == plan['source_request_sha256']
        assert sha(response) == plan['source_response_sha256']
        add(row['initial'], row['fork_step'], request, response, outcomes,
            dict(first_choice_audit_sha256=sha(audit_path)))
    output.mkdir(exist_ok=False)
    np.savez_compressed(output/'INPUTS.npz', **arrays)
    (output/'PAIRS.json').write_text(json.dumps(pairs, indent=2)+'\n')
    result = dict(status='EXPORTED_FACTUAL_QUALITY_COST_PAIRS', pairs=len(pairs),
        context_groups=sorted({r['development_group'] for r in pairs}),
        informative_preference_groups=sorted({r['development_group'] for r in pairs
            if r['quality_preference'] is not None or r['cost_preference_given_both_success'] is not None}),
        quality_preference_pairs=sum(r['quality_preference'] is not None for r in pairs),
        conditional_action_cost_preference_pairs=sum(r['cost_preference_given_both_success'] is not None for r in pairs),
        both_failed_pairs=sum(r['both_failed_does_not_create_preferred_action'] for r in pairs),
        online_input_keys=['existing_policy_prefix_feature', 'robot8', 'candidate_action_blocks', 'remaining_budget'],
        excluded_online_keys=['initial_state_id', 'source_episode_id', 'future_success', 'hidden_simulator_state'],
        discounted_success_replaced_by_separate_factual_labels=True,
        model_updates=0, robot_executions=0, new_feature_encodings=0,
        learned_benefit=False, goal_complete=False,
        limitation='Eight branch points in four groups; only three groups have a preference. Do not fit a large decision model or infer broad reliability from this export.')
    (output/'RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.project, args.output)

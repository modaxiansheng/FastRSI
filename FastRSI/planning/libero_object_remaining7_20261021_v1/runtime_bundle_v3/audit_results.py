"""Read explicit new-development condition receipts; no discovery, launch or tuning.

Partial/unsupported outcomes remain unknown. A paired mean is withheld unless all
14 precommitted identities have both outcomes. Nested actor timers are not added
to policy residence or host wall time; full monetary/assistant costs stay unknown.
"""
from pathlib import Path
import argparse
import json
from coordinator_contract import read, sha
from row_factory import MODELS, TASKS, BRANCHES, final_outcomes, paired_difference


def closed_condition(root, row):
    case = (root / 'campaigns' / ('libero_object__' + row['model']) /
            ('task' + str(row['source']['task_id'])) / ('node' + row['node']) /
            ('gpu' + str(row['gpu'])) / row['identity_id'])
    path = case / 'CONDITION_RESULT.json'
    value = read(path)
    assert value['condition'] == row['identity_id']
    assert value['physical_identity'] == row['physical_identity_id']
    assert value['source'] == row['source'] and value['skill_by_branch'] == row['skill_by_branch']
    assert value['software_ok'] is True and value['success_pooling'] is False
    baseline = Path(value['baseline_protocol_path'])
    assert baseline.resolve() == (case / 'base' / row['physical_identity_id'] /
                                  'stream' / (row['identity_id'] + '--base') / 'PROTOCOL_RESULT.json').resolve()
    assert sha(baseline) == value['baseline_protocol_sha256']
    protocol = read(baseline)
    assert protocol['software_ok'] is True and type(protocol['final_success']) is bool
    assert 1 <= len(protocol['attempts']) <= 3
    assert protocol['first_success'] == value['first_success']
    assert protocol['final_success'] == value['native_up_to3']
    raw_costs = {'shared_native': []}
    for attempt in protocol['attempts']:
        raw = baseline.parent / ('attempt_' + str(attempt['index'])) / 'RAW.json'
        if attempt.get('raw_receipt_sha256'):
            assert sha(raw) == attempt['raw_receipt_sha256']
        content = read(raw)
        assert content['error'] is None and content['episode_completed'] is True
        assert type(content['physical_success']) is bool
        assert content['initial_state_sha256'] == row['identity']['initial_state_sha256']
        raw_costs['shared_native'].append(content)
    for arm in BRANCHES:
        raw_costs[arm] = []
        if value['native_up_to3']:
            continue
        item = value['branches'][arm]
        if item.get('status') == 'SKIPPED_UNSUPPORTED':
            assert arm == 'candidate' and row['source']['task_id'] == 2
            assert item['physical_attempts'] == 0 and item['physical_success'] is None
            assert not (case / arm / 'RAW.json').exists()
            continue
        raw_path = case / arm / 'RAW.json'
        assert Path(item['raw_path']).resolve() == raw_path.resolve()
        assert sha(raw_path) == item['raw_sha256']
        content = read(raw_path)
        assert content['error'] is None and content['episode_completed'] is True
        assert type(content['physical_success']) is bool
        assert content['physical_success'] == item['result']['physical_success']
        raw_costs[arm].append(content)
    expected = final_outcomes(value['native_up_to3'], value['branches'])
    assert value['final'] == expected
    assert value['net_vs_ordinary'] == paired_difference(expected['candidate'], expected['ordinary'])
    assert value['net_vs_frozen'] == paired_difference(expected['candidate'], expected['frozen'])
    return dict(condition=row['identity_id'], physical_identity=row['physical_identity_id'],
                model=row['model'], task_id=row['source']['task_id'],
                first=value['first_success'], native_up_to3=value['native_up_to3'],
                final=expected, raw_costs=raw_costs,
                evidence_path=str(path), evidence_sha256=sha(path))


def model_summary(rows, records):
    known = [r for r in records if r.get('status') == 'VERIFIED_CLOSED_CONDITION']
    scored = [r for r in known if all(type(r['final'][b]) is bool for b in BRANCHES)]
    finals = {b: dict(successes=sum(r['final'][b] is True for r in known),
                      known=len([r for r in known if type(r['final'][b]) is bool]),
                      unknown=len(rows) - len([r for r in known if type(r['final'][b]) is bool]))
              for b in BRANCHES}
    costs = {}
    for arm in ('shared_native',) + BRANCHES:
        raw = [item for r in known for item in r['raw_costs'][arm]]
        costs[arm] = dict(closed_physical_attempts=len(raw),
                         closed_actor_seconds=sum(item['seconds'] for item in raw),
                         raw_declared_policy_counts={key: (sum(item['policy_counts'][key] for item in raw)
                             if all(key in item.get('policy_counts', {}) for item in raw) else None)
                             for key in ('policy_rpc_begin', 'policy_rpc_end', 'action_generation_begin',
                                         'action_generation_end', 'neural_forward_begin', 'neural_forward_end')})
    paired = {}
    for control in ('ordinary', 'frozen'):
        wins = sum(r['final']['candidate'] and not r['final'][control] for r in scored)
        losses = sum(r['final'][control] and not r['final']['candidate'] for r in scored)
        paired[control] = dict(known_paired_identities=len(scored), wins=wins, losses=losses,
                              net=wins - losses,
                              complete_14_identity_paired_mean=((wins - losses) / 14.0 if len(scored) == 14 else None))
    return dict(precommitted_physical_identities=len(rows), verified_closed_conditions=len(known),
                first_successes=sum(r['first'] is True for r in known),
                native_up_to3_successes=sum(r['native_up_to3'] is True for r in known),
                final_by_arm=finals, paired_subset_only=paired,
                closed_attempt_actor_costs_only=costs,
                partial_or_failed_attempt_costs_not_in_closed_sums=True,
                full_lifecycle_model_load_wait_host_and_monetary_costs_complete=False)


def audit(root):
    root = Path(root)
    scope = read(root / 'FROZEN_RUNTIME_SCOPE.json')
    assert len(scope['conditions']) == 56
    records = []
    for row in scope['conditions']:
        try:
            records.append(dict(closed_condition(root, row), status='VERIFIED_CLOSED_CONDITION'))
        except (OSError, ValueError, KeyError, AssertionError) as error:
            records.append(dict(condition=row['identity_id'], model=row['model'],
                                task_id=row['source']['task_id'], status='UNKNOWN_NOT_SCORED',
                                error=str(error), physical_success=None))
    return dict(schema='fastrsi_object7_development_result_audit_v1', full_evaluation='PAUSED',
                source_scope_sha256=sha(root / 'FROZEN_RUNTIME_SCOPE.json'),
                models={model: model_summary([r for r in scope['conditions'] if r['model'] == model],
                                            [r for r in records if r['model'] == model]) for model in MODELS},
                per_task={str(task): {model: [dict(condition=r['condition'], status=r['status'],
                    final=r.get('final'), first=r.get('first'), native_up_to3=r.get('native_up_to3'))
                    for r in records if r['model'] == model and r['task_id'] == task]
                    for model in MODELS} for task in TASKS}, conditions=records,
                score_is_development_only=True, frozen_acceptance_reads=0,
                candidate_successes_not_pooled=True, learning_claim=False,
                money=None, assistant_tokens=None,
                cost_scope='Closed actor attempts only; preserve separate partial attempts, cold-load, service/lifecycle and preparation ledgers.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    if args.output:
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

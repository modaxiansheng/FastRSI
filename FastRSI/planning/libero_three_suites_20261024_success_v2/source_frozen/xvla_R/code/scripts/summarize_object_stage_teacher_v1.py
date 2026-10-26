"""Derived audit of frozen teacher diagnoses; never creates training labels."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); rows = []; manifest = {}
    for version in (1, 2, 3):
        run = args.project / 'runs' / f'object_stage_teacher_v{version}_20261006'
        status = read(run / 'STATUS.json'); result = read(run / 'diagnosis/RESULT.json')
        if status['status'] != 'COMPLETED' or result['teacher_calls'] != 4:
            raise ValueError('Teacher batch incomplete')
        services = [read(p) for p in sorted((run / 'service_calls').glob('call_*/RECEIPT.json'))]
        if len(services) != 4: raise ValueError('Missing actual model calls, including rejected ones')
        audit = []
        for item in result['rows']:
            typed = item['status'] == 'TYPED_DIAGNOSIS'
            proxy = item['offline_audit']['current_complete_proxy']
            stage = item.get('proposed_stage') if typed else None
            audit.append(dict(id=item['id'], status=item['status'], proposed_stage=stage,
                offline_complete_proxy=proxy,
                agreement_with_proxy=(stage == 'complete') == proxy if typed and stage != 'unknown' else None,
                missed_complete_proxy=typed and proxy and stage == 'incomplete',
                false_complete_proxy=typed and not proxy and stage == 'complete',
                evidence=item.get('facts', {}).get('evidence'), training_label=None))
        stop = read(run / 'STOP_RECEIPT.json')
        if len(stop) != 1 or stop[0]['running']: raise ValueError('Teacher service not stopped')
        rows.append(dict(version=version, view_mode=read(run / 'REGISTRATION.json').get('view_mode'),
            teacher_calls=4, typed_calls=sum(x['status'] == 'TYPED_DIAGNOSIS' for x in audit),
            agreement_with_proxy=sum(x['agreement_with_proxy'] is True for x in audit),
            missed_complete_proxy=sum(x['missed_complete_proxy'] for x in audit),
            false_complete_proxy=sum(x['false_complete_proxy'] for x in audit), checkpoints=audit,
            startup_seconds=status['startup_seconds'], diagnosis_seconds=status['diagnostic_seconds'],
            reserved_card_seconds=status['reserved_card_seconds'], job_wall_seconds=status['seconds'],
            service_generation_seconds=sum(s['generation_seconds'] for s in services),
            prompt_tokens=sum(s['prompt_tokens'] for s in services),
            completion_tokens=sum(s['completion_tokens'] for s in services),
            rejected_call_cost_included=True, stopped_container=stop[0]['container']))
        for path in run.rglob('*'):
            if path.is_file() and ('__pycache__' not in path.parts):
                manifest[str(path.relative_to(args.project))] = sha(path)
    args.out.mkdir(exist_ok=False, parents=True)
    summary = dict(status='COMPLETED_DERIVED_AUDIT', campaigns=rows, calls=12,
        serial_job_wall_seconds=sum(r['job_wall_seconds'] for r in rows),
        one_card_reservation_seconds=sum(r['reserved_card_seconds'] for r in rows),
        reliable_labels=0, new_robot_episodes=0, learned_benefit=False,
        input_bindings='Private proxies read after calls; teacher opinions remain unverified',
        cost_scope='Three serial one-card job reservations, startup and teardown included. '
                   'Generation, request time and startup are nested, never added again. '
                   'Not full project expenditure or method cold-start bill.',
        correction='Rejected v1 checkpoint agreement is null regardless of legacy RESULT audit_complete_matches.')
    (args.out / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (args.out / 'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'campaigns'}, ensure_ascii=False))


if __name__ == '__main__': main()

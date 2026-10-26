"""Analyze one real wiring run without inventing a frozen-arm counterfactual."""
import argparse, hashlib, json
from pathlib import Path
from analyze_resident_full_os import service_index
from analyze_structural_window import analyze_development_case
from orchestrate_task_snapshot_native import check_execution

def analyze(run):
    errors, sources=[],{}
    service=service_index(run/'resident/pool/qwen/service_calls', errors, sources)
    row, details, bound=analyze_development_case(run/'pair0/N',service,'N')
    sources.update(bound)
    native=check_execution(run)
    for name in ['STATUS.json','PROTOCOL.json','UPDATE_RECEIPT.json','FROZEN.json','UPDATED.json','ADDED_PAIR.json','CONTAINER_AUDIT.json']:
        p=run/name;sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    status=json.loads((run/'STATUS.json').read_text());receipt=json.loads((run/'UPDATE_RECEIPT.json').read_text())
    errors.extend(row['errors_or_missing_evidence'])
    runtime_passed=not errors and row['trajectory_audit']['status']=='PASSED' and row['worker_status']=='COMPLETED'
    postcheck_error=(status['status']=='FAILED' and "assert len(reads)==1" in status.get('error',''))
    passed=runtime_passed and (status['status']=='COMPLETED' or postcheck_error)
    return {'status':('RUNTIME_PASSED_POSTCHECK_REPAIRED' if postcheck_error else 'PASSED') if passed else 'NOT_ESTABLISHED', 'run_id':run.name,
        'original_campaign_status':status['status'],'original_campaign_error':status.get('error'),
        'postcheck_repair':'Read task_window_read from actual AGENT_EVENTS, not OS_COST_EVENTS; archived run unchanged; no robot rerun' if postcheck_error else None,
        'scope':'One native dev Session using a new snapshot made from paid historical feedback. No learned gain or counterfactual saving established.',
        'native_check':native,'case':row,'source_errors':errors,
        'costs':{'campaign_full_wall_seconds':status['full_wall_seconds'],
                 'pool_startup_seconds':status['pool_startup_seconds'],
                 'snapshot_preparation_wall_seconds':receipt['preparation_wall_seconds'],
                 'case':row['costs'], 'qwen_service_calls':len(service),
                 'units':'seconds; nested/overlapping components must not be summed to wall time',
                 'accounting':'Engineering verification cost. Historical robot execution paid previously; no free cold-start claim. No frozen run, no estimated measured saving.'},
        'sources':sources,'learning_gain_established':False,'default_changed':False}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args()
    report=analyze(a.run)
    with a.output.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':report['status'],'costs':{k:v for k,v in report['costs'].items() if k!='case'},
        'quality':report['case']['quality'],'VLA_calls':report['case']['VLA_calls'],
        'executed_steps':report['case']['executed_steps'],'source_errors':report['source_errors']},ensure_ascii=False))

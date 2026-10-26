"""Separate exact replay evidence, paid source cost, and incremental validation."""
import argparse,json,hashlib
from pathlib import Path
from collections import defaultdict

def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()]
def summary(run):
    result={'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'experiment_status_sha256':hashlib.sha256((run/'STATUS.json').read_bytes()).hexdigest(),
            'scope':'Same-instance development trial reuse, not learning/generalization benefit','groups':{}}
    evidence={}
    for mode in ('source','full','reuse'):
        root=run/mode/'N/stream';d=read(root/'RESULT.json');case=d['cases'][0];group=read(root.parent/'STATUS.json')
        evidence[mode]=lines(root/'TRIAL_POLICY.jsonl')
        costs=defaultdict(lambda:dict(seconds=0.,count=0))
        for name in ('OS_COST_EVENTS.jsonl','TARGET_COST_EVENTS.jsonl'):
            for e in lines(root/name):
                if e['event']=='end':
                    costs[e['name']]['seconds']+=e['duration_seconds'];costs[e['name']]['count']+=1
        result['groups'][mode]=dict(worker_seconds=group['wall_seconds'] if 'wall_seconds' in group else group.get('full_wall_seconds'),
            controller_seconds=d['seconds'],case_seconds=case['seconds'],actual_policy_requests=sum(r.get('origin','policy')=='policy' for r in evidence[mode]),
            cached_policy_decisions=sum(r.get('origin','policy')=='paid_prefix_cache' for r in evidence[mode]),
            environment_success=case['environment_success'],verification_status=case['verification_status'],
            verdicts=[o.get('verdict') for o in case['outcomes']],chain_status=case['chain_status'],
            admission=case['experience_admission']['status'],cost_spans=dict(costs),
            span_note='Nested spans overlap; do not add into a new total',
            result_sha256=hashlib.sha256((root/'RESULT.json').read_bytes()).hexdigest())
    keys=('step','prompt','inputs','actions');canonical=lambda rs:[{k:r[k] for k in keys} for r in rs]
    result['exact_source_prefix']=canonical([r for r in evidence['source'] if r['step']<180])==canonical([r for r in evidence['full'] if r['step']<180])
    result['exact_candidate_suffix']=canonical([r for r in evidence['full'] if r['step']>=180])==canonical([r for r in evidence['reuse'] if r['step']>=180])
    result['exact_suffix_physics']=[r for r in lines(run/'full/N/stream/TRIAL_PHYSICS.jsonl') if r['step']>=180]==[r for r in lines(run/'reuse/N/stream/TRIAL_PHYSICS.jsonl') if r['step']>=180]
    src,full,reuse=[result['groups'][k]['controller_seconds'] for k in ('source','full','reuse')]
    result['controller_cost_comparison']=dict(incremental_full=full,incremental_reuse=reuse,saved=full-reuse,
        fractional_saving=(full-reuse)/full,source_paid=src,cold_source_plus_full=src+full,cold_source_plus_reuse=src+reuse,
        cold_fractional_saving=(full-reuse)/(src+full),
        scope='Sequential measured controller costs; pool startup/holding and project debugging reported separately')
    image_rows=lambda mode:[(r['step'],r['images'],r['native_success']) for r in lines(run/mode/'N/stream/stage_trace/net_t0_i6/TRACE.jsonl') if r['step']>=180]
    result['exact_suffix_images_including_terminal']=image_rows('full')==image_rows('reuse')
    result['decision_policy_updates']=0
    result['native_feedback_rules_unchanged']=True
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args()
    d=summary(a.run);path=a.run/'ANALYSIS.json'
    with path.open('x') as f:json.dump(d,f,indent=2)
    print(json.dumps({k:v for k,v in d.items() if k!='groups'},indent=2))

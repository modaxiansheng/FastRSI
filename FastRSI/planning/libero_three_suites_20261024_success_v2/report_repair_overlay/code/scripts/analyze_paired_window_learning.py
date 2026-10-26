"""Same-trained-snapshot ablation with separately billed candidate validation."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    import argparse,re
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);args=parser.parse_args()
    assert re.fullmatch(r'[a-zA-Z0-9_]+',args.name)
    run=ROOT/'runs'/args.name;out=ROOT/'planning'/args.name
    assert read(run/'STATUS.json')['status']=='COMPLETED'
    subprocess.run([sys.executable,str(ROOT/'scripts/analyze_feedback_window.py'),'--name',args.name],check=True,stdout=subprocess.DEVNULL)
    initial=read(run/'LEARNING_PREPARE.json');summary=read(out/'SUMMARY.json')
    assert initial['frozen_sha256']==initial['learned_sha256']
    previous=run/'learning_initial/LEARNED.json';proof=[]
    for pair in sorted(run.glob('pair*')):
        assert sha(pair/'LEARNED.json')==sha(previous)
        assert sha(pair/'FROZEN.json')==initial['frozen_sha256']
        frozen=read(pair/'N/stream/RESULT.json')['cases'][0]['window_learning_decision']
        learned=read(pair/'R/stream/RESULT.json')['cases'][0]['window_learning_decision']
        assert frozen['snapshot_sha256']==sha(pair/'FROZEN.json')
        assert frozen['updates']==initial['initial_updates']
        assert learned['snapshot_sha256']==sha(pair/'LEARNED.json')
        before=read(pair/'LEARNED.json');after=read(pair/'LEARNED_NEXT.json')
        feedback=read(pair/'LEARNING_FEEDBACK_BATCH.json');update=read(pair/'LEARNING_UPDATE.json')
        assert after['records']==before['records']+feedback
        assert {r['evidence_id'] for r in feedback}=={str((pair/arm).relative_to(ROOT)) for arm in ('F','R')}
        assert len({r['comparison_signature'] for r in feedback})==1
        for record in feedback:
            assert record['mode']=='development'
            assert all(sha(ROOT/rel)==digest for rel,digest in record['sources'].items())
        assert update['before_sha256']==sha(pair/'LEARNED.json') and update['after_sha256']==sha(pair/'LEARNED_NEXT.json')
        proof.append({'pair':pair.name,'frozen_policy':frozen['selected_mode'],'learning_policy':learned['selected_mode'],
                      'next_policy':after['incumbent'],'before_records':len(before['records']),'after_records':len(after['records']),
                      'before_sha256':sha(pair/'LEARNED.json'),'next_sha256':sha(pair/'LEARNED_NEXT.json'),
                      'session_snapshot_unchanged':True,'frozen_snapshot_unchanged':True,
                      'selection_changed_relative_to_frozen':frozen['selected_mode']!=learned['selected_mode'],
                      'update_reason':after['last_update']['reason'],
                      'update_seconds':update['seconds'],'probe_lifecycle_seconds':update['candidate_validation_lifecycle_seconds']})
        previous=pair/'LEARNED_NEXT.json'
    extra=initial['prepare_seconds']+sum(p['update_seconds'] for p in proof)
    probes=sum(p['probe_lifecycle_seconds'] for p in proof)
    totals=summary['totals']; r=totals['R'];comparisons={}
    for arm in ('N','F'):
        b=totals[arm]
        comparisons[arm]={'calls_saved':b['vla_calls']-r['vla_calls'],
                          'execution_lifecycle_seconds_saved_including_updates':b['full_wall_seconds']-r['full_wall_seconds']-extra,
                          'execution_percent_saved_including_updates':100*(1-(r['full_wall_seconds']+extra)/b['full_wall_seconds']),
                          'learning_plus_probe_seconds_saved':b['full_wall_seconds']-r['full_wall_seconds']-extra-probes}
    report={'scope':'Matched development diagnostic; no final evaluation, noninferiority or cold-start net-benefit claim',
            'initial_snapshots_identical':True,'proof':proof,'totals':totals,'comparisons':comparisons,
            'costs':{'common_history_lifecycle_seconds':initial['paid_history_lifecycle_seconds'],
                     'update_and_prepare_seconds':extra,'candidate_validation_lifecycle_seconds':probes,
                     'learning_execution_plus_updates':r['full_wall_seconds']+extra,
                     'learning_execution_plus_updates_and_validation':r['full_wall_seconds']+extra+probes,
                     'current_all_arm_lifecycle_seconds':sum(t['full_wall_seconds'] for t in totals.values()),
                     'process_wall_seconds':read(run/'STATUS.json')['full_wall_seconds']},
            'quality':'Native and semantic status reported separately; narrative quarantine unchanged.',
            'resource_scope':'Each execution arm has two GPUs. R also consumes declared F development probe; extra compute billed explicitly, not an equal-total-compute efficacy claim.',
            'limitation':'Two-SE switching screen is an engineering heuristic. No observed switch is not itself proof that learning saves cost.'}
    (out/'LEARNING_SUMMARY.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

"""Audit the actual feedback -> snapshot -> new Session -> action path."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    import argparse,re
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);args=parser.parse_args()
    assert re.fullmatch(r'[a-zA-Z0-9_]+',args.name)
    run=ROOT/'runs'/args.name;out=ROOT/'planning'/args.name
    assert read(run/'STATUS.json')['status']=='COMPLETED'
    subprocess.run([sys.executable,str(ROOT/'scripts/analyze_feedback_window.py'),'--name',args.name],check=True,stdout=subprocess.DEVNULL)
    summary=read(out/'SUMMARY.json');initial=read(run/'LEARNING_PREPARE.json')
    previous=run/'learning_initial/LEARNED.json';proof=[]
    for pair in sorted(run.glob('pair*')):
        assert sha(pair/'LEARNED.json')==sha(previous)
        assert sha(pair/'FROZEN.json')==initial['frozen_sha256']
        frozen=read(pair/'N/stream/RESULT.json')['cases'][0]['window_learning_decision']
        result=read(pair/'R/stream/RESULT.json')['cases'][0]
        learned=result['window_learning_decision']
        assert frozen['snapshot_sha256']==sha(pair/'FROZEN.json') and frozen['updates']==0
        assert learned['snapshot_sha256']==sha(pair/'LEARNED.json')
        state=read(pair/'LEARNED.json');next_state=read(pair/'LEARNED_NEXT.json')
        feedback=read(pair/'LEARNING_FEEDBACK.json');update=read(pair/'LEARNING_UPDATE.json')
        assert next_state['records']==state['records']+[feedback]
        assert feedback['evidence_id']==str((pair/'R').relative_to(ROOT))
        assert feedback['mode']=='development'
        assert all(sha(ROOT/rel)==digest for rel,digest in feedback['sources'].items())
        assert update['before_sha256']==sha(pair/'LEARNED.json')
        assert update['after_sha256']==sha(pair/'LEARNED_NEXT.json')
        events=[json.loads(s) for s in (pair/'R/stream/OS_COST_EVENTS.jsonl').read_text().splitlines() if s.strip()]
        choices=[e for e in events if e.get('event')=='feedback_window_choice']
        assert choices and all(e['mode']==learned['selected_mode'] for e in choices)
        proof.append({'pair':pair.name,'frozen_mode':frozen['selected_mode'],'learned_mode':learned['selected_mode'],
                      'prior_records':len(state['records']),'next_records':len(next_state['records']),
                      'actual_vla_calls':len(choices),'selected_policy_actually_used':True,
                      'previous_session_snapshot_loaded':True,'session_snapshot_unchanged':True,
                      'update_seconds':update['seconds'],'feedback_id':feedback['evidence_id'],
                      'semantic_verdicts':feedback['semantic_verdicts'],
                      'narrative_admission':feedback['native_experience_admission']})
        previous=pair/'LEARNED_NEXT.json'
    overhead=initial['prepare_seconds']+sum(x['update_seconds'] for x in proof)
    r=summary['totals']['R'];comparisons={}
    for arm in ['N','F']:
        base=summary['totals'][arm]
        comparisons[arm]={'vla_calls_saved':base['vla_calls']-r['vla_calls'],
                          'vla_reduction_percent':100*(1-r['vla_calls']/base['vla_calls']),
                          'lifecycle_seconds_saved_including_updates':base['full_wall_seconds']-r['full_wall_seconds']-overhead,
                          'lifecycle_reduction_percent_including_updates':100*(1-(r['full_wall_seconds']+overhead)/base['full_wall_seconds'])}
    report={'scope':'Full native chain mechanism diagnostic, paid development history, two new development roots. Not a cold-start efficacy or noninferiority test.',
            'proof':proof,'totals':summary['totals'],'comparisons':comparisons,
            'costs':{'history_lifecycle_seconds':initial['paid_history_lifecycle_seconds'],
                     'prepare_and_updates_seconds_outside_workers':overhead,
                     'current_learned_lifecycle_plus_updates':r['full_wall_seconds']+overhead,
                     'learned_plus_entire_history_conservative_bill':initial['paid_history_lifecycle_seconds']+r['full_wall_seconds']+overhead,
                     'current_all_arm_lifecycle_seconds':sum(t['full_wall_seconds'] for t in summary['totals'].values()),
                     'accounting':'Snapshot read/selection, validation and native feedback already inside worker lifecycle. Updates outside added once. Historical unsuccessful startup kept in prior project bill, not repeated here.'},
            'limitations':['Frozen N is pre-update, not same trained snapshot with future updates disabled.',
                           'Historical records cover only articulated goals. Other scopes keep fallback.',
                           'Selected fixed10 is an existing candidate, not a novel execution algorithm.',
                           'Native outcome learning does not lift narrative quarantine or establish visual-verifier equivalence.']}
    (out/'LEARNING_SUMMARY.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

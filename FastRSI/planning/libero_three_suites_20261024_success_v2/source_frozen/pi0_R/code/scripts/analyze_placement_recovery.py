"""Read-only stage-flow results, including failures and actual policy prompts."""
import argparse,hashlib,json,sys
from pathlib import Path
from analyze_representative_batch import analyze
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fastrsi.placement_recovery import GOAL,CORRECTION
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run_id');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();run=ROOT/'runs'/a.run_id
    control=read(run/'PROTOCOL.json').get('control_kind','placement')
    status=read(run/'STATUS.json');assert status['status'] in ('COMPLETED','FAILED')
    pairs=[]
    for p in sorted(run.glob('pair*')):
        arms={}
        for arm in ('N','F'):
            root=p/arm;result=root/'stream/RESULT.json'
            if not result.exists():continue
            obj=analyze(root);r=read(result)['cases'][0]
            events=[json.loads(x) for x in (root/'stream/OS_COST_EVENTS.jsonl').read_text().splitlines()]
            actual=[e for e in events if e.get('event')=='actual_policy_input']
            choices=[e for e in events if e.get('event')==('action_window_choice' if control=='action_window' else 'stage_flow_choice')]
            logs=root/'stream'/r['case']['id']/'AGENT_EVENTS.jsonl'
            reads=[json.loads(x) for x in logs.read_text().splitlines() if json.loads(x).get('event')==('action_window_read' if control=='action_window' else 'stage_flow_read')]
            native=[json.loads(x) for x in (root/'stream/TARGET_COST_EVENTS.jsonl').read_text().splitlines()]
            native=[e for e in native if e.get('event')=='native_observation' and str(e.get('session_id','')).startswith(r['case']['id'])]
            last={e['session_id']:e for e in native}
            w=root/'stream'/r['case']['id']/'workspace'
            episode=w/'artifacts/runtime'/r['case']['id']/'episode.json'
            executed=sum(e['steps'] for e in last.values());resets=sum(e['method']=='reset' for e in native)
            goal_kept=r['case']['task_description']==GOAL and r['planned_session']['task_description']==GOAL and read(episode)['benchmark']['task_description']==GOAL
            expected=[GOAL if e['step']<260 else CORRECTION for e in choices if e.get('action')=='staged']
            prompts=[e['prompt'] for e in actual]
            route_ok=(prompts==expected and bool(expected)) if arm=='F' else all(x==GOAL for x in prompts)
            tracefile=root/'stream/stage_trace'/r['case']['id']/'TRACE.jsonl'
            trace=[json.loads(x) for x in tracefile.read_text().splitlines()] if tracefile.exists() else []
            chunks=[x for x in trace if x['method']=='action_chunk']
            if control=='action_window':
                expected_steps=5 if arm=='F' else 10
                actual_window_ok=bool(chunks) and all(len(x['actions'])==expected_steps for x in chunks)
                route_ok=all(x==GOAL for x in prompts) and actual_window_ok
                if arm=='F':route_ok=route_ok and len(choices)==len(prompts)==len(chunks) and all(e['submitted_steps']==5 for e in choices)
            else:actual_window_ok=None
            arms[arm]={'success':r.get('environment_success'),'verdicts':[o.get('verdict') for o in r.get('outcomes',[])],
              'chain':r['chain_status'],'steps':executed,'resets':resets,'within_520_single_reset':executed<=520 and resets==1,
              'original_goal_preserved':goal_kept,'policy_route_verified':route_ok,'actual_policy_calls':len(prompts),
              'actual_action_window_verified':actual_window_ok,
              'executed_action_lengths':sorted(set(len(x['actions']) for x in chunks)),
              'prompt_counts':{s:prompts.count(s) for s in sorted(set(prompts))},
              'phase_switch_steps':[e.get('step') for i,e in enumerate(choices) if not i or e.get('phase')!=choices[i-1].get('phase')] if control!='action_window' else [],
              'experience_reads':reads,'admission':r.get('experience_admission'),
              'evaluation_unchanged':r['experience_before']==r['experience_after'] if r['case']['mode']=='evaluation' else None,
              'evaluation_closed':(w/'.FASTRSI_EVALUATION_CLOSED.json').exists(),
              'cost':obj['bill'],'stage_seconds':obj['stage_seconds'],'model_purposes':obj['purpose_summary'],
              'sources':{str(result.relative_to(ROOT)):sha(result),str(episode.relative_to(ROOT)):sha(episode)}}
        pairs.append({'pair':p.name,'plan':read(p/'PLAN.json'),'arms':arms})
    report={'control_kind':control,'status':status,'pairs':pairs,'learned_flow':read(run/'LEARNED_FLOW.json') if (run/'LEARNED_FLOW.json').exists() else None,
      'scope':'Task-scoped engineering candidate and prospective evidence-based reuse screen; not learned candidate discovery, calibrated route policy, or general speedup.',
      'cost_note':'Native startup, execution, reports, verification and failed branches retained. Prior failure-discovery campaign and human development are not free; cross-campaign cold-start allocation is not estimated.',
      'full_wall_seconds':status['full_wall_seconds'],
      'two_gpu_reservation_seconds':sum(r['cost']['full_wall_seconds'] for p in pairs for r in p['arms'].values()),
      'gpu_allocation_seconds':sum(r['cost']['hcu_allocation_seconds'] for p in pairs for r in p['arms'].values())}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    for p in pairs:
        print(p['pair'],{a:{k:r[k] for k in ['success','verdicts','chain','steps','resets','original_goal_preserved','policy_route_verified','actual_policy_calls','phase_switch_steps']} for a,r in p['arms'].items()})

if __name__=='__main__':main()

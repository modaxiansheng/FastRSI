"""Check full native execution, fixed experience and selected actual commands."""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from PhyAgentOS.runtime.policy.msgpack_numpy import unpackb
from phase_iql_feedback_update_v1 import AdaptivePredictor

def main():
    root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=False)
    rows=[];firsts=[];first_observations=[];commands=0;models={}
    assert json.loads((root/'STATUS.json').read_text())['status']=='COMPLETED'
    for arm in ('N','F','R'):
        predictor=AdaptivePredictor(root/('VALUE_MODEL_'+arm+'.npz'));models[arm]=predictor.sha256
        stream=root/arm/'stream';result=json.loads((stream/'RESULT.json').read_text())
        assert result['status']=='COMPLETED' and len(result['cases'])==1
        case=result['cases'][0]
        assert case['chain_status']=='PASSED' and case['experience_before']==case['experience_after']
        assert case['terminal_report']['read_coverage']['complete']
        assert not case['terminal_report']['maintenance_writes']
        events=list(map(json.loads,(stream/'OS_COST_EVENTS.jsonl').read_text().splitlines()))
        reads=[e for e in events if e['event']=='native_value_experience_read']
        calls=list(map(json.loads,(stream/'value_service/VALUE_POLICY_CALLS.jsonl').read_text().splitlines()))
        sessions={s['session_id']:s for s in case['sessions']}
        assert set(sessions)=={r['session_id'] for r in reads}=={r['session_id'] for r in calls}
        for e in reads: assert e['model_sha256']==predictor.sha256
        traces={sid:list(map(json.loads,(stream/'stage_trace'/sid/'TRACE.jsonl').read_text().splitlines())) for sid in sessions}
        totals=defaultdict(float);counts=defaultdict(int)
        for file in ('OS_COST_EVENTS.jsonl','TARGET_COST_EVENTS.jsonl'):
            for e in map(json.loads,(stream/file).read_text().splitlines()):
                if e['event']=='end':
                    totals[e['name']]+=e['duration_seconds'];counts[e['name']]+=1
        extras=defaultdict(int);selected_second=0
        for r in calls:
            sid=r['session_id'];name=r['evidence_name']
            req=unpackb((stream/'value_service'/(name+'_request.msgpack')).read_bytes())
            res=unpackb((stream/'value_service'/(name+'_response.msgpack')).read_bytes())
            cand=res['candidates'];selected=r['selected_index'];step=r['step']
            assert np.array_equal(res['actions'],cand[selected])
            assert r['remaining_budget']==400-step
            trace=[t for t in traces[sid] if t['method']=='action_chunk' and t['step']>step]
            t=min(trace,key=lambda t:t['step']);submitted=np.asarray(t['actions'],np.float32)
            assert np.array_equal(submitted,cand[selected][:min(10,400-step)])
            assert 1<=t['step']-step<=len(submitted)
            for j,digest in enumerate(r['candidate_noise_sha256']):
                noise=np.random.default_rng(np.random.SeedSequence([r['noise_base'],r['index'],j])).standard_normal((10,32)).astype(np.float32)
                assert hashlib.sha256(noise.tobytes()).hexdigest()==digest
            if r['enabled']:
                feature=res['prefix_feature'];robot=req['observation/state']
                i,rank=predictor.score(feature,robot,cand,400-step)
                assert selected==i and np.allclose(rank['q_scores'],r['choice_meta']['q_scores'],atol=1e-6,rtol=0)
                value=predictor.value(feature,robot,400-step)
                assert abs(value-r['choice_meta']['state_value'])<1e-5
                needed=rank['q_scores'][0]<value and extras[sid]<20
                assert needed==(len(cand)==2) and extras[sid]==r['choice_meta']['extras_before']
                extras[sid]+=int(needed)
            else:assert selected==0 and len(cand)==1
            selected_second+=int(selected==1);commands+=1
            if r['index']==0 and sid==case['case']['id']:
                firsts.append(cand[0]);first_observations.append(req)
        assert all(n<=20 for n in extras.values())
        agent=list(map(json.loads,(stream/case['case']['id']/'AGENT_EVENTS.jsonl').read_text().splitlines()))
        assert any(e['event']=='agent_closed' for e in agent)
        assert any(e['event']=='update_blocked' for e in agent)
        status=json.loads((root/arm/'STATUS.json').read_text())
        stops=json.loads((stream/'STOP_RECEIPT.json').read_text())
        assert all(not s['running'] and s['finished'] for s in stops)
        verdicts=[o.get('verdict') for o in case['outcomes']]
        rows.append(dict(arm=arm,native_success=case['environment_success'],verification_status=case['verification_status'],
            semantic_verdicts=verdicts,native_semantic_conflict=case['environment_success']!=('success' in verdicts),
            chain_status=case['chain_status'],sessions=len(sessions),actions=sum(s['result']['num_steps'] for s in sessions.values()),
            control_requests=len(calls),candidate_inferences=sum(r['candidate_count'] for r in calls),selected_second=selected_second,
            case_seconds=case['seconds'],controller_seconds=status['controller_seconds'],startup_seconds=status['startup_seconds'],
            full_wall_seconds=status['full_wall_seconds'],components_seconds=dict(totals),components_calls=dict(counts),
            fixed_experience_unchanged=True,actual_session_read=True,closed_agent=True))
    assert len(firsts)==3 and all(np.array_equal(firsts[0],x) for x in firsts[1:])
    for key in ('observation/image','observation/wrist_image','observation/state'):
        assert all(np.array_equal(first_observations[0][key],x[key]) for x in first_observations[1:])
    assert all(first_observations[0]['prompt']==x['prompt'] for x in first_observations[1:])
    (out/'AUDIT.json').write_text(json.dumps(dict(status='PASSED_NATIVE_BINDING_AUDIT',cases=rows,
        checked_commands=commands,first_native_candidate_exact=True,initial_policy_observation_exact=True,model_sha256_by_arm=models,
        cold_start_net_gain_established=False,cross_session_update_gain_established=False,
        warning='Nested timing spans overlap. This audit is not itself a generalization or RSI gain claim.'),indent=2)+'\n')
    print(json.dumps(dict(commands=commands,cases=rows)),flush=True)

if __name__=='__main__':main()

"""Validate actual omitted-last-choice branches, including partial terminal chunks."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
from phase_iql_value_v1 import Predictor

def main(project,output):
    codec_path=project/'third_party/PhyAgentOS-core/PhyAgentOS/runtime/policy/msgpack_numpy.py'
    spec=importlib.util.spec_from_file_location('last_audit_codec',codec_path)
    codec=importlib.util.module_from_spec(spec);spec.loader.exec_module(codec)
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    cases=[]
    for slot,initial in enumerate((34,35)):
        root=project/f'planning/native_last_choice_gpu{slot}_v2_20261010'
        result=json.loads((root/'RESULT.json').read_text());plan=result['plan']
        assert result['status']=='COMPLETED' and result['initial']==initial==plan['initial']
        for key in ('source_request','source_response','reference_outcome','model'):
            assert sha(project/plan[key])==plan[key+'_sha256']
        reference=json.loads((project/plan['reference_outcome']).read_text())['cases'][0]['sessions'][0]['result']
        assert reference['success'] and reference['num_steps']-plan['fork_step']==plan['reference_remaining_actions']
        service=(project/plan['source_response']).parent
        calls=[json.loads(x) for x in (service/'VALUE_POLICY_CALLS.jsonl').read_text().splitlines()]
        assert max(r['step'] for r in calls if r['selected_index']==1)==plan['fork_step']
        assert all(r['selected_index']==0 for r in calls if r['step']>plan['fork_step'])
        prefix=[codec.unpackb((service/(r['evidence_name']+'_response.msgpack')).read_bytes())['actions'].tolist()
                for r in calls if r['step']<plan['fork_step']]
        assert prefix==plan['prefix']
        expected=codec.unpackb((project/plan['source_request']).read_bytes())
        original=codec.unpackb((project/plan['source_response']).read_bytes())
        predictor=Predictor(project/plan['model']);count=0;before=plan['fork_step'];records=[]
        for i,row in enumerate(result['rows']):
            assert row['before']==before and 0<row['after']-before<=10
            assert row['after']-before==10 or i==len(result['rows'])-1
            index=before//10;request_path=root/f'{index:03d}_request.msgpack';response_path=root/f'{index:03d}_response.msgpack'
            request=codec.unpackb(request_path.read_bytes());reply=codec.unpackb(response_path.read_bytes())
            tag=request['_branch_baseline_v2'];n=2 if before==plan['fork_step'] else 1
            assert tag['step']==before and tag['fork_step']==plan['fork_step'] and tag['noise_base']==plan['noise_base']
            candidates=np.asarray(reply['candidates']);assert candidates.shape==(n,10,7)
            assert row['selected_index']==0 and row['candidates']==n
            assert np.array_equal(np.load(root/f'{index:03d}_command.npz',allow_pickle=False)['actions'],candidates[0])
            for j in range(n):
                noise=np.random.default_rng(np.random.SeedSequence([plan['noise_base'],index,j])).standard_normal((10,32)).astype(np.float32)
                assert hashlib.sha256(noise.tobytes()).hexdigest()==reply['branch_baseline']['noise_sha256'][j]
            if before==plan['fork_step']:
                for key in ('observation/image','observation/wrist_image','observation/state'):
                    assert np.array_equal(request[key],expected[key])
                assert request['prompt']==expected['prompt'] and np.array_equal(candidates,original['candidates'])
                chosen,score=predictor.score(reply['prefix_feature'],request['observation/state'],candidates,400-before)
                assert chosen==1 and result['intervention_selected_index']==0
                assert np.allclose(score['q_scores'],result['reference_model_scores']['q_scores'],atol=1e-6)
            count+=n;before=row['after']
            records.append(dict(before=row['before'],after=before,request_sha256=sha(request_path),response_sha256=sha(response_path),
                command_sha256=sha(root/f'{index:03d}_command.npz')))
        assert before==result['physical_steps'] and before<=400
        assert count==result['actor_inferences'] and len(records)==result['remaining_control_requests']
        assert result['remaining_actions']==before-plan['fork_step']
        assert result['parameter_updates']==result['persistent_updates']==0 and not result['full_agent_chain']
        for stop in json.loads((root/'STOP_RECEIPT.json').read_text()):
            assert not stop['state']['Running'] and stop['state']['Pid']==0
        successes=[result['native_success'],reference['success']]
        remaining=[result['remaining_actions'],plan['reference_remaining_actions']]
        preference=(None if successes[0]==successes[1] and remaining[0]==remaining[1] else
            int(successes[1]>successes[0]) if successes[0]!=successes[1] else int(remaining[1]<remaining[0]) if all(successes) else None)
        cases.append(dict(initial=initial,fork_step=plan['fork_step'],native_success=successes,remaining_actions=remaining,
            preferred_observed_candidate=preference,result_sha256=sha(root/'RESULT.json'),checked_commands=len(records),
            checked_candidates=count,records=records,reference_suffix_default_pi=True,
            diagnostic_seconds=result['seconds_including_close'],host_seconds=json.loads((root/'HOST.json').read_text())['seconds']))
    output.mkdir(exist_ok=False)
    result=dict(status='AUDITED_REAL_LAST_CHOICE_PAIR_SUPERVISION',cases=cases,goal_complete=False,
        reference_reuse_bound_to_exact_fork_and_candidates=True,robot_executions=2,model_updates=0,
        independent_performance_claim=False,default_unchanged=True)
    (output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({**result,'cases':[{k:v for k,v in r.items() if k!='records'} for r in cases]}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.project,args.output)

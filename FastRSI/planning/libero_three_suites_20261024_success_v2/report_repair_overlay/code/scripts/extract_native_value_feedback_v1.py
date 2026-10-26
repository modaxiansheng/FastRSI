"""Extract factual native transitions; no unexecuted branch or prose labels."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from PhyAgentOS.runtime.policy.msgpack_numpy import unpackb

def reference(path,project):
    raw=path.read_bytes()
    return dict(path=str(path.relative_to(project)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

def main():
    project=Path('/project');out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
    rows=[];episodes=[]
    for name in sys.argv[2:]:
        root=project/'runs'/name
        audit=json.loads((project/'planning'/(name+'_audit')/'report_v2/AUDIT.json').read_text())
        assert audit['status']=='PASSED_NATIVE_BINDING_AUDIT'
        stream=root/'F/stream';case=json.loads((stream/'RESULT.json').read_text())['cases'][0]
        assert case['experience_before']==case['experience_after']
        calls=list(map(json.loads,(stream/'value_service/VALUE_POLICY_CALLS.jsonl').read_text().splitlines()))
        groups={}
        for call in calls:groups.setdefault(call['session_id'],[]).append(call)
        for session in case['sessions']:
            sid=session['session_id'];group=groups[sid]
            if not all(c['enabled'] for c in group):continue
            trace=list(map(json.loads,(stream/'stage_trace'/sid/'TRACE.jsonl').read_text().splitlines()))
            for i,c in enumerate(group):
                request=stream/'value_service'/(c['evidence_name']+'_request.msgpack')
                response=stream/'value_service'/(c['evidence_name']+'_response.msgpack')
                reply=unpackb(response.read_bytes());obs=unpackb(request.read_bytes())
                t=min((r for r in trace if r['method']=='action_chunk' and r['step']>c['step']),key=lambda r:r['step'])
                planned=np.asarray(reply['actions']);submitted=np.asarray(t['actions'],np.float32)
                assert np.array_equal(planned,reply['candidates'][c['selected_index']])
                assert np.array_equal(planned[:len(submitted)],submitted)
                n=t['step']-c['step'];assert 0<n<=len(submitted)
                terminal=i==len(group)-1
                next_call=group[i+1] if not terminal else None
                rows.append(dict(episode=name+'/'+sid,chunk=i,budget=400-c['step'],n=n,terminal=terminal,
                    native_success=bool(t['native_success']),reward=int(terminal and t['native_success']),
                    planned_actions=planned.tolist(),request=reference(request,project),response=reference(response,project),
                    next_request=reference(stream/'value_service'/(next_call['evidence_name']+'_request.msgpack'),project) if next_call else None,
                    next_response=reference(stream/'value_service'/(next_call['evidence_name']+'_response.msgpack'),project) if next_call else None,
                    prefix_feature_already_computed=True,selected_index=c['selected_index'],counterfactual_label=None,
                    qwen_prose_used_for_label=False,actual_native_agent=True,evaluation_snapshot_unchanged=True))
            episodes.append(dict(campaign=name,session_id=sid,native_success=session['result']['metadata']['final_status']['success'],
                semantic_status=session['status'],actions=session['result']['num_steps'],requests=len(group),
                model_sha256=group[0]['model_sha256'],use='eligible_future_development_only_after_this_frozen_batch_is_sealed'))
    (out/'TRANSITIONS.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (out/'EPISODES.json').write_text(json.dumps(episodes,indent=2)+'\n')
    (out/'RESULT.json').write_text(json.dumps(dict(status='FACTUAL_NATIVE_FEEDBACK_READY',episodes=len(episodes),transitions=len(rows),
        extra_robot_actions=0,new_visual_encoding=0,parameter_updates=0,unexecuted_branch_labels=0,
        completed_evaluation_must_not_be_reused_as_independent_evidence_for_future_updates=True),indent=2)+'\n')
    print(json.dumps(dict(episodes=len(episodes),transitions=len(rows))),flush=True)
if __name__=='__main__':main()

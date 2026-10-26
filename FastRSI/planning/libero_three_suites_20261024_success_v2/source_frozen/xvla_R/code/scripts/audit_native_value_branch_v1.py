"""Actual branch labels plus reproduction of the original default continuation."""
import hashlib,json
from pathlib import Path
import numpy as np
from PhyAgentOS.runtime.policy.msgpack_numpy import unpackb
P=Path('/project');O=Path('/check');groups={};checked=0
for slot in range(4):
    root=P/'runs'/f'native_value_branch_gpu{slot}_v1_20261010'
    result=json.loads((root/'RESULT.json').read_text());assert result['status']=='COMPLETED'
    assert all(result['fork_inputs_exact'].values())
    spec=json.loads((root/'SPEC.json').read_text());source=P/spec['source'];plan=json.loads((source/'PLAN.json').read_text())
    stops=json.loads((root/'STOP_RECEIPT.json').read_text());assert all(not s['state']['Running'] and s['state']['Pid']==0 for s in stops)
    raw_source=unpackb((P/plan['source_response']).read_bytes());candidates=raw_source['candidates']
    isolated=result['rows'][0];assert isolated['submitted_sha256']==hashlib.sha256(candidates[spec['arm']].tobytes()).hexdigest()
    for row in result['rows'][1:]:
        index=row['before']//10;request=unpackb((root/f'{index:03d}_request.msgpack').read_bytes())
        response=unpackb((root/f'{index:03d}_response.msgpack').read_bytes());command=np.load(root/f'{index:03d}_command.npz',allow_pickle=False)['actions']
        assert np.array_equal(response['actions'],command)
        noise=np.random.default_rng(np.random.SeedSequence([plan['noise_base'],index,0])).standard_normal((10,32)).astype(np.float32)
        assert response['branch_baseline']['noise_sha256']==hashlib.sha256(noise.tobytes()).hexdigest()
        if spec['arm']==0:
            native_root=(P/plan['source_request']).parents[3]/'N/stream/value_service'
            # source_request is under F/stream/value_service; bind its sibling N.
            native_root=(P/plan['source_request']).parents[3]/'N/stream/value_service'
            oldname=hashlib.sha256(f'net_t64_i{plan["initial"]}'.encode()).hexdigest()[:20]+f'_{index:03d}'
            previous=unpackb((native_root/(oldname+'_request.msgpack')).read_bytes())
            chosen=unpackb((native_root/(oldname+'_response.msgpack')).read_bytes())
            for key in ('observation/image','observation/wrist_image','observation/state'):assert np.array_equal(request[key],previous[key])
            assert np.array_equal(command,chosen['actions'])
        checked+=1
    groups.setdefault(plan['initial'],{})[spec['arm']]=dict(native_success=result['native_success'],actions=result['physical_steps'],
        remaining_actions=result['remaining_actions'],remaining_control_requests=result['remaining_control_requests'],
        candidate_inferences=result['actor_inferences'],seconds=result['seconds_including_close'],
        source_root=str(root.relative_to(P)),model_noise_base=plan['noise_base'])
labels=[]
for initial,arms in groups.items():
    a,b=arms[0],arms[1]
    priority=lambda r:(r['native_success'],-r['remaining_control_requests'])
    preferred=0 if priority(a)>priority(b) else 1 if priority(b)>priority(a) else None
    labels.append(dict(initial=initial,fork_step=10,arms=arms,preferred_observed_candidate=preferred,
        label_rule='Native success first, remaining control requests second; both complete branches actually executed.',
        noisy_single_continuation_evidence_not_expected_value_or_safety_proof=True,development_only=True,
        held_out_generalization_or_learned_gain_claimed=False))
(O/'RESULT.json').write_text(json.dumps(dict(status='AUDITED_REAL_BRANCH_SUPERVISION',checked_continuation_commands=checked,
    default_branch_reproduced_native_rgb_robot_and_actions=True,pairs=labels,parameter_updates=0,
    goal_complete=False),indent=2)+'\n')
print(json.dumps(labels),flush=True)

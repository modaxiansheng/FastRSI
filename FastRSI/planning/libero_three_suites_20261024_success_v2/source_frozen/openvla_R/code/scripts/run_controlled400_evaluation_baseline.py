"""Fixed controlled400 evaluation; original frozenv3 actor and Agent retained."""
import asyncio
import hashlib
import json
import os
import re
import time
import numpy as np
import run_native_baseline as native
from cost_trace import emit, span
from fastrsi.phase_transfer import load_config, phase_window_apply
from fastrsi.phase_transport_priority import make_priority_carrier
from fastrsi.controlled400_evaluation import (PROTOCOL, install_task_scope, install_update_guard,
    protect_inputs, validate_case, validate_phase_case, verify_inputs)

install_task_scope()
install_update_guard()

def make_carrier(config, candidate):
    if config['source_selection']=='oldest_eligible_v1':
        return make_priority_carrier(config, candidate)
    if config['source_selection']=='nearest_eligible_v1':
        from fastrsi.phase_transfer_v3 import make_carrier_v3
        return make_carrier_v3(config, candidate)
    if config['source_selection']=='learned_source_choice_stage3_v1':
        from fastrsi.source_choice_learning_stage3 import make_learned_source_carrier
        return make_learned_source_carrier(config, candidate)
    raise ValueError('Unknown frozen evaluation selection policy')
from fastrsi.feedback_window import FeedbackWindow
from PhyAgentOS.runtime.adapters.openpi.pi05_policy_adapter import OpenPIPi05Adapter
from PhyAgentOS.runtime.sessions.target_session_handle import TargetSessionHandle

original_case=native.run_case


async def run_case(case,accepted):
    config,candidate=load_config(os.environ['FASTRSI_PHASE_CONFIG'],os.environ['FASTRSI_PHASE_CONFIG_SHA256'])
    if os.environ.get('FASTRSI_CONTROLLED400_EVALUATION') != PROTOCOL:
        raise ValueError('Explicit controlled400 evaluation environment required')
    validate_phase_case(case,config)
    # This is the same MODEL_CONFIG file whose SHA the native runner uses for
    # its Agent/Verifier receipts, rather than a synthetic adapter setting.
    paths=[os.environ['FASTRSI_PHASE_CONFIG'],config['artifact'],native.CAMPAIGN/'MODEL_CONFIG.json']
    if config['source_selection']=='learned_source_choice_stage3_v1':
        paths.append(config['decision_model']['artifact'])
    fixed_inputs=protect_inputs(paths,emit)
    carrier=make_carrier(config,candidate);current={};rows=[]
    infer0=native.OpenPIClientPolicyWrapper.infer
    input0=OpenPIPi05Adapter.to_policy_input;output0=OpenPIPi05Adapter.from_policy_output
    action0=TargetSessionHandle.action_chunk
    window0=FeedbackWindow.apply
    def apply(window,chunk,context):return phase_window_apply(window,chunk,context,config,window0)
    def commit(handle,chunk):
        out=action0(handle,chunk);exit_record=out.get('phase_transfer_exit')
        if exit_record is not None:
            if (exit_record.get('session_id')!=config['root_session']
                    or exit_record.get('candidate_sha256')!=config['sha256']):
                raise ValueError('Unbound target carrier exit')
            carrier.force_native=True
            emit(dict(event='phase_transfer_native_handoff',case=case['id'],**exit_record))
        return out
    def bind(adapter,observation,context):
        match=re.fullmatch(r'libero_obs_(\d+)',str(observation['observation_id']))
        if match is None:raise ValueError('Missing physical step binding')
        current.update(session=context['session_id'],step=int(match[1]),observation=observation['observation_id'])
        return input0(adapter,observation,context)
    def infer(policy,observation):
        tick=time.perf_counter();origin='native';status='failed';tag=None
        try:
            # Agent retries in other Sessions stay native and cannot inherit phase state.
            if current['session']==config['root_session']:
                tag=carrier.choose(current['session'],current['step'],observation.get('observation/state'))
            emit(dict(event='phase_transfer_public_input',case=case['id'],runtime_session=current['session'],
                step=current['step'],prompt=observation.get('prompt'),robot_state=observation['observation/state'].tolist(),
                array_bindings={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.tobytes()).hexdigest())
                    for k,v in observation.items() if hasattr(v,'tobytes')},
                image_semantics_extracted=False,held_object_verified=False))
            if tag:
                origin='phase_transfer'
                # Placeholders are decoded only by the explicit target carrier.
                output={'actions':np.zeros((10,7),np.float32),
                    'policy_meta':{'prediction_origin':'explicit_phase_transfer_request','phase_transfer':tag}}
                carrier.record(current['step'],observation['observation/state'],np.zeros((tag['count'],7),np.float32),origin)
            else:
                with span('phase_transfer_native_inference',case=case['id']):output=infer0(policy,observation)
                if current['session']==config['root_session']:
                    carrier.record(current['step'],observation['observation/state'],np.asarray(output['actions'])[:10],origin)
            status='returned';return output
        finally:
            row=dict(event='phase_transfer_choice',case=case['id'],runtime_session=current['session'],step=current['step'],
                origin=origin,status=status,seconds=time.perf_counter()-tick,request=tag,
                config_sha256=os.environ['FASTRSI_PHASE_CONFIG_SHA256'],candidate_sha256=config['sha256'],
                source_sha256=candidate['source_sha256'],carrier_mode=config.get('carrier_mode','dense'),
                experimental_task0_context=True,held_object_verified=False)
            rows.append(row);emit(row)
            with (native.ROOT/'PHASE_TRANSFER_CALLS.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    def decode(adapter,output,context):
        chunk=output0(adapter,output,context);tag=output.get('policy_meta',{}).get('phase_transfer')
        if tag is not None:
            chunk['provenance']={**chunk.get('provenance',{}),'phase_transfer':tag}
        return chunk
    native.OpenPIClientPolicyWrapper.infer=infer;OpenPIPi05Adapter.to_policy_input=bind;OpenPIPi05Adapter.from_policy_output=decode
    TargetSessionHandle.action_chunk=commit
    FeedbackWindow.apply=apply
    try:
        result=await original_case(case,accepted)
        row,next_source=result
        if (row['case']['mode']!='evaluation' or row['experience_before']!=row['experience_after']
                or next_source!=accepted or 'experience_admission' in row):
            raise ValueError('Fixed evaluation changed persistent experience or promoted test output')
        row['controlled400_evaluation']=verify_inputs(fixed_inputs)
        (native.ROOT/case['id']/'RESULT.json').write_text(json.dumps(row,indent=2,default=str)+'\n')
        return row,next_source
    finally:
        native.OpenPIClientPolicyWrapper.infer=infer0;OpenPIPi05Adapter.to_policy_input=input0;OpenPIPi05Adapter.from_policy_output=output0
        TargetSessionHandle.action_chunk=action0
        FeedbackWindow.apply=window0
        input_audit=verify_inputs(fixed_inputs)
        (native.ROOT/'CONTROLLED400_INPUT_AUDIT.json').write_text(json.dumps(input_audit,indent=2)+'\n')
        (native.ROOT/'PHASE_TRANSFER_RECEIPT.json').write_text(json.dumps(dict(
            enabled=config['enabled'],actual_vla_calls=sum(x['origin']=='native' for x in rows),
            failed_vla_calls=sum(x['origin']=='native' and x['status']=='failed' for x in rows),
            carrier_requests=sum(x['origin']=='phase_transfer' and x['status']=='returned' for x in rows),
            used_segments=sorted(carrier.used),training_updates=0,
            source_decisions=getattr(carrier,'decisions',[]),source_selection=config['source_selection'],
            model_decisions=getattr(carrier,'model_decisions',[]),
            phase_progress=[] if not hasattr(carrier,'progress') else carrier.progress.events,
            config_sha256=os.environ['FASTRSI_PHASE_CONFIG_SHA256'],candidate_sha256=config['sha256'],
            source_sha256=candidate['source_sha256'],carrier_mode=config.get('carrier_mode','dense'),
            scope='Experimental public proprio envelope, not object/fixture proof. Native grasp and release; source transport candidate needs real transfer validation. Skipped VLA calls alter subsequent RNG positions.',
            cost_scope='Choice/preparation included in inference wall time; actual target carrier inside environment_execution; source acquisition/extraction costs remain separate.'),indent=2))

native.run_case=run_case
async def main():
    from fastrsi.resident_client import from_environment
    client=from_environment()
    if client is None:
        await native.main()
        return
    from run_resident_native_baseline import run_owned_session,valid_roster
    import signal
    roster=json.loads((native.ROOT.parent/'ROSTER.json').read_text());valid_roster(roster)
    loop=asyncio.get_running_loop();task=asyncio.current_task()
    loop.add_signal_handler(signal.SIGTERM,task.cancel)
    try:
        await run_owned_session(native.main,client,roster,native.ROOT,
            int(os.environ['FASTRSI_RESIDENT_MAX_INFERENCES']),span_factory=span)
    finally:
        loop.remove_signal_handler(signal.SIGTERM)

if __name__=='__main__':asyncio.run(main())

"""Opt-in native registered program consumer; no replacement Agent or policy.

A tagged goal request is a transport placeholder, never a VLA prediction or a
robot action. A matching target-side consumer is required before enabling this
runner. Only initial legal camera evidence determines the geometry once.
"""
import asyncio
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
import re
import threading
import time
import numpy as np


_BOUNDARY_KEY = 'registered_program_boundary_v1'


def _boundary_config(case):
    if _BOUNDARY_KEY not in case:
        return None
    value = case[_BOUNDARY_KEY]
    if (not isinstance(value, dict) or set(value) != {'version', 'step', 'decision'}
            or value.get('version') != _BOUNDARY_KEY or type(value.get('step')) is not int
            or value['step'] != 120 or value.get('decision') not in ('continue', 'handoff')
            or case.get('mode') != 'development'):
        raise ValueError('Invalid development registered program boundary configuration')
    return copy.deepcopy(value)


def _boundary_public_observation(observation):
    summary = {'observation_id': observation['observation_id'], 'rgb': {}}
    for camera in ('front_rgb', 'wrist_rgb'):
        array = np.ascontiguousarray(observation['sensors'][camera]['data'])
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3 or not array.size:
            raise ValueError('Boundary requires current public uint8 HWC RGB')
        summary['rgb'][camera] = {'sha256': hashlib.sha256(array.tobytes()).hexdigest(),
                                 'shape': list(array.shape), 'dtype': str(array.dtype)}
    robot = np.asarray(observation['sensors']['proprio']['data'])
    if robot.shape != (8,) or not np.issubdtype(robot.dtype, np.number) or not np.all(np.isfinite(robot)):
        raise ValueError('Boundary requires finite current public robot8')
    summary['robot8'] = robot.tolist()
    return summary


def install(workspace, case, emit, *, adapter_class=None, policy_class=None):
    from fastrsi.native_goal_component_v1 import extract_component, bind_selection
    from fastrsi.native_component_updates_v1 import registry_hashes
    from fastrsi.goal_skill import marker
    from PhyAgentOS.runtime.schemas import SkillRuntimeDocument
    from PhyAgentOS.runtime.state_io.markdown_yaml import read_yaml_block
    if adapter_class is None:
        from PhyAgentOS.runtime.adapters.openpi.pi05_policy_adapter import OpenPIPi05Adapter
        adapter_class=OpenPIPi05Adapter
    if policy_class is None:
        from PhyAgentOS.runtime.policy.openpi.client import OpenPIClientPolicyWrapper
        policy_class=OpenPIClientPolicyWrapper
    config=case.get('registered_goal_component')
    if config is not True:
        raise ValueError('Explicit registered goal component case required')
    boundary=_boundary_config(case)
    boundary_done=False
    model_sha=case['model_config_sha256'];target_ref='target://libero_real_remote'
    if hashlib.sha256(Path(case['model_config']).read_bytes()).hexdigest()!=model_sha:
        raise ValueError('Current staged model configuration changed')
    workspace=Path(workspace)
    original_input=adapter_class.to_policy_input
    original_output=adapter_class.from_policy_output
    original_infer=policy_class.infer
    local=threading.local();state={'bound':False,'retired':False,'last_step':None,'prepared':None}
    rows=[]
    selection_path=workspace/'reports/NATIVE_GOAL_COMPONENT_SELECTION.json'
    def log(**row):
        row={'event':'registered_goal_component',**row};rows.append(row);emit(row)
    def check_boundary_contract(context):
        if _boundary_config(case) != boundary:
            raise ValueError('Registered program boundary configuration changed')
        if context.get('session_id') != case['id']:
            return
        execution=context.get('execution') or {}
        if (type(execution.get('max_steps')) is not int or execution['max_steps'] != 400
                or type(execution.get('replan_every_steps')) is not int or execution['replan_every_steps'] != 10
                or context.get('task_description') != case['task_description']):
            raise ValueError('Boundary requires unchanged root goal and native 400/H10 execution')
    def bind(adapter,observation,context):
        if boundary is not None:check_boundary_contract(context)
        if getattr(local,'pending',None) is not None:
            raise ValueError('Previous registered inference was not consumed')
        payload=original_input(adapter,observation,context)
        sid=context.get('session_id');match=re.fullmatch(r'libero_obs_(\d+)',str(observation.get('observation_id')))
        if not match:raise ValueError('Missing native physical-step observation binding')
        step=int(match[1])
        if context.get('source_observation_id')!=observation['observation_id']:
            raise ValueError('Context/observation physical step mismatch')
        selected=context.get('skillruntime_ref')
        if sid==case['id'] and not state['bound']:
            state['bound']=True;tick=time.perf_counter()
            try:
                if step!=0:raise ValueError('Program binding requires initial observation')
                before=registry_hashes(workspace)
                skills=SkillRuntimeDocument.model_validate(read_yaml_block(workspace/'SKILLRUNTIME.md')).skillruntimes
                found=[s for s in skills if 'skillruntime://'+s.id==selected]
                if len(found)!=1:raise ValueError('Selected native skill missing or duplicate')
                expected=extract_component(found[0])
                declared=(context.get('policy_action_contract') or {}).get('chunk',{}).get('native_goal_component')
                if expected!=declared:raise ValueError('Actual native context component differs from selected registry')
                prepared=bind_selection(workspace,found[0],expected_goal=context['task_description'],
                    expected_model_config_sha256=model_sha,expected_target_ref=target_ref,
                    current_rgb=observation['sensors']['front_rgb']['data'])
                if prepared is None:raise ValueError('Selected native skill has no goal component')
                if context.get('target_ref')!=target_ref:raise ValueError('Selected target differs from case binding')
                if registry_hashes(workspace)!=before:raise ValueError('Registry changed during program preparation')
                rgb=np.ascontiguousarray(observation['sensors']['front_rgb']['data'])
                document={'version':'native_goal_selection_v1','workspace':str(workspace),
                    'root_session':sid,'session_id':sid,'skillruntime_ref':selected,
                    'target_ref':target_ref,'registry_sha256':before,
                    'initial_observation_id':observation['observation_id'],
                    'initial_rgb':{'sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),
                                   'shape':list(rgb.shape),'dtype':str(rgb.dtype)},
                    'initial_rgb_by_camera':{
                        camera:{'sha256':hashlib.sha256(np.ascontiguousarray(observation['sensors'][sensor]['data']).tobytes()).hexdigest(),
                                'shape':list(np.asarray(observation['sensors'][sensor]['data']).shape),
                                'dtype':str(np.asarray(observation['sensors'][sensor]['data']).dtype)}
                        for camera,sensor in [('agentview_image','front_rgb'),('robot0_eye_in_hand_image','wrist_rgb')]},
                    'declaration':prepared['declaration'],
                    'source_artifact_sha256':prepared['source_artifact_sha256'],
                    'derivation_receipt':prepared['derivation_receipt'],
                    'derived_waypoints':prepared['derived_waypoints']}
                selection_path.parent.mkdir(parents=True,exist_ok=True)
                if selection_path.is_symlink():raise RuntimeError('Selection report cannot be a symlink')
                if selection_path.exists():
                    old_raw=selection_path.read_bytes();old=json.loads(old_raw)
                    if old.get('root_session')==sid:
                        raise RuntimeError('Duplicate root Session selection report')
                    old_sha=hashlib.sha256(old_raw).hexdigest()
                    history=selection_path.parent/'component_selection_history'/ (old_sha+'.json')
                    history.parent.mkdir(parents=True,exist_ok=True)
                    if history.exists():
                        if history.is_symlink() or history.read_bytes()!=old_raw:
                            raise RuntimeError('Selection history SHA collision')
                    else:
                        with history.open('xb') as f:f.write(old_raw)
                    document['inherited_selection_sha256']=old_sha
                raw=json.dumps(document,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
                temporary=None
                try:
                    with tempfile.NamedTemporaryFile(dir=str(selection_path.parent),prefix='.selection-',delete=False) as f:
                        temporary=Path(f.name);f.write(raw);f.flush();os.fsync(f.fileno())
                    os.replace(str(temporary),str(selection_path));temporary=None
                finally:
                    if temporary is not None:temporary.unlink()

                state.update(prepared=prepared,registry=before,skill=selected,
                    selection_sha=hashlib.sha256(raw).hexdigest(),total=len(prepared['derived_waypoints']),
                    action_contract=copy.deepcopy(context.get('policy_action_contract')))
                log(stage='initial_binding',status='accepted',session_id=sid,step=step,
                    selection_path=str(selection_path),selection_sha256=state['selection_sha'],seconds=time.perf_counter()-tick)
            except (ValueError,KeyError,TypeError,OSError) as exc:
                state['retired']=True
                log(stage='initial_binding',status='fallback_native',reason=str(exc),session_id=sid,
                    step=step,seconds=time.perf_counter()-tick)
        if sid==case['id'] and state['prepared'] is not None and (not state['retired'] or boundary is not None):
            if selected!=state['skill'] or registry_hashes(workspace)!=state['registry']:
                raise ValueError('Consumed component changed after initial binding')
            if boundary is not None and context.get('policy_action_contract')!=state['action_contract']:
                raise ValueError('Boundary action contract changed after initial binding')
            if state['last_step'] is not None and step<=state['last_step']:
                raise ValueError('Repeated or reversed physical observation')
            if state.get('expected_next_step') is not None and step!=state['expected_next_step']:
                raise ValueError('Registered program physical step is not contiguous')
        public_summary=None;summary_seconds=0.0
        if boundary is not None and sid==case['id']:
            if step >= 400:raise ValueError('Boundary root inference exceeds remaining native budget')
            if step==120 and state['prepared'] is not None and not state['retired']:
                summary_tick=time.perf_counter()
                public_summary=_boundary_public_observation(observation)
                summary_seconds=time.perf_counter()-summary_tick
        local.pending={'session':sid,'step':step,'payload':payload,'context':copy.deepcopy(context),
                       'public_summary':public_summary,'summary_seconds':summary_seconds}
        return payload
    def infer(policy,payload):
        nonlocal boundary_done
        current=getattr(local,'pending',None)
        if current is None or current['payload'] is not payload:
            raise ValueError('Policy infer bypassed the registered native input')
        local.pending=None
        sid=current['session'];step=current['step'];tick=time.perf_counter();status='failed'
        reuse=(sid==case['id'] and state['prepared'] is not None and not state['retired'] and step<state['total'])
        attempted=False;program_issued=False
        try:
            if boundary is not None:
                check_boundary_contract(current['context'])
                if reuse and step==120:
                    decision_tick=time.perf_counter()
                    if boundary_done or current['public_summary'] is None:
                        raise ValueError('Missing or repeated registered boundary decision')
                    boundary_done=True
                    if boundary['decision']=='handoff':
                        state['retired']=True
                        reuse=False
                    log(stage='program_boundary_decision',session_id=sid,step=step,
                        decision=boundary['decision'],public_observation=current['public_summary'],
                        remaining_native_steps=400-step,permanent_handoff=state['retired'],
                        selection_sha256=state['selection_sha'],
                        config_sha256=hashlib.sha256(json.dumps(boundary,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                        observation_summary_seconds=current['summary_seconds'],
                        seconds=time.perf_counter()-decision_tick,
                        scope='Fixed development operation; not a learned decision or physical completion label')
                elif reuse and step>120 and not boundary_done:
                    raise ValueError('Registered boundary was not consumed')
            if reuse:
                if step%10:raise ValueError('Unaligned registered goal prefix request')
                tag=marker(state['prepared']['source_artifact_sha256'],sid,step,steps=state['total'])
                from fastrsi.dual_anchor_retarget_v1 import _sha
                tag.update(component_selection_sha256=state['selection_sha'],
                           derived_waypoints_sha256=_sha(state['prepared']['derived_waypoints']))
                output={'actions':np.zeros((tag['count'],7),np.float32),'policy_meta':{
                    'prediction_origin':'registered_goal_request_not_policy_inference','goal_skill':tag,
                    'native_goal_component':{'selection_path':str(selection_path),'selection_sha256':state['selection_sha']}}}
                program_issued=True
            else:
                attempted=True
                output=original_infer(policy,payload)
            status='returned'
            if sid==case['id']:
                state['last_step']=step
                state['expected_next_step']=step+tag['count'] if reuse else None
            return output
        finally:
            origin=('registered_program' if reuse else 'native_vla')
            if boundary is not None and not program_issued and not attempted:origin='not_dispatched'
            log(stage='infer_dispatch',session_id=sid,step=step,origin=origin,
                actual_policy_call=(attempted if boundary is not None else not reuse),status=status,seconds=time.perf_counter()-tick)
    def decode(adapter,output,context):
        chunk=original_output(adapter,output,context)
        for key in ('goal_skill','native_goal_component'):
            tag=output.get('policy_meta',{}).get(key)
            if tag is not None:chunk.setdefault('provenance',{})[key]=copy.deepcopy(tag)
        return chunk
    adapter_class.to_policy_input=bind;adapter_class.from_policy_output=decode;policy_class.infer=infer
    def restore():
        adapter_class.to_policy_input=original_input;adapter_class.from_policy_output=original_output;policy_class.infer=original_infer
    return restore,rows


async def main():
    import run_native_baseline as native
    from cost_trace import emit
    original=native.run_case;validate=native.validate_component_selection
    async def run_case(case,accepted):
        config=case.get('registered_goal_component')
        if not config:return await original(case,accepted)
        workspace=native.ROOT/case['id']/'workspace'
        restore,rows=install(workspace,case,emit)
        def authorized(w,planned,anchor):
            return validate(w,planned,anchor,goal_component_case=case)
        native.validate_component_selection=authorized
        try:return await original(case,accepted)
        finally:
            restore();native.validate_component_selection=validate
            path=native.ROOT/case['id']/'REGISTERED_GOAL_CALLS.json';path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps({'rows':rows,'actual_vla_attempts':sum(r.get('actual_policy_call',False) for r in rows),
                'program_requests':sum(r.get('origin')=='registered_program' for r in rows),
                'scope':'Actual selected program requests, not model inference; task quality is in native runtime results.'},indent=2))
    native.run_case=run_case
    try:await native.main()
    finally:native.run_case=original

if __name__=='__main__':asyncio.run(main())

"""Sampler extension around the unchanged registered/native Agent runner.

Case native_sampler_binding + raw SHA must identify the same target config.
Sampling identity is written outside the Agent workspace, never in REQUEST.
"""
import asyncio
import copy
import hashlib
import json
from pathlib import Path


def bind_setup(case, workspace, spec, campaign_name):
    from fastrsi.native_sampler_agent_binding_v1 import VERSION,benchmark_metadata
    from fastrsi.native_sampler_restore_v1 import registration_role
    receipt_path=Path(workspace).parent/'NATIVE_SAMPLER_AGENT_BINDING.json'
    if receipt_path.exists():raise ValueError('Duplicate native sampler setup')
    raw=Path(case['native_sampler_binding']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=case['native_sampler_binding_sha256']:
        raise ValueError('Sampler binding SHA differs')
    config=json.loads(raw)
    if (config.get('version')!=VERSION or Path(config['workspace']).resolve()!=Path(workspace).resolve() or
            config['restore']['root_session']!=case['id'] or spec['session_id']!=case['id'] or
            case.get('init_state_id')!=0 or not config['run_id'].startswith('native_sampler_')):
        raise ValueError('Sampler binding does not identify this root workspace')
    restore=config['restore'];regraw=Path(restore['registration']).read_bytes()
    if hashlib.sha256(regraw).hexdigest()!=restore['registration_sha256']:
        raise ValueError('Sampler registration SHA differs')
    reg=json.loads(regraw);role=registration_role(reg)
    if (case['task_description']!=reg['task_description'] or case['task_id']!=64 or
            spec['execution']['max_steps']!=400 or spec['execution']['replan_every_steps']!=10 or
            spec['execution']['reset_policy']!='session_runner' or spec['verification_profile']!='recovery'):
        raise ValueError('Native sampler task or recovery budget changed')
    if case['mode'] not in ('development','evaluation') or (case['mode']=='development')!=(role=='DEVELOPMENT_INPUT'):
        raise ValueError('Sampler data role and case mode differ')
    if restore['sample_id'] not in ['sample_'+str(s) for s in reg['seeds']]:
        raise ValueError('Sample outside registered partition')
    spec['benchmark']=benchmark_metadata(spec['benchmark'],config['run_id'])
    if case.get('instruction_experience'):
        raise ValueError('Sampler glue has no instruction-experience extension')
    if case.get('session_intent'):
        from fastrsi.session_flow import intent_from_spec
        payload=intent_from_spec(spec,case['task_description'])
    else:
        payload={'session':spec,'goal':case['task_description']}
    (Path(workspace)/'REQUEST.json').write_text(json.dumps(payload,indent=2))
    receipt=dict(version=VERSION,instance_id=None,input_role=role,sample_id=restore['sample_id'],
        root_session=case['id'],binding_sha256=case['native_sampler_binding_sha256'],
        collection_manifest_sha256=restore['collection_manifest_sha256'],run_id=config['run_id'],
        scope='Native sampler extension, not official fixed50 score',maximum_native_replans=2,
        per_session_max_steps=400,maximum_chain_actions=1200,maximum_resets=3,
        root_program_then_pi_same_world=True,recovery_sessions_reset=True)
    with receipt_path.open('x') as f:json.dump(receipt,f,indent=2)
    return receipt


def install_runner(native, flow):
    """Scope public construction's internal-slot projection to pinned roots."""
    original_setup,original_construct=native.setup_workspace,flow.construct
    bound={}
    def setup(folder,case,accepted):
        workspace,spec,previous=original_setup(folder,case,accepted)
        if case.get('native_sampler_binding'):
            bind_setup(case,workspace,spec,native.CAMPAIGN.name)
            if case['id'] in bound:raise ValueError('Duplicate sampler root construction')
            bound[case['id']]=copy.deepcopy(spec['benchmark'])
        return workspace,spec,previous
    def construct(request,targets,skills,sessions):
        sid=request.get('session_id')
        if sid not in bound:return original_construct(request,targets,skills,sessions)
        if request.get('benchmark')!=bound[sid]:raise ValueError('Bound sampler benchmark changed')
        view=copy.deepcopy(targets)
        base=[t for t in view['targets'] if t['id']=='libero_real_remote']
        if len(base)!=1:raise ValueError('Sampler base target absent or duplicate')
        cfg=base[0]['config']
        if (cfg.get('benchmark_name'),cfg.get('task_id'),cfg.get('init_state_id'),cfg.get('max_steps'))!=('libero_90',64,0,400):
            raise ValueError('Sampler physical target changed before public construction')
        # Only the matching view has no official instance. The native registry
        # and actual runtime keep slot0, whose state assignment is bypassed.
        cfg['init_state_id']=None
        return original_construct(request,view,skills,sessions)
    native.setup_workspace=setup;flow.construct=construct
    def restore():
        native.setup_workspace=original_setup;flow.construct=original_construct
    return restore


async def main():
    import run_native_baseline as native
    import run_registered_goal_component_native as registered
    import fastrsi.session_flow as flow
    restore=install_runner(native,flow)
    try:await registered.main()
    finally:restore()

if __name__=='__main__':asyncio.run(main())

"""Bind the actual backend and native control mode to all three system arms."""
import hashlib
import json
import os
from pathlib import Path


def first_budget(task,arm):
    if type(task) is not int or task not in range(10) or arm not in ('P','F','R'):
        raise ValueError('Unregistered task/arm for fixed development roster')
    return 520


def install(native):
    original = native.setup_workspace
    def setup(folder, case, accepted):
        workspace, spec, previous = original(folder, case, accepted)
        model = os.environ['FASTRSI_VLA_MODEL']
        assert model=='pi05'
        policy_id = model+'_libero'
        spec['benchmark']['policy_id'] = policy_id
        spec['timeouts']['execute_timeout_s'] = 3300
        budget=first_budget(case['task_id'],os.environ['SYSTEM_EFFICIENCY_ARM'])
        if os.environ['SYSTEM_EFFICIENCY_ARM']=='R':
            spec['execution']['max_steps']=budget
        targets = native.read_yaml_block(workspace/'TARGETS.md')
        target = targets['targets'][0]
        target['config']['control_mode'] = 'absolute' if model == 'xvla' else 'relative'
        if os.environ['SYSTEM_EFFICIENCY_ARM']=='R':target['config']['max_steps']=budget
        native.write_yaml_block(workspace/'TARGETS.md','Runtime Targets',targets)
        skills = native.read_yaml_block(workspace/'SKILLRUNTIME.md')
        action = skills['skillruntimes'][0]['output_contract']['action']
        if model == 'xvla':
            action.update(action_space_id='libero_xvla_absolute_eef_gripper_v1',
                          representation='absolute_eef_pose_gripper',control_mode='absolute')
        native.write_yaml_block(workspace/'SKILLRUNTIME.md','Runtime Skill Runtimes',skills)
        request = json.loads((workspace/'REQUEST.json').read_text())
        if 'session' in request:
            request['session'] = spec
        else:
            from fastrsi.session_flow import intent_from_spec
            request = intent_from_spec(spec,case['task_description'])
        (workspace/'REQUEST.json').write_text(json.dumps(request,indent=2))
        receipt = {'model':model,'policy_id':policy_id,'control_mode':target['config']['control_mode'],
                   'legacy_wire_adapter_alias':'pi05_libero_remote',
                   'alias_scope':'Generic OpenPI-compatible observation/action wire adapter; not the loaded model identity.',
                   'registered_first_budget':budget,'budget_scope':'Uniform520 first attempt; original child520/max2 recovery unchanged. Root begin resets post-load RNG; child handoff keeps natural advanced stream.',
                   'bindings_sha256':{name:hashlib.sha256((workspace/name).read_bytes()).hexdigest()
                                      for name in ('TARGETS.md','SKILLRUNTIME.md','REQUEST.json')}}
        (folder/'VLA_BINDING.json').write_text(json.dumps(receipt,indent=2))
        return workspace,spec,previous
    native.setup_workspace = setup

"""PI05 R task8 first100, other9 first320 / all children520; natural RNG retained."""
def expected_context_budget(ctx):
    from pi_vla_binding_v8 import first_budget
    benchmark=ctx.get('benchmark') or {}
    task=benchmark.get('task_index');initial=benchmark.get('instance_id')
    if type(initial) is not int or initial not in range(50) or benchmark.get('suite_id')!='libero_10':
        raise ValueError('Unregistered fixed development suite/initial state')
    root='system_t'+str(task)+'_i'+str(initial)
    sid=ctx.get('session_id')
    if sid==root:return first_budget(task,'R')
    if sid in (root+'_replan_1',root+'_replan_1_replan_2'):
        first_budget(task,'R')
        return 520
    raise ValueError('Unregistered root/child session binding')

def install(verifier_class,model,arm):
    if model!='pi05' or arm not in ('P','F','R'):raise ValueError('Unregistered model/arm')
    if arm!='R':return {'changed':False,'reason':'Native P/F budgets retained'}
    original=verifier_class._build_replanned_session
    def build(self,session,description):
        child=original(self,session,description)
        child.execution.max_steps=520
        return child
    verifier_class._build_replanned_session=build
    from PhyAgentOS.runtime.targets.remote.libero.proxy import LiberoRemoteTargetProxy
    original_config=LiberoRemoteTargetProxy._with_libero_config
    def config(self,ctx):
        merged=original_config(self,ctx)
        steps=ctx['execution']['max_steps']
        if type(steps) is not int or steps not in (100,320,520):raise ValueError('Unregistered physical budget')
        if steps!=expected_context_budget(ctx):raise ValueError('Physical budget does not match registered task/attempt rule')
        merged['libero']['max_steps']=steps
        merged['metadata']['libero']['max_steps']=steps
        return merged
    LiberoRemoteTargetProxy._with_libero_config=config
    return {'changed':True,'first_task8_steps':100,'first_other_steps':320,'child_steps':520,'max_replans':2,'scope':'Fixed known-development task-conditioned budget; each independent root begin restores post-load RNG, native recovery handoff does not reset; all paid attempts counted'}

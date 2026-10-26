"""Explicit PRO variant binding; preserve base suite names for policy normalization."""
import hashlib,json,os
from pathlib import Path
ROOT=Path('/eval')
def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def install():
    from libero.libero import benchmark
    suite=os.environ['LIBEROPRO_BASE_SUITE'];variant=os.environ['LIBEROPRO_VARIANT']
    index=json.loads((ROOT/'ASSET_INDEX_FINAL.json').read_text())
    rows=sorted((r for r in index if r['suite']==suite and r['variant']==variant),key=lambda r:r['task_id'])
    assert len(rows)==10 and [r['task_id'] for r in rows]==list(range(10))
    for r in rows:
        assert file_sha(ROOT/'assets'/r['bddl_rel'])==r['bddl_sha256']
        assert file_sha(ROOT/'assets'/r['init_rel'])==r['init_sha256']
    class BoundSuite:
        name=suite
        def __init__(self,*a,**kw):
            self.n_tasks=10
            self.tasks=[benchmark.Task(name=r['task_name'],language=r['language'],problem='Libero',problem_folder=r['official_suite'],bddl_file=Path(r['bddl_rel']).name,init_states_file=Path(r['init_rel']).name) for r in rows]
        def get_task(self,i):return self.tasks[i]
        def get_num_tasks(self):return 10
        def get_task_init_states(self,i):
            import torch
            return torch.load(str(ROOT/'assets'/rows[i]['init_rel']))
        def get_task_bddl_file_path(self,i):return str(ROOT/'assets'/rows[i]['bddl_rel'])
    benchmark.BENCHMARK_MAPPING[suite]=BoundSuite
    return rows

def budget_compatibility():
    """Only adapt uniform520-specific PI05 guards to official suite limits."""
    if os.environ.get('FASTRSI_VLA_MODEL')!='pi05':return
    suite=os.environ['LIBEROPRO_BASE_SUITE'];limit={'libero_spatial':220,'libero_object':280,'libero_goal':300,'libero_10':520}[suite]
    import pi_vla_binding_v8 as binding
    def first_budget(task,arm):
        if type(task) is not int or task not in range(10) or arm not in ('P','F','R'):raise ValueError('Invalid task/arm')
        return limit
    binding.first_budget=first_budget
    import multi_vla_binding
    multi_vla_binding.first_budget=first_budget
    import pi_recovery_budget_v8 as recovery
    def expected_context_budget(ctx):
        b=ctx.get('benchmark') or {};t=b.get('task_index');i=b.get('instance_id')
        if b.get('suite_id')!=suite or type(i) is not int or i not in range(50):raise ValueError('Invalid suite/init')
        first_budget(t,'R');root='system_t'+str(t)+'_i'+str(i)
        registered=[root]
        for attempt in range(1,6):registered.append(registered[-1]+'_replan_'+str(attempt))
        if ctx.get('session_id') not in registered:raise ValueError('Invalid attempt identity')
        return limit
    recovery.expected_context_budget=expected_context_budget
    original_install=recovery.install
    if suite=='libero_10':return
    def install_budget(verifier_class,model,arm):
        if model!='pi05' or arm not in ('P','F','R'):raise ValueError('Invalid PI05 binding')
        if arm!='R':return {'changed':False,'scope':'native official suite budget'}
        from PhyAgentOS.runtime.targets.remote.libero.proxy import LiberoRemoteTargetProxy
        original_child=verifier_class._build_replanned_session
        def child(self,session,description):
            value=original_child(self,session,description);value.execution.max_steps=limit;return value
        verifier_class._build_replanned_session=child
        original_config=LiberoRemoteTargetProxy._with_libero_config
        def config(self,ctx):
            value=original_config(self,ctx)
            if ctx['execution']['max_steps']!=expected_context_budget(ctx):raise ValueError('Suite budget mismatch')
            value['libero']['max_steps']=limit;value['metadata']['libero']['max_steps']=limit;return value
        LiberoRemoteTargetProxy._with_libero_config=config
        return {'changed':True,'compatibility_only':True,'official_suite':suite,'first_steps':limit,'child_steps':limit}
    recovery.install=install_budget

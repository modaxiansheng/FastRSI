"""Bounded task-table experiment and existing budget-gate dispatch integration."""
import hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fastrsi import task_window, validation_budget as budget

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pair_record(root, pair, arms=('N','F'), cost_compatible=True, *, arm_paths=None, record_schema='feedback_window'):
    if record_schema not in ('feedback_window','action_window_v1'):raise ValueError('Unknown original trace schema')
    if arm_paths is not None and set(arm_paths)!=set(arms):raise ValueError('Explicit arm paths must match the two arms')
    result={};sources={};cases=[];initial=[];models=[]
    for arm,mode in zip(arms,('fixed10','fixed5')):
        group=(root/arm_paths[arm]).resolve() if arm_paths is not None else pair/arm
        if not group.resolve().is_relative_to(root.resolve()):raise ValueError('Arm source outside project')
        rp=group/'stream/RESULT.json';sp=group/'STATUS.json'
        rs=read(rp);row=rs['cases'][0];s=read(sp);c=row['case'];cases.append(c)
        assert rs['status']==s['status']=='COMPLETED' and row['chain_status']=='PASSED'
        assert c['mode']=='development' and c['max_steps']==520 and c['suite']=='libero_10'
        if record_schema=='feedback_window':
            actual=row.get('task_window_decision',row.get('window_learning_decision',{})).get('selected_mode',c.get('feedback_window_mode'))
            assert actual==mode
        native=row['native_terminal_by_session'];assert len(native)==1
        fact=next(iter(native.values()));assert fact['native_success']==row['environment_success']
        events=group/'stream/OS_COST_EVENTS.jsonl';ev=[json.loads(l) for l in events.read_text().splitlines() if l.strip()]
        choices=[e for e in ev if e.get('event')=='feedback_window_choice'];inputs=[e for e in ev if e.get('event')=='actual_policy_input']
        assert inputs
        if record_schema=='feedback_window':
            assert len(choices)==len(inputs) and all(e['mode']==mode for e in choices)
        assert all(e['prompt']==c['task_description'] for e in inputs)
        ep=group/'stream'/c['id']/'workspace/artifacts/runtime'/c['id']/'episode.json'
        episode=read(ep);assert episode['success']==fact['native_success'] and episode['num_steps']==fact['steps']
        extra=[]
        if record_schema=='action_window_v1':
            assert not choices and not c.get('feedback_window_mode')
            chunks=episode['metadata']['final_status']['episode_summary']['chunks']
            size=10 if mode=='fixed10' else 5
            assert len(chunks)==len(inputs) and sum(x['executed_steps'] for x in chunks)==episode['num_steps']
            cursor=0
            for chunk in chunks:
                assert chunk['first_step']==cursor+1 and chunk['requested_steps']==min(size,520-cursor)
                assert chunk['action_shape']==[chunk['requested_steps'],7]
                assert 0<chunk['executed_steps']<=chunk['requested_steps']
                cursor+=chunk['executed_steps']
            if mode=='fixed10':
                assert not c.get('action_window_artifact') and not row.get('action_window_decision')
                assert not any(e.get('event')=='action_window_choice' for e in ev)
            else:
                assert row['action_window_decision']['action']=='short_window'
                commits=[e for e in ev if e.get('event')=='action_window_choice']
                assert len(commits)==len(inputs)
                assert all(e['original_steps']==10 and e['submitted_steps']==chunk['requested_steps']
                           and e['step']==chunk['first_step']-1 for e,chunk in zip(commits,chunks))
                rel=c['action_window_artifact'];assert rel.startswith('/project/')
                artifact=(root/rel[len('/project/'):]).resolve();assert artifact.is_relative_to(root.resolve())
                assert sha(artifact)==c['action_window_sha256']==row['action_window_decision']['sha256']
                proposal=read(artifact)
                assert (proposal['operation'],proposal['steps'],proposal['goal'])==('execute_prefix',5,c['task_description'])
                assert proposal['scope']['model_sha']==sha(group/'MODEL_CONFIG.json')
                extra.append(artifact)
        images=sorted((ep.parent/'rgb').glob('initial_*.png'));assert images
        initial.append({x.name:sha(x) for x in images});models.append(sha(group/'MODEL_CONFIG.json'))
        paths=[rp,sp,events,ep,group/'MODEL_CONFIG.json']+images+extra
        sources.update({str(x.relative_to(root)):sha(x) for x in paths})
        result[mode]={'native_success':fact['native_success'],'seconds':s['full_wall_seconds'],'vla_calls':len(inputs),'steps':fact['steps'],'semantic_verdicts':[o['verdict'] for o in row['outcomes']],'admission':row['experience_admission']['status']}
    assert initial[0]==initial[1] and models[0]==models[1]
    assert all(cases[0][k]==cases[1][k] for k in ('task_description','task_id','init_state_id','max_steps'))
    return {**result,'goal':cases[0]['task_description'],'task':cases[0]['task_id'],'init':cases[0]['init_state_id'],'phase':'development','evidence_id':str(pair.relative_to(root)),'cost_compatible':cost_compatible,'sources':sources}

def continuation(root,run,protocol,index):
    if protocol.get('budget_policy_version')=='window_budget_scoped_v3':
        from window_budget_continuation import continuation_scoped
        return continuation_scoped(root,run,protocol,index)
    if protocol.get('budget_policy_version')=='window_budget_v2':
        from window_budget_continuation import continuation_v2
        return continuation_v2(root,run,protocol,index)
    # This is evaluated BEFORE the optional next pair creates services or dispatches workers.
    fact=pair_record(root,run/('pair'+str(index-1)))
    candidate=protocol['task_window_sha256'];scope=budget.digest({'goal':fact['goal'],'model':sha(run/'MODEL_CONFIG.json'),'cost_unit':'two_gpu_reservation_seconds'})
    state=budget.initial(candidate,scope)
    a,b=fact['fixed10'],fact['fixed5']
    resolved=all(v['native_success'] and v['semantic_verdicts']==['success'] for v in (a,b))
    state=budget.observe(state,{'candidate_sha256':candidate,'scope_sha256':scope,'mode':'development','pair_id':fact['evidence_id'],'init_id':fact['init'],'native_task_seconds':a['seconds'],'candidate_task_seconds':b['seconds'],'project_source_run_seconds':a['seconds']+b['seconds'],'quality_resolved':resolved,'sources':fact['sources']})
    request={'candidate_sha256':candidate,'scope_sha256':scope,'operation':'matched_development_pair','mode':'development','init_id':protocol['first_batch'][index]['init_state_id'],'remaining_uses':protocol['remaining_uses'],'remaining_budget_seconds':protocol['lifecycle_budget_seconds']-a['seconds']-b['seconds'],'estimated_next_pair_seconds':750,'maximum_next_pair_seconds':1700}
    decision=budget.decide(state,request)
    receipt=budget.dispatch(state,request,decision,lambda req:{'authorized_next_loop_index':index,'wall_cap_per_arm_seconds':850})
    return {**receipt,'state':state,'request':request,'units':'two_gpu_reservation_seconds; per-arm lifecycle sum, not parallel wall','actual_worker_dispatch_is_next_loop':True,'estimated_saving_claim':None}

def prepare(root,out):
    specs=[('feedback_window_v1_20261003/pair1',('N','F'),False),('feedback_window_known_v1_20261003/pair0',('N','F'),False),('feedback_window_structures_v1_20261003/pair0',('N','F'),True),('feedback_window_drawer_retry_v1_20261003/pair0',('N','F'),True),('window_learning_v1_20261003/pair0',('R','F'),True),('window_learning_v2_20261003/pair0',('N','F'),True),('window_learning_v2_20261003/pair1',('N','F'),True)]
    pairs=[pair_record(root,root/'runs'/name,arms,compatible) for name,arms,compatible in specs]
    table=task_window.fit(pairs,phase='development');out.mkdir(parents=True,exist_ok=True)
    (out/'TASK_TABLE.json').write_text(json.dumps(table,indent=2))
    (out/'HISTORY.json').write_text(json.dumps({'pairs':pairs,'paid_history_lifecycle_seconds':sum(p[m]['seconds'] for p in pairs for m in ('fixed10','fixed5')),'older_mug_scope':'Native quality evidence only; older report-routing cost excluded from new-entry cost fitting','no_free_history':True},indent=2))
    return table
if __name__=='__main__':
    out=ROOT/'planning/task_window_v1_20261003';table=prepare(ROOT,out)
    roster=read(ROOT/'planning/feedback_window_v1_20261003/PROTOCOL.json')['roster']
    cases=[next(c for c in roster if c['task_id']==4 and c['init_state_id']==i) for i in (5,2)]
    assert task_window.select(table,cases[0]['task_description'])['selected_mode']=='fixed5'
    protocol={'phase':'Paid development discrimination and frozen task-table transfer diagnostic; not learned net benefit','task_window_study':True,'task_window_artifact':str((out/'TASK_TABLE.json').relative_to(ROOT)),'task_window_sha256':sha(out/'TASK_TABLE.json'),'active_arms':['N','F'],'arms':{'N':'fixed10','F':'fixed5'},'first_batch':cases,'public_report_destination_binding':True,'remaining_uses':1,'lifecycle_budget_seconds':3400,'bootstrap_pairs':1,'maximum_pairs':2,'bootstrap_reason':'One predeclared unused mug development initial state tests historical native quality preference under common current full-OS entry; research acquisition expense, no assumed financial payoff.','continuation':'Existing finite-horizon budget gate before second pair; no automatic validation loop','sampling_steps':10,'task_budget':520,'evaluation':'No reserved final instances; table immutable during sessions; no production default promotion; narrative admission unchanged'}
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2));print(json.dumps(table['table'],indent=2))

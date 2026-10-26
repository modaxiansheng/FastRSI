"""Read-only H/W diagnosis. No trajectory splicing or learned-gain attribution."""
import argparse, hashlib, json, math
from pathlib import Path
from analyze_resident_full_os import analyze_case, service_index, known_sum, read_json
from analyze_structural_window import input_issues, quality_state, begin_receipt

ARMS={'A':(10,10),'B':(20,10),'C':(20,20)}

def finite_prediction(actions,h):
    return isinstance(actions,list) and len(actions)==h and all(isinstance(a,list) and len(a)==7 and all(type(v) in (int,float) and math.isfinite(v) for v in a) for a in actions)

def trajectory(details,h,w,max_steps=520):
    counts={k:len(details[k]) for k in ('inputs','choices','submitted')}; issues=[]
    if not counts['inputs'] or len(set(counts.values()))!=1:issues.append('missing_or_unequal_trajectory')
    for i in range(min(counts.values())):
        inp,choice,actual=[details[k][i] for k in ('inputs','choices','submitted')]
        bad=input_issues(inp)
        step,n=choice.get('step'),choice.get('submitted_steps')
        if choice.get('mode')!='fixed_commitment' or choice.get('requested_steps')!=w:bad.append('wrong_commitment')
        if not finite_prediction(choice.get('predicted_actions'),h):bad.append('wrong_full_prediction')
        if type(step) is not int or not 0<=step<max_steps:bad.append('bad_observation_step')
        elif n!=min(w,max_steps-step):bad.append('bad_prefix_length')
        if choice.get('original_steps')!=h:bad.append('wrong_original_horizon')
        if type(n) is not int or not 1<=n<=w or actual.get('actions')!=choice.get('predicted_actions',[])[:n]:bad.append('not_exact_submitted_prefix')
        if type(step) is not int or type(actual.get('step')) is not int or type(n) is not int or not step<actual['step']<=step+n:bad.append('bad_executed_step')
        if choice.get('source_observation_id')!='libero_obs_'+str(step):bad.append('wrong_observation_binding')
        goal=details.get('session_goals',{}).get(actual.get('session_id'))
        if inp.get('prompt')!=goal or not goal:bad.append('wrong_native_session_goal')
        issues.extend({'index':i,'reason':v} for v in bad)
    if details.get('source_errors'):issues.append('source_errors')
    return {'status':'PASSED' if not issues else 'NOT_ESTABLISHED','counts':counts,'issues':issues}

def initial_pair(left,right,require_prediction_equal):
    issues=[]
    if not left.get('inputs') or not right.get('inputs') or not left.get('choices') or not right.get('choices'):
        return {'status':'NOT_ESTABLISHED','issues':['missing_initial']}
    a,b=left['inputs'][0],right['inputs'][0]
    for key in ('keys','prompt','array_bindings','robot_state'):
        if a.get(key)!=b.get(key):issues.append('initial_'+key+'_mismatch')
    issues+=input_issues(a)+input_issues(b)
    if any(d['choices'][0].get('step')!=0 for d in (left,right)):issues.append('not_from_initial_state')
    equal=left['choices'][0].get('predicted_actions')==right['choices'][0].get('predicted_actions')
    if require_prediction_equal and not equal:issues.append('same_H20_initial_prediction_mismatch')
    return {'status':'PASSED' if not issues else 'NOT_ESTABLISHED','issues':issues,'first_prediction_equal':equal,'prediction_equality_required':require_prediction_equal}

def analyze(run):
    errors,sources=[],{}; protocol=read_json(run/'PROTOCOL.json',errors,sources); status=read_json(run/'STATUS.json',errors,sources)
    branches={}; details={}
    binding_names=('MODEL_CONFIG.json','MODEL_BINDING.json','CORE_SOURCE_BINDING.json','FLOW.json','EXPERIENCE_BINDING.json')
    root_bindings={r:read_json(run/r,errors,sources) for r in binding_names}
    for arm,(h,w) in ARMS.items():
        folder=run/arm;state=read_json(folder/'STATUS.json',errors,sources)
        services=service_index(folder/'pool/qwen/service_calls',errors,sources); cases=[]
        startup=read_json(folder/'pool/pi05/resident_receipts/STARTUP.json',errors,sources)
        binding_issues=[]
        meta=startup.get('metadata',{});argv=startup.get('argv',{})
        expected={'action_horizon':h,'action_dim':7,'torch_seed':7,'torch_compile_disabled':True,'policy_config':'pi05_libero','checkpoint_dir':'/model'}
        if any(meta.get(k)!=v for k,v in expected.items()):binding_issues.append('resident_configuration_mismatch')
        if argv.get('pytorch_device')!='cuda:0' or argv.get('default_prompt') is not None:binding_issues.append('unexpected_device_or_prompt')
        if h==20 and argv.get('experimental_action_horizon')!=20:binding_issues.append('H20_not_explicit')
        launch=read_json(folder/'pool/pi05/LAUNCH_pi05.json',errors,sources)
        reference=read_json(run.parent/'pilot_20261001/LAUNCH_pi05_v2.json',errors,sources)
        # Native launch mounts checkpoint at /model; compare its actual source.
        model_mounts=lambda v:[v[i+1] for i,x in enumerate(v[:-1]) if x=='-v' and ':/model:' in v[i+1]]
        if not model_mounts(launch.get('argv',[])) or model_mounts(launch.get('argv',[]))!=model_mounts(reference.get('argv',[])):binding_issues.append('checkpoint_mount_mismatch')
        for index in range(len(protocol['cases'])):
            group=folder/('pair'+str(index))/'N'
            summary,trace,bound=analyze_case(group,services,'R'); sources.update(bound)
            raw=read_json(group/'stream/RESULT.json',errors,sources);row=(raw.get('cases') or [{}])[0]
            trace['session_goals']={s['session_id']:s['task_description'] for s in row.get('sessions',[]) if s.get('session_id')}
            case_binding_issues=list(binding_issues)
            for rel in binding_names:
                if read_json(group.parent/rel,errors,sources)!=root_bindings[rel]:case_binding_issues.append('source_binding_mismatch:'+rel)
            fields=('suite','task_id','init_state_id','task_description','max_steps','mode')
            if any(summary.get('case',{}).get(k)!=protocol['cases'][index].get(k) for k in fields):case_binding_issues.append('case_differs_from_protocol')
            if summary.get('case',{}).get('terminal_evidence_view') is not True:case_binding_issues.append('terminal_protocol_mismatch')
            if row.get('horizon_commitment_binding')!={'inference_horizon':h,'commit_steps':w}:case_binding_issues.append('H_W_result_binding_mismatch')
            summary['configuration_issues']=case_binding_issues
            summary.update(arm=arm,inference_horizon=h,commit_steps=w,VLA_calls=len(trace['inputs']),experience_admission=row.get('experience_admission'),native_window_attempts=row.get('native_window_attempts',[]),actual_native_session_count=len(row.get('sessions',[])))
            summary['resident_inferences']=(summary.get('resident_session',{}).get('status_before_release',{}).get('active') or {}).get('inferences')
            if summary['resident_inferences']!=len(trace['inputs']):summary['configuration_issues'].append('server_inference_count_differs_from_recorded_calls')
            summary['trajectory_audit']=trajectory(trace,h,w)
            admission_path=group/'stream'/summary['case'].get('id','_unknown')/'admission/ADMISSION.json'
            summary['admission_receipt']=read_json(admission_path,errors,sources)
            summary['quality']=quality_state(summary)
            native=summary.get('native_terminal') or []
            summary['executed_steps']=sum(n['steps'] for n in native) if native and all(type(n.get('steps')) is int for n in native) else None
            summary['costs']['experience_admission']=summary['costs']['os_spans']['completed_seconds_by_name'].get('experience_admission')
            summary.pop('evaluation_persistence_unchanged',None)
            summary['accepted_source']=row.get('accepted_source')
            expected=(group.parent/'initial_experience').resolve().as_posix().split('/runs/',1)[-1]
            summary['independent_initial_lineage']=str(row.get('accepted_source')).split('/runs/',1)[-1]==expected
            cases.append(summary);details[arm,index]=trace
        branches[arm]={'status':state.get('status'),'full_two_gpu_branch_seconds':state.get('full_wall_seconds'),'model_pool_startup_seconds':state.get('pool_startup_seconds'),'cases':cases,
            'hot_worker_seconds':known_sum([c['costs'] for c in cases],'worker_full_wall_seconds'),
            'all_qwen_service_calls':len(services),'all_qwen_service_seconds':known_sum([s['receipt'] for s in services],'seconds')}
    comparisons=[]
    for i in range(len(protocol['cases'])):
        for a,b in [('A','B'),('B','C'),('A','C')]:
            x,y=[branches[k]['cases'][i] for k in (a,b)]
            paired=initial_pair(details[a,i],details[b,i],a=='B' and b=='C')
            problems=list(paired['issues'])+x['configuration_issues']+y['configuration_issues']
            if begin_receipt(x).get('rng_hashes')!=begin_receipt(y).get('rng_hashes') or not begin_receipt(x).get('rng_hashes'):problems.append('RNG_start_mismatch')
            if x['initial_experience']!=y['initial_experience'] or not x['initial_experience']['actual_equals_manifest']:problems.append('initial_experience_mismatch')
            for v in (x,y):
                if not v['independent_initial_lineage'] or v['chain_status']!='PASSED' or v['trajectory_audit']['status']!='PASSED' or v['resident_session'].get('release_confirmed') is not True:problems.append('incomplete_native_execution_binding')
            delta={}
            for key in ('VLA_calls','executed_steps'):
                delta[key]=y[key]-x[key] if all(type(v[key]) in (int,float) for v in (x,y)) else None
            for key in ('worker_full_wall_seconds','os_execution','policy_client','semantic_verification','native_terminal_report','experience_admission'):
                av,bv=x['costs'].get(key),y['costs'].get(key);delta[key]=bv-av if all(type(v) in (int,float) for v in (av,bv)) else None
            comparisons.append({'case':i,'contrast':a+'_to_'+b,'validity':'PASSED' if not problems else 'NOT_ESTABLISHED','issues':problems,'initial_pair':paired,'environment_success':[x['environment_success'],y['environment_success']],'quality':[x['quality'],y['quality']],'observed_second_minus_first':delta})
    if errors:
        for comparison in comparisons:
            comparison['validity']='NOT_ESTABLISHED'
            comparison['issues'].append('source_read_or_binding_errors')
    return {'status':status,'protocol':protocol,'branches':branches,'comparisons':comparisons,'errors':errors,'sources':sources,
        'analyzer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Six development episodes, static inference/execution configuration diagnosis; not learned benefit, not held-out quality or stable speedup.',
        'accounting':'Campaign is parallel wall. Branch seconds reserve two GPUs each. Startup is included in branch wall. Hot case and nested spans overlap; never add them together. All failures retained.'}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    out=analyze(args.run);args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps({'campaign_status':out['status'].get('status'),'comparisons':[{k:c[k] for k in ('case','contrast','validity','environment_success','observed_second_minus_first')} for c in out['comparisons']],'errors':out['errors']},ensure_ascii=False,indent=2))

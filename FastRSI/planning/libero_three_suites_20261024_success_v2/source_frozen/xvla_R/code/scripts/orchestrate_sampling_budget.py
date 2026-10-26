"""At most two real paired development probes; continuation is budget-gated."""
import fcntl,hashlib,json,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
import orchestrate_full_os as life
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fastrsi.validation_budget import initial,observe,decide,dispatch,digest

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tree(p):return {str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file()}

def main():
    name='sampling_budget_cache_v1_20261002';p=life.PROJECT;source=p/'runs/shared_flow_net_parallel_v2_20261002'
    lock=open(p/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=p/'runs'/name;run.mkdir(exist_ok=False);started=time.time()
    absolute=min(started+1100,read(p/'runs/representative_native_v1_20261002/PROTOCOL.json')['campaign_absolute_deadline'])
    status={'status':'RUNNING','started_at':started,'pairs':[]};life.save(run/'STATUS.json',status)
    life.CAMPAIGN=run;life.deadline=absolute
    def interrupted(n,f):raise KeyboardInterrupt('Bounded probe interrupted '+str(n))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        assert absolute-started>=1000,'No new allocation after original study deadline'
        life.save(run/'PRIOR_ATTEMPT.json',read(p/'runs/sampling_budget_v2_20261002/STATUS.json'))
        life.gate(run)
        openpi=life.CORE.parent/'openpi'
        source_binding={}
        for rel,expected in [('src/openpi/policies/policy.py','a2ac5236e23b11d0f5a840d4a8fef148d89dba921d5452ec5dfcd5a6ffa46f20'),('src/openpi/models_pytorch/pi0_pytorch.py','dcd7b6f508808ce4b7170a1720a4c62c6da04edae3652d16c94a469dca924e16')]:
            actual=sha(openpi/rel);assert actual==expected;source_binding[rel]=actual
        assert sha(life.CORE/'PhyAgentOS/runtime/skillruntime/policy/openpi.py')=='ec691c814f7ed1ff92c90f5d031654b4624e36d9d2b1776ee1906b9765c2e4d5'
        life.save(run/'SAMPLER_SOURCE_BINDING.json',source_binding)
        for rel in ['scripts','src/fastrsi','tests','configs']:shutil.copytree(p/rel,run/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        life.save(run/'CODE_MANIFEST.json',tree(run/'code'))
        roster=read(p/'planning/representative_coverage_v1/PROPOSED_ROSTER.json')
        next_init=next(c['init_state_id'] for c in roster['streams'][1]['cases'] if c['task_id']==0)
        assert next_init!=45
        candidate={'operation':'OpenPI sample_actions num_steps 10 -> 5','weights':'unchanged pi05_libero','task':0,'bootstrap_init':44,'conditional_next_init':next_init}
        state=initial(digest(candidate),digest({'protocol':'sampling_budget_cache_v1','common_counter_binding_sha':sha(p/'src/fastrsi/native_binding.py'),'common_lifecycle_sha':sha(p/'scripts/orchestrate_full_os.py'),'model':read(source/'MODEL_BINDING.json'),'initial_experience':read(source/'EXPERIENCE_BINDING.json')}))
        life.save(run/'CANDIDATE.json',candidate)
        life.save(run/'PLAN.json',{'scope':'Prospective numerical-inference candidate probe on full native OS; not main RSI ABC or learned efficacy',
          'bootstrap_reason':'Single follow-up with common verified offline token cache after asymmetric token-count delay was found. Previous runs preserved and not pooled. Observed VLA calls consume measurable time; native sampler supports num_steps=10. Test whether 5 reduces complete cost without quality loss, not assume quality or 30% potential.',
          'operations':candidate,'max_pairs':1,'absolute_deadline':absolute,'pair_wall_cap_seconds':850,
          'same':['weights','task goal','initial images','step and recovery caps','native feedback triggers','required verification/report','public Session constructor','per-arm resource quota'],
          'resources':{'N':[0,1],'F':[2,3]},'initialization':'Fresh model seed 7 and independent same accepted experience for each arm; no inter-arm transfer.',
          'evaluation':'Development only; final reserved test pool untouched. No evaluation results feed selection.',
          'continuation_horizon':20,'horizon_scope':'Predeclared exploratory upper workload of 20 future comparable uses, not a committed deployment or established generalization. Report sensitivity and no payback claim.',
          'decision_cost_unit':'seconds of reserved two-GPU service capacity: per-arm lifecycle wall; paired validation sum of arm lifecycle walls. Concurrent wall reported separately.',
          'policy':'Existing fixed mean-gain economic rule; no learning advantage claim. Quality unresolved means STOP. No default promotion.',
          'first_pair':'First sampler test is explicit paid bootstrap, not silently requested by zero-evidence policy.'})
        def run_pair(index,init_id,wall_cap):
            pair=run/('pair'+str(index));pair.mkdir();tick=time.time();deadline=min(absolute,tick+wall_cap)
            life.gate(pair)
            for port in [49600,49602,49610,49620]:
                import socket
                with socket.socket() as s:assert s.connect_ex(('127.0.0.1',port))!=0
            for rel in ['MODEL_CONFIG.json','MODEL_BINDING.json','CORE_SOURCE_BINDING.json','FLOW.json','EXPERIENCE_BINDING.json']:shutil.copyfile(source/rel,pair/rel)
            shutil.copytree(source/'initial_experience',pair/'initial_experience');assert tree(pair/'initial_experience')==read(pair/'EXPERIENCE_BINDING.json')
            plan={'cases':[{'suite':'libero_10','task_id':0,'init_state_id':init_id,'max_steps':520,'mode':'development','task_description':'put both the alphabet soup and the tomato sauce in the basket'}]}
            life.save(pair/'PLAN.json',plan);life.save(pair/'PROTOCOL.json',{'absolute_deadline':deadline,'scope':'Paid native OS development with isolated experience; full outcomes kept'})
            workers=[];result={'status':'RUNNING','started_at':tick,'init_id':init_id,'groups':[]};life.save(pair/'STATUS.json',result)
            try:
                for arm in ['N','F']:
                    group=pair/arm;shutil.copytree(run/'code',group/'code')
                    if arm=='F':
                        for rel in ['scripts/run_native_baseline.py','scripts/qwen_agent_server.py']:
                            f=group/'code'/rel;s=f.read_text()
                            for port in [49500,49502,49510,49520]:s=s.replace(str(port),str(port+100))
                            f.write_text(s)
                    life.save(group/'CODE_MANIFEST.json',tree(group/'code'))
                    log=open(group/'controller.log','x')
                    proc=subprocess.Popen(['python3',str(run/'code/scripts/run_sampling_probe_worker.py'),name+'/pair'+str(index),arm],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    workers.append((arm,proc));life.save(group/'DISPATCH.json',{'pid':proc.pid,'started_at':time.time()})
                while any(proc.poll() is None for _,proc in workers):
                    if time.time()>deadline:raise TimeoutError('Real paired driver wall cap reached')
                    if any(proc.poll() not in (None,0) for _,proc in workers):raise RuntimeError('Worker failed; stop both and preserve costs')
                    time.sleep(2)
                assert all(proc.returncode==0 for _,proc in workers)
                assert tree(pair/'initial_experience')==read(pair/'EXPERIENCE_BINDING.json')
                result['status']='COMPLETED'
            except BaseException:result.update(status='FAILED',error=traceback.format_exc());raise
            finally:
                for _,proc in workers:
                    if proc.poll() is None:proc.send_signal(signal.SIGTERM)
                for arm,proc in workers:
                    try:proc.wait(timeout=40)
                    except subprocess.TimeoutExpired:result['cleanup_attention']='Worker cleanup exceeded bound; inspect owned containers'
                    if (pair/arm/'STATUS.json').exists():result['groups'].append(read(pair/arm/'STATUS.json'))
                result.update(finished_at=time.time(),full_wall_seconds=time.time()-tick);life.save(pair/'STATUS.json',result);status['pairs'].append(result);life.save(run/'STATUS.json',status)
            return {'pair':str(pair.relative_to(run)),'status':result['status']}
        def feedback(index):
            pair=run/('pair'+str(index));reports=[read(pair/a/'stream/RESULT.json') for a in ['N','F']];rows=[r['cases'][0] for r in reports]
            images=[];sources={};budgets=[]
            for arm,row in zip(['N','F'],rows):
                group=pair/arm;folder=group/'stream'/row['case']['id'];images.append(tree(folder/'workspace/artifacts/runtime'/row['case']['id']/'rgb'))
                events=[json.loads(x) for x in (group/'stream/SAMPLING_EVENTS.jsonl').read_text().splitlines()]
                calls=[e for e in events if e['event']=='sample_actions'];assert calls
                assert {e['num_steps'] for e in calls}=={10 if arm=='N' else 5};assert all(e['shape'][1:]==[10,32] for e in calls)
                os_events=[json.loads(x) for x in (group/'stream/OS_COST_EVENTS.jsonl').read_text().splitlines()]
                prompts=[e['prompt'] for e in os_events if e.get('event')=='actual_policy_input'];assert prompts and set(prompts)=={row['case']['task_description']}
                for f in [group/'stream/RESULT.json',group/'STATUS.json',group/'stream/SAMPLING_EVENTS.jsonl',group/'stream/OS_COST_EVENTS.jsonl']:
                    sources[str(f.relative_to(p))]=sha(f)
            initial_images=[{k:v for k,v in d.items() if k.startswith('initial_')} for d in images];assert initial_images[0] and initial_images[0]==initial_images[1]
            good=all(r['chain_status']=='PASSED' and r['environment_success'] is True and r['outcomes'] and all(o.get('verdict')=='success' for o in r['outcomes']) and r['terminal_report']['read_coverage']['complete'] and r['experience_admission']['status']=='ADMITTED' for r in rows)
            return {'pair_id':'pair'+str(index),'init_id':rows[0]['case']['init_state_id'],'mode':'development',
               'candidate_sha256':state['candidate_sha256'],'scope_sha256':state['scope_sha256'],'sources':sources,
               'native_task_seconds':reports[0]['seconds'],'candidate_task_seconds':reports[1]['seconds'],
               'project_source_run_seconds':sum(read(pair/a/'STATUS.json')['full_wall_seconds'] for a in ['N','F']),
               'quality_resolved':good,'quality_scope':'One matched development pair, not statistical noninferiority or guaranteed semantic accuracy'}
        run_pair(0,44,850);state=observe(state,feedback(0));life.save(run/'STATE_AFTER_P0.json',state)
        estimate=state['observations'][0]['project_source_run_seconds']
        request={'mode':'development','operation':'matched_development_pair','candidate_sha256':state['candidate_sha256'],'scope_sha256':state['scope_sha256'],
          'init_id':next_init,'remaining_uses':20,'remaining_budget_seconds':0,
          'estimated_next_pair_seconds':estimate,'maximum_next_pair_seconds':1700}
        request['budget_note']='Single predeclared follow-up pair exhausted this branch allocation; no automatic third/fourth pair.'
        life.save(run/'REQUEST.json',request);decision=decide(state,request);life.save(run/'DECISION.json',decision)
        receipt=dispatch(state,request,decision,lambda job:run_pair(1,job['init_id'],job['maximum_next_pair_seconds']/2))
        life.save(run/'DISPATCH_RECEIPT.json',receipt)
        if receipt['executor_calls']:
            state=observe(state,feedback(1));life.save(run/'STATE_AFTER_P1.json',state)
        status['status']='COMPLETED';status['decision']=decision;status['real_additional_pair_calls']=receipt['executor_calls']
    except BaseException:status.update(status='FAILED',error=traceback.format_exc())
    finally:
        status.update(finished_at=time.time(),full_wall_seconds=time.time()-started);life.save(run/'STATUS.json',status);print(json.dumps(status),flush=True)
    if status['status']!='COMPLETED':raise SystemExit(1)
if __name__=='__main__':main()

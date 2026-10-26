"""Two-arm prospective staged-flow capability gate, followed by one independent reuse pair."""
import concurrent.futures,copy,fcntl,hashlib,json,shutil,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
import orchestrate_full_os as life
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fastrsi.staged_flow import proposal,update,GOAL
from fastrsi.validation_policy import UNIT
from fastrsi.validation_budget import digest

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tree(p):return {str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file()}

def main():
    name='staged_flow_v1_20261003';p=life.PROJECT;run=p/'runs'/name
    lock=open(p/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run.mkdir(exist_ok=False);started=time.time();life.deadline=started+1800
    life.CAMPAIGN=run;life.BASE='/project/runs/'+name
    status={'status':'RUNNING','started_at':started};life.save(run/'STATUS.json',status)
    for rel in ['scripts','src/fastrsi','tests','configs']:
        shutil.copytree(p/rel,run/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    life.save(run/'CODE_MANIFEST.json',tree(run/'code'))
    source=p/'runs/shared_flow_net_parallel_v2_20261002'
    for rel in ['MODEL_CONFIG.json','MODEL_BINDING.json','CORE_SOURCE_BINDING.json','FLOW.json','EXPERIENCE_BINDING.json']:
        shutil.copyfile(source/rel,run/rel)
    flow=proposal(sha(run/'MODEL_CONFIG.json'))
    candidate=flow;scope={'goal':GOAL,'max_steps':520,'native_sampling':10,'source':str(source)}
    def case(init,mode):return dict(suite='libero_10',task_id=4,init_state_id=init,mode=mode,max_steps=520,task_description=GOAL)
    life.save(run/'PROTOCOL.json',{'candidate':candidate,'scope':scope,'deadline':life.deadline,
      'development':case(31,'development'),'evaluation':case(39,'evaluation'),
      'candidate_source':'Observed old task4/init31 failure; human-authored stage allocation, not learned discovery.',
      'phase':'First 260 executed steps white mug only, then original complete goal until 520. Original scorer and Session goal unchanged.',
      'replan':'Native failure/replan triggers unchanged; any extra reset or aggregate steps >520 invalidates this candidate screen.',
      'acceptance':'Both chains and admissions pass, native fails, candidate succeeds with semantic success under same initial images and one reset. Then freeze artifact before new Session evaluation.',
      'resources':'Each arm 2 GPUs, equal model/seed/limits; parallel arms. All current samples are single-seed diagnostics.',
      'accounting':'All real branches, service startup, verifier, native learning/reporting, failures and teardown included; no claimed unexecuted counterfactual.',
      'evaluation':'Only after development admission. No final results enter candidate state. If no win, stop without more parameter search.'})
    def interrupt(n,f):raise KeyboardInterrupt('Study interrupted '+str(n))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    def pair(index,c,slot):
        folder=run/('pair'+str(index));folder.mkdir();tick=time.time();deadline=min(life.deadline,tick+850)
        for rel in ['MODEL_CONFIG.json','MODEL_BINDING.json','CORE_SOURCE_BINDING.json','FLOW.json','EXPERIENCE_BINDING.json']:
            shutil.copyfile(run/rel,folder/rel)
        shutil.copytree(source/'initial_experience',folder/'initial_experience')
        life.save(folder/'STAGE_FLOW.json',flow)
        assert tree(folder/'initial_experience')==read(folder/'EXPERIENCE_BINDING.json')
        life.save(folder/'PLAN.json',{'cases':[c]});life.save(folder/'PROTOCOL.json',{'slot':slot,'absolute_deadline':deadline})
        result={'status':'RUNNING','started_at':tick,'case':c,'slot':slot};life.save(folder/'STATUS.json',result)
        workers=[]
        try:
            for arm in ('N','F'):
                group=folder/arm;shutil.copytree(run/'code',group/'code')
                shift=slot*1000+(100 if arm=='F' else 0)
                for rel in ['scripts/run_native_baseline.py','scripts/qwen_agent_server.py']:
                    f=group/'code'/rel;s=f.read_text()
                    for port in (49500,49502,49510,49520):s=s.replace(str(port),str(port+shift))
                    f.write_text(s)
                life.save(group/'CODE_MANIFEST.json',tree(group/'code'))
                with (group/'controller.log').open('x') as log:
                    proc=subprocess.Popen(['python3',str(run/'code/scripts/run_staged_flow_worker.py'),name+'/pair'+str(index),arm],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                workers.append((arm,proc));life.save(group/'DISPATCH.json',{'pid':proc.pid,'started_at':time.time()})
            while any(proc.poll() is None for _,proc in workers):
                if time.time()>deadline:raise TimeoutError('Pair deadline')
                if any(proc.poll() not in (None,0) for _,proc in workers):raise RuntimeError('Worker failure')
                time.sleep(2)
            assert all(proc.returncode==0 for _,proc in workers)
            result['status']='COMPLETED'
        except BaseException:result.update(status='FAILED',error=traceback.format_exc());raise
        finally:
            for _,proc in workers:
                if proc.poll() is None:proc.send_signal(signal.SIGTERM)
            for _,proc in workers:
                try:proc.wait(timeout=45)
                except subprocess.TimeoutExpired:result['cleanup_attention']='Inspect exact worker and owned containers'
            result.update(full_wall_seconds=time.time()-tick,finished_at=time.time())
            life.save(folder/'STATUS.json',result)
        reports={a:read(folder/a/'stream/RESULT.json')['cases'][0] for a in ('N','F')}
        source_hashes={};images={};cost={}
        for arm,row in reports.items():
            group=folder/arm;workspace=group/'stream'/row['case']['id']/'workspace'
            images[arm]={f.name:sha(f) for f in (workspace/'artifacts/runtime'/row['case']['id']/'rgb').glob('initial_*.png')}
            assert images[arm]
            assert row['terminal_report']['prefetch_enabled'] is False
            assert row['case']['task_description']==c['task_description']
            cost[arm]=read(group/'STATUS.json')['full_wall_seconds']
            for f in [group/'STATUS.json',group/'stream/RESULT.json',group/'stream/OS_COST_EVENTS.jsonl']:
                source_hashes[str(f.relative_to(p))]=sha(f)
        assert images['N']==images['F']
        def good(r):return r['chain_status']=='PASSED' and r['environment_success'] is True and r['outcomes'] and all(o.get('verdict')=='success' for o in r['outcomes']) and r['terminal_report']['read_coverage']['complete'] and (c['mode']=='evaluation' or r['experience_admission']['status']=='ADMITTED')
        quality='pass' if all(good(r) for r in reports.values()) else 'unresolved'
        if good(reports['N']) and not good(reports['F']):quality='candidate_failure'
        row={'pair_id':'pair'+str(index),'instance_id':f"libero_10/t{c['task_id']}/i{c['init_state_id']}",
             'mode':c['mode'],'candidate_sha256':digest(candidate),'scope_sha256':digest(scope),
             'cost_unit':UNIT,'baseline_cost':cost['N'],'candidate_cost':cost['F'],'paid_cost':sum(cost.values()),
             'measurement':'valid','quality':quality,'sources':source_hashes}
        life.save(folder/'FEEDBACK.json',row);return row
    def batch(items):
        life.gate(run)
        for port in [base+s for base in (49500,49502,49510,49520) for s in (0,100,1000,1100)]:
            with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',port))!=0,'Port in use'
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(pair,index,c,slot) for slot,(index,c) in enumerate(items)]
            return [future.result() for future in futures]
    def facts(index,arm):
        folder=run/('pair'+str(index))/arm;result=folder/'stream/RESULT.json';r=read(result)['cases'][0]
        w=folder/'stream'/r['case']['id']/'workspace';ep=w/'artifacts/runtime'/r['case']['id']/'episode.json'
        events=[json.loads(line) for line in (folder/'stream/TARGET_COST_EVENTS.jsonl').read_text().splitlines()]
        own=[e for e in events if e.get('event')=='native_observation' and str(e.get('session_id','')).startswith(r['case']['id'])]
        terminal=r['native_terminal_by_session']
        return {'mode':r['case']['mode'],'goal':r['case']['task_description'],
          'steps':sum(x['steps'] for x in terminal.values()),'resets':sum(e['method']=='reset' for e in own),
          'initial_images':{f.name:sha(f) for f in (ep.parent/'rgb').glob('initial_*.png')},
          'chain_passed':r['chain_status']=='PASSED','admitted':r.get('experience_admission',{}).get('status')=='ADMITTED',
          'success':r['environment_success'],'verdict':r['outcomes'][-1]['verdict'],
          'seconds':read(folder/'STATUS.json')['full_wall_seconds'],
          'sources':{str(result.relative_to(p)):sha(result),str(ep.relative_to(p)):sha(ep)}}
    try:
        cpu=run/'cpu';cpu.mkdir();life.gate(cpu)
        command='source /opt/dtk/env.sh && cd '+life.BASE+'/code/tests && /project/envs/os-v1/bin/python -m unittest test_staged_flow test_terminal_results test_native_isolation test_native_memory_budget -v'
        args=life.cpu_args('yuchen-fastrsi-'+name+'-cpu',cpu)+['-e','FASTRSI_NATIVE_RUN=/tmp/staged-flow-cpu',life.IMAGE,'timeout','150','bash','-lc',command]
        cid=life.launch('cpu',args,cpu);life.wait_exit(cid,160);life.save(cpu/'STOP_RECEIPT.json',life.stop_all());status['cpu']='PASSED'
        development=batch([(0,case(31,'development'))]);status['development']=development
        flow=update(flow,facts(0,'N'),facts(0,'F'));life.save(run/'LEARNED_FLOW.json',flow)
        status['selection']=flow['status'];life.save(run/'STATUS.json',status)
        if flow['status']=='ACCEPTED_DEVELOPMENT':
            before=digest(flow);evaluation=batch([(1,case(39,'evaluation'))])
            assert before==digest(flow)
            life.save(run/'INDEPENDENT_EVALUATION.json',{'records':evaluation,'flow_state_unchanged':True,'sha256':sha(run/'LEARNED_FLOW.json')})
            status['evaluation']=evaluation
        else:status['evaluation']='NOT_RUN_NO_DEVELOPMENT_WIN'
        status['status']='COMPLETED'
    except BaseException:status.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        life.save(run/'STOP_RECEIPT.json',life.stop_all());status.update(finished_at=time.time(),full_wall_seconds=time.time()-started);life.save(run/'STATUS.json',status);print(json.dumps(status),flush=True)

if __name__=='__main__':main()

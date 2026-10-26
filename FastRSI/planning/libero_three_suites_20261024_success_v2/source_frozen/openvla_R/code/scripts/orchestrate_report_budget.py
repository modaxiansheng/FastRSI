"""Real paired full-OS outcomes for a prefix-shared budget mechanism study."""
import concurrent.futures
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import orchestrate_full_os as life
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fastrsi.validation_policy import create,record,freeze,decide,step,UNIT,MODES
from fastrsi.validation_budget import digest

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tree(p):return {str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file()}


def main():
    name='report_budget_v2_20261003';p=life.PROJECT;run=p/'runs'/name
    lock=open(p/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run.mkdir(exist_ok=False);started=time.time();life.deadline=started+3600
    prior=p/'runs/report_budget_v1_20261003'
    if prior.exists():
        assert read(prior/'STATUS.json')['status']=='FAILED'
        life.deadline=min(life.deadline,read(prior/'PROTOCOL.json')['absolute_deadline'])
        life.save(run/'PRIOR_ATTEMPT.json',read(prior/'STATUS.json'))
    life.CAMPAIGN=run;life.BASE='/project/runs/'+name
    status={'status':'RUNNING','started_at':started,'phases':[]};life.save(run/'STATUS.json',status)
    source=p/'runs/shared_flow_net_parallel_v2_20261002'
    for rel in ['scripts','src/fastrsi','tests','configs']:
        shutil.copytree(p/rel,run/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    life.save(run/'CODE_MANIFEST.json',tree(run/'code'))
    for rel in ['MODEL_CONFIG.json','MODEL_BINDING.json','CORE_SOURCE_BINDING.json','FLOW.json','EXPERIENCE_BINDING.json']:
        shutil.copyfile(source/rel,run/rel)
    roster=read(p/'planning/representative_coverage_v1/PROPOSED_ROSTER.json')
    def case(stream,task,mode):
        c=copy.deepcopy(next(c for c in roster['streams'][stream]['cases'] if c['task_id']==task))
        c.update(mode=mode,max_steps=520);return c
    development=[case(1,t,'development') for t in [2,8,1,4]]
    evaluation=[case(2,t,'evaluation') for t in [2,8]]
    candidate={'operation':'fresh terminal read scheduling before first report-model call',
               'source_sha256':sha(run/'code/src/fastrsi/terminal_results.py'),
               'unchanged':['VLA sampling 10','task goal','native AgentLoop','required verification','fresh canonical evidence','report write','memory triggers']}
    scope={'model':read(run/'MODEL_BINDING.json'),'experience':read(run/'EXPERIENCE_BINDING.json'),
           'protocol':'report_budget_v1','cost_unit':UNIT,'development':development,'evaluation':evaluation}
    cfg=dict(cost_unit=UNIT,prior_mean=0.,prior_sd=120.,observation_sd=60.,
             total_budget=6800.,next_pair_estimate=700.,next_pair_cap=1700.,remaining_uses=100,
             fixed_pairs=4,max_pairs=4,minimum_quality_pairs=2)
    states={m:create(digest(candidate),digest(scope),cfg,m) for m in MODES}
    life.save(run/'PROTOCOL.json',{'candidate':candidate,'scope':scope,'config':cfg,
      'absolute_deadline':life.deadline,'max_real_development_pairs':4,'common_independent_evaluation_pairs':2,
      'initial_pairs':'Two predeclared paid bootstrap pairs, parallel; not data-adaptive launches.',
      'freeze':'Freeze cost selection after first bootstrap record; all arms still record quality and fees.',
      'horizon':'100 is a prospective planning scenario (LIBERO-10 x 10), not committed deployment or observed savings. Report 2/20/100/200 sensitivity. Priors/noise are declared uncalibrated pilot assumptions.',
      'comparison':'Same initial experience; common-prefix and identical requested branch outcomes shared to reduce research bill. Record per-policy attributed costs separately. Not independent method cold starts or main RSI ABC.',
      'feedback':'Full native OS dev feedback retained in isolated workspaces. Budget states carry over; OS skill snapshots fixed for this mechanism diagnostic.',
      'evaluation':'Fixed common snapshots, native close/background isolation, no result feedback into budget state.',
      'resources':'Each execution arm has two GPUs and identical limits; two pairs may occupy 8 idle cards for throughput.',
      'quality':'Common environment, verifier, complete report evidence and admission checks; independent reports inspected. Not 2pp noninferiority.',
      'budget_termination':'New approved study; old campaign deadlines and results remain untouched.'})
    def signal_stop(n,f):raise KeyboardInterrupt('Study interrupted '+str(n))
    signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
    def pair(index,c,slot):
        folder=run/('pair'+str(index));folder.mkdir();tick=time.time();deadline=min(life.deadline,tick+850)
        for rel in ['MODEL_CONFIG.json','MODEL_BINDING.json','CORE_SOURCE_BINDING.json','FLOW.json','EXPERIENCE_BINDING.json']:
            shutil.copyfile(run/rel,folder/rel)
        shutil.copytree(source/'initial_experience',folder/'initial_experience')
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
                    proc=subprocess.Popen(['python3',str(run/'code/scripts/run_report_budget_worker.py'),name+'/pair'+str(index),arm],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
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
            assert row['terminal_report']['prefetch_enabled']==(arm=='F')
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
    try:
        cpu=run/'cpu';cpu.mkdir();life.gate(cpu)
        command='source /opt/dtk/env.sh && cd '+life.BASE+'/code/tests && /project/envs/os-v1/bin/python -m unittest test_terminal_prefetch test_terminal_results test_native_isolation test_native_memory_budget -v'
        args=life.cpu_args('yuchen-fastrsi-'+name+'-cpu',cpu)+['-e','FASTRSI_NATIVE_RUN=/tmp/report-budget-cpu',life.IMAGE,'timeout','150','bash','-lc',command]
        cid=life.launch('cpu',args,cpu);life.wait_exit(cid,160);life.save(cpu/'STOP_RECEIPT.json',life.stop_all())
        status['cpu']='PASSED';life.save(run/'STATUS.json',status)
        bootstrap=batch(list(enumerate(development[:2])))
        for row in bootstrap:
            for mode in MODES:
                states[mode]=record(states[mode],row)
                if mode=='frozen' and states[mode]['frozen_record_ids'] is None:states[mode]=freeze(states[mode])
        life.save(run/'STATES_AFTER_BOOTSTRAP.json',states)
        status['phases'].append('bootstrap_complete');life.save(run/'STATUS.json',status)
        for index in (2,3):
            c=development[index];request={'mode':'development','operation':'matched_development_pair',
                'candidate_sha256':digest(candidate),'scope_sha256':digest(scope),
                'instance_id':f"libero_10/t{c['task_id']}/i{c['init_state_id']}",'bootstrap':False}
            decisions={mode:decide(states[mode],request) for mode in MODES}
            life.save(run/f'REQUEST_{index}.json',request)
            life.save(run/f'DECISIONS_{index}.json',decisions)
            cache=[]
            def execute(job):
                if not cache:cache.extend(batch([(index,c)]))
                return copy.deepcopy(cache[0])
            for mode in MODES:
                events=[];states[mode],receipt=step(states[mode],request,execute,events.append)
                life.save(run/f'{mode}_STEP_{index}.json',{'events':events,'receipt':receipt,'shared_outcome':True})
            life.save(run/f'STATES_AFTER_{index}.json',states)
            if not cache:break
        # Common fixed independent checks are paid by every policy; never fed back.
        state_sha=digest(states);life.save(run/'FROZEN_POLICY_STATES.json',states)
        final=batch([(4,evaluation[0]),(5,evaluation[1])])
        assert digest(states)==state_sha
        life.save(run/'INDEPENDENT_EVALUATION.json',{'records':final,'policy_state_unchanged':True,
            'state_sha256':state_sha,'scope':'Two common fixed-snapshot diagnostic pairs; not 2pp noninferiority or independent cold-start RSI'})
        status.update(status='COMPLETED',policy_paid_development_cost={m:sum(r['paid_cost'] for r in s['records']) for m,s in states.items()},
                      common_independent_cost=sum(r['paid_cost'] for r in final))
    except BaseException:status.update(status='FAILED',error=traceback.format_exc())
    finally:
        life.save(run/'STOP_RECEIPT.json',life.stop_all())
        status.update(finished_at=time.time(),full_wall_seconds=time.time()-started)
        life.save(run/'STATUS.json',status);print(json.dumps(status),flush=True)
    if status['status']!='COMPLETED':raise SystemExit(1)


if __name__=='__main__':main()

"""Two concurrent, isolated native experience streams; common lifecycle and limits."""
import argparse,fcntl,hashlib,json,os,shutil,signal,socket,subprocess,time,traceback
from pathlib import Path
import orchestrate_full_os as life

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tree(p):return {str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file()}

def validate(protocol):
    if protocol['arms']!={'P':{'gpu_base':0,'port_offset':0},'F':{'gpu_base':2,'port_offset':100}}:raise ValueError('Unexpected arm placement')
    if protocol['maximum_campaign_seconds']!=2100 or protocol['maximum_root_episodes']!=6:raise ValueError('Unexpected experiment budget')
    cases=protocol['cases']
    if len(cases)!=3 or [c['mode'] for c in cases]!=['development','development','evaluation']:raise ValueError('Invalid sequence')
    if [(c['task_id'],c['init_state_id']) for c in cases]!=[(0,3),(0,1),(0,0)]:raise ValueError('Changed declared paired instances')
    if any(c.get('session_flow_artifact') or c.get('budget_checkpoint') for c in cases):raise ValueError('Method binding belongs to parent')
    if not all(c['suite']=='libero_10' and c['max_steps']==520 and c['native_experience_maintenance'] is True and c['feedback_window_mode']=='fixed10' for c in cases):raise ValueError('Changed native constraints')
    if protocol['initial_experience']!='empty' or protocol['forced_consolidation'] is not False:raise ValueError('Changed learning initial state')

def cleanup(run, children, name):
    for arm,proc in children:
        if proc.poll() is None:proc.send_signal(signal.SIGTERM)
    pending=[]
    for arm,proc in children:
        try:proc.wait(timeout=50)
        except subprocess.TimeoutExpired:pending.append(arm)
    audit=[]
    for path in run.rglob('LAUNCH_*.json'):
        cid=read(path).get('stdout','').strip()
        if len(cid)!=64 or any(c not in '0123456789abcdef' for c in cid):continue
        info=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
        assert info['Id']==cid and info['Name'].startswith('/yuchen-fastrsi-'+name)
        if info['State']['Running']:
            life.active.append((cid,path.stem,path.parent));life.stop_all()
        info=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
        audit.append({'container':cid,'name':info['Name'],'running':info['State']['Running'],'exit_code':info['State']['ExitCode']})
    life.save(run/'CONTAINER_AUDIT.json',audit)
    if pending or any(v['running'] for v in audit):raise RuntimeError('Cleanup requires attention: '+repr(pending))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);args=ap.parse_args();name=args.name
    if not name.startswith('continuous_comparison_') or not name.replace('_','').isalnum():raise ValueError('Invalid campaign name')
    p=life.PROJECT;protocol=read(p/'planning'/name/'PROTOCOL.json');validate(protocol)
    lock=open(p/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=p/'runs'/name;run.mkdir(exist_ok=False);started=time.time();deadline=started+protocol['maximum_campaign_seconds']
    life.CAMPAIGN=run;life.BASE='/project/runs/'+name;life.deadline=deadline
    state={'status':'PREFLIGHT','started_at':started,'deadline':deadline};children=[]
    def interrupt(n,f):raise KeyboardInterrupt('Interrupted campaign '+str(n))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    life.save(run/'STATUS.json',state)
    try:
        life.gate(run)
        for port in (49600,49602,49610,49620):
            with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',port))!=0
        for rel in ('scripts','src/fastrsi','tests','configs'):
            shutil.copytree(p/rel,run/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        life.save(run/'CODE_MANIFEST.json',tree(run/'code'))
        source=p/'runs/native_stream_v4_20261003'
        for rel in ('MODEL_CONFIG.json','MODEL_BINDING.json','CORE_SOURCE_BINDING.json'):shutil.copyfile(source/rel,run/rel)
        for rel,h in read(run/'CORE_SOURCE_BINDING.json').items():assert sha(life.CORE/'PhyAgentOS'/rel)==h
        shutil.copyfile(p/'runs/shared_flow_net_parallel_v2_20261002/FLOW.json',run/'FLOW.json')
        assert read(run/'FLOW.json')['benchmark_skill_sha256']==sha(life.CORE/'PhyAgentOS/skills/benchmarking/SKILL.md')
        shutil.copyfile(p/'planning/scoped_allocation_v1_20261003/PROTOCOL.json',run/'BUDGET_TEMPLATE.json')
        protocol.update(absolute_deadline=deadline,flow_sha256=sha(run/'FLOW.json'),budget_template_sha256=sha(run/'BUDGET_TEMPLATE.json'))
        life.save(run/'PROTOCOL.json',protocol)
        for arm,placement in protocol['arms'].items():
            group=run/arm;group.mkdir();shutil.copytree(run/'code',group/'code')
            changes={}
            if placement['port_offset']:
                for rel in ('scripts/run_native_baseline.py','scripts/qwen_agent_server.py'):
                    path=group/'code'/rel;before=path.read_text();after=before
                    for port in (49500,49502,49510,49520):after=after.replace(str(port),str(port+placement['port_offset']))
                    inverse=after
                    for port in (49500,49502,49510,49520):inverse=inverse.replace(str(port+placement['port_offset']),str(port))
                    assert inverse==before;compile(after,str(path),'exec');path.write_text(after)
                    changes[rel]={'before':sha(run/'code'/rel),'after':sha(path)}
            life.save(group/'PORT_RELOCATION.json',changes);life.save(group/'CODE_MANIFEST.json',tree(group/'code'))
            for rel in ('MODEL_CONFIG.json','MODEL_BINDING.json','CORE_SOURCE_BINDING.json'):shutil.copyfile(run/rel,group/rel)
            cases=[]
            for i,item in enumerate(protocol['cases']):
                case={**item,'id':'continuous_'+str(i)+'_t0_i'+str(item['init_state_id']),
                      'conversation_key':'cli:'+name+'_'+arm+('_development' if item['mode']=='development' else '_evaluation')}
                if arm=='F':case.update(session_flow_artifact=life.BASE+'/FLOW.json',session_flow_sha256=protocol['flow_sha256'])
                if arm=='F' and i==0:
                    proposal={k:protocol['cases'][1][k] for k in ('mode','suite','task_id','init_state_id','task_description','max_steps')}
                    proposal['id']='optional_fixed5_t0_i1'
                    case['budget_checkpoint']={'template_path':life.BASE+'/BUDGET_TEMPLATE.json','template_sha256':protocol['budget_template_sha256'],'proposal':proposal}
                cases.append(case)
            life.save(group/'ROSTER.json',cases);life.save(group/'PROTOCOL.json',{**protocol,'selected_arm':arm,'cases':cases})
        cpu=run/'cpu';cpu.mkdir()
        modules='test_continuous_comparison test_continuous_budget_checkpoint test_native_queue_schema test_native_maintenance test_native_stream_history test_session_flow test_native_isolation'
        cmd='source /opt/dtk/env.sh && cd '+life.BASE+'/code/tests && /project/envs/os-v1/bin/python -m unittest '+modules+' -v'
        cid=life.launch('cpu',life.cpu_args('yuchen-fastrsi-'+name+'-cpu',cpu)+['-e','FASTRSI_NATIVE_RUN=/tmp/continuous-comparison-cpu',life.IMAGE,'timeout','150','bash','-lc',cmd],cpu)
        life.wait_exit(cid,160);life.save(cpu/'STOP_RECEIPT.json',life.stop_all());state['cpu']='PASSED'
        # Fresh check after CPU setup, immediately before the independent arms.
        life.gate(run)
        for arm in protocol['arms']:
            with (run/arm/'controller.log').open('x') as log:
                proc=subprocess.Popen(['python3',str(run/'code/scripts/run_continuous_comparison_worker.py'),name,arm],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append((arm,proc));life.save(run/arm/'DISPATCH.json',{'pid':proc.pid,'time':time.time(),'arm':arm})
        state['status']='RUNNING';life.save(run/'STATUS.json',state)
        while any(proc.poll() is None for _,proc in children):
            if time.time()>deadline:raise TimeoutError('Campaign time budget exhausted')
            time.sleep(2)
        state['groups']={arm:read(run/arm/'STATUS.json') for arm,_ in children}
        assert all(proc.returncode==0 for _,proc in children) and all(v['status']=='COMPLETED' for v in state['groups'].values()),state['groups']
        state['status']='COMPLETED'
    except BaseException:state.update(status='FAILED',error=traceback.format_exc())
    finally:
        try:cleanup(run,children,name)
        except BaseException:state.update(status='FAILED',cleanup_attention=traceback.format_exc())
        life.save(run/'STOP_RECEIPT.json',life.stop_all());state.update(finished_at=time.time(),full_wall_seconds=time.time()-started);life.save(run/'STATUS.json',state)
        print(json.dumps(state),flush=True)
    if state['status']!='COMPLETED':raise SystemExit(1)
if __name__=='__main__':main()

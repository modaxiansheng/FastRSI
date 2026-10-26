"""Matched fresh/resident two-case OS lifecycle check; independent branch pacing."""
import argparse,fcntl,hashlib,json,os,shutil,signal,socket,subprocess,time,traceback
from pathlib import Path
import orchestrate_full_os as life
from native_service_config import configure_qwen
from orchestrate_resident_services import gpu,configure_context,check_ports

NAME='resident_full_os_v1_20261003'
BINDINGS=('MODEL_CONFIG.json','MODEL_BINDING.json','CORE_SOURCE_BINDING.json','FLOW.json','EXPERIENCE_BINDING.json')

def read(p):return json.loads(p.read_text())
def tree(p):return {str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file()}
def freeze(source,dest):shutil.copytree(source,dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))

def pool(run,branch,deadline):
    result={};cfg=read(run/'MODEL_CONFIG.json');t=time.time()
    for role,receipt,port,index in [('pi05','LAUNCH_pi05_v2.json',22300,2),('qwen','LAUNCH_qwen.json',22310,3)]:
        service=branch/'pool'/role;service.mkdir(parents=True);freeze(run/'code',service/'code')
        configure_context(service,deadline)
        argv=read(life.OLD/receipt)['argv']
        argv=[x.replace(str(life.OLD),str(service)).replace('/project/runs/pilot_20261001',life.BASE) for x in argv]
        if role=='qwen':
            path=service/'code/scripts/qwen_agent_server.py';path.write_text(path.read_text().replace('49510',str(port)))
            argv=configure_qwen(argv,cfg,life)
        else:
            shutil.copytree(life.PROJECT/'models/openpi_data',service/'tokenizer_cache')
            argv=[x.replace(str(life.PROJECT/'models/openpi_data'),str(service/'tokenizer_cache')).replace('49500',str(port)) for x in argv]
            pos=argv.index(life.IMAGE);argv[pos:pos]=['-v',str(life.PROJECT)+':/project:rw']
            argv=[x.replace('PYTHONPATH=','PYTHONPATH='+life.BASE+'/code/src:') if x.startswith('PYTHONPATH=') else x for x in argv]
            argv[-1]=argv[-1].replace('python -m PhyAgentOS.runtime.policy.openpi.native_openpi_server','python '+life.BASE+'/code/scripts/resident_openpi_server.py')+' --receipt-dir /pilot/resident_receipts'
        argv=gpu(argv,index);argv[argv.index('--name')+1]='yuchen-fastrsi-'+NAME+'-pool-'+role
        ix=argv.index(life.IMAGE)
        if argv[ix+1]=='timeout':argv[ix+2]=str(int(deadline-time.time()))
        life.save(service/'CODE_MANIFEST.json',tree(service/'code'))
        result[role]=life.launch(role,argv,service)
    life.health(22300,limit=360);life.health(22310,limit=360)
    return result,time.time()-t


def branch(arm):
    p=life.PROJECT;run=p/'runs'/NAME;protocol=read(run/'PROTOCOL.json');deadline=protocol['deadline']
    folder=run/arm;folder.mkdir(exist_ok=False);configure_context(run,deadline)
    state={'status':'RUNNING','pid':os.getpid(),'started_at':time.time(),'groups':[]};life.save(folder/'STATUS.json',state)
    worker=None
    def interrupt(n,f):raise KeyboardInterrupt('Branch interrupted '+str(n))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        services=None
        if arm=='R':
            services,state['pool_startup_seconds']=pool(run,folder,deadline)
            configure_context(run,deadline)
        for index,case in enumerate(protocol['cases']):
            pair=folder/('pair'+str(index));pair.mkdir()
            for rel in BINDINGS:shutil.copyfile(run/rel,pair/rel)
            shutil.copytree(run/'initial_experience',pair/'initial_experience')
            assert tree(pair/'initial_experience')==read(run/'EXPERIENCE_BINDING.json')
            life.save(pair/'PLAN.json',{'cases':[case]})
            case_protocol={'slot':0,'absolute_deadline':deadline,'diagnostic_trace':True,'control_kind':'feedback_window','arm_modes':{'N':'gripper_transition','R':'gripper_transition'},'port_offset':-27400,'gpu_bases':{'N':0,'R':2},'terminal_evidence_view':True}
            if services:
                case_protocol['resident_services']={'policy_port':22300,'qwen_port':22310,'policy_container':services['pi05'],'qwen_container':services['qwen'],'session_id':NAME+'-'+arm+'-case'+str(index),'max_inferences':400}
            life.save(pair/'PROTOCOL.json',case_protocol)
            group=pair/arm;freeze(run/'code',group/'code');shift=-27400+('N','F','R','S').index(arm)*100
            for rel in ['scripts/run_native_baseline.py','scripts/qwen_agent_server.py']:
                path=group/'code'/rel;text=path.read_text()
                for port in (49500,49502,49510,49520):text=text.replace(str(port),str(port+shift))
                path.write_text(text)
            life.save(group/'CODE_MANIFEST.json',tree(group/'code'))
            name=NAME+'/'+arm+'/pair'+str(index)
            with (group/'controller.log').open('x') as log:
                worker=subprocess.Popen(['python3',str(run/'code/scripts/run_placement_worker.py'),name,arm],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            life.save(group/'DISPATCH.json',{'pid':worker.pid,'started_at':time.time()})
            while worker.poll() is None:
                if time.time()>deadline-30:raise TimeoutError('Branch deadline')
                time.sleep(2)
            assert worker.returncode==0,read(group/'STATUS.json')
            status=read(group/'STATUS.json');assert status['status']=='COMPLETED'
            state['groups'].append({'path':str(group.relative_to(run)),**status});life.save(folder/'STATUS.json',state)
        if arm=='R':
            configure_context(run,deadline)
            probe=folder/'handoff_probe';probe.mkdir()
            cmd='source /opt/dtk/env.sh && python '+life.BASE+'/code/scripts/check_resident_handoff.py --port 22300 --output '+life.BASE+'/R/handoff_probe/replay'
            cid=life.launch('handoff_probe',life.cpu_args('yuchen-fastrsi-'+NAME+'-handoff-probe',probe)+[life.IMAGE,'timeout','100','bash','-lc',cmd],probe)
            life.wait_exit(cid,110)
            result=read(probe/'replay/RESULT.json');assert result['status']=='PASSED';state['handoff_probe_seconds']=result['seconds']
        state['status']='COMPLETED'
    except BaseException:state.update(status='FAILED',error=traceback.format_exc())
    finally:
        if worker is not None and worker.poll() is None:
            worker.send_signal(signal.SIGTERM)
            try:worker.wait(timeout=90)
            except subprocess.TimeoutExpired:state['cleanup_attention']='worker still running; inspect exact IDs'
        life.save(folder/'POOL_STOP_RECEIPT.json',life.stop_all())
        state.update(finished_at=time.time(),full_wall_seconds=time.time()-state['started_at']);life.save(folder/'STATUS.json',state)
    if state['status']!='COMPLETED':raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--branch',choices=['N','R']);args=parser.parse_args()
    if args.branch:return branch(args.branch)
    p=life.PROJECT;run=p/'runs'/NAME
    lock=open(p/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prereq=read(p/'runs/resident_services_v1_20261003/STATUS.json');assert prereq['status']=='COMPLETED'
    assert read(p/'runs/resident_services_v1_20261003/qwen/replay/RESULT.json')['equivalence_passed']
    run.mkdir(exist_ok=False);started=time.time();deadline=started+1900;configure_context(run,deadline)
    state={'status':'RUNNING','started_at':started,'deadline':deadline,'children':[]};life.save(run/'STATUS.json',state)
    for rel in ('scripts','src/fastrsi','tests','configs'):freeze(p/rel,run/'code'/rel)
    life.save(run/'CODE_MANIFEST.json',tree(run/'code'))
    source=p/'runs/shared_flow_net_parallel_v2_20261002'
    for rel in BINDINGS:shutil.copyfile(source/rel,run/rel)
    shutil.copytree(source/'initial_experience',run/'initial_experience')
    protocol=read(p/'planning'/NAME/'PROTOCOL.json');protocol['deadline']=deadline;life.save(run/'PROTOCOL.json',protocol)
    children=[]
    def interrupt(n,f):raise KeyboardInterrupt('Comparison interrupted '+str(n))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        cpu=run/'cpu';cpu.mkdir();life.gate(cpu);check_ports([22100,22102,22110,22120,22300,22302,22310,22320])
        cmd='source /opt/dtk/env.sh && cd '+life.BASE+'/code/tests && python -m unittest test_resident_policy test_resident_client test_resident_worker test_resident_qwen_replay -v'
        cid=life.launch('cpu',life.cpu_args('yuchen-fastrsi-'+NAME+'-cpu',cpu)+[life.IMAGE,'timeout','150','bash','-lc',cmd],cpu)
        life.wait_exit(cid,160);life.save(cpu/'STOP_RECEIPT.json',life.stop_all());state['cpu']='PASSED'
        for arm in ('N','R'):
            with (run/(arm+'_controller.log')).open('x') as log:
                proc=subprocess.Popen(['python3',str(run/'code/scripts/orchestrate_resident_full_os.py'),'--branch',arm],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append((arm,proc));state['children'].append({'arm':arm,'pid':proc.pid})
        life.save(run/'STATUS.json',state)
        while any(proc.poll() is None for _,proc in children):
            if time.time()>deadline:raise TimeoutError('Comparison deadline')
            time.sleep(2)
        state['branches']={arm:read(run/arm/'STATUS.json') for arm,_ in children}
        assert all(proc.returncode==0 for _,proc in children),state['branches']
        state['status']='COMPLETED'
    except BaseException:state.update(status='FAILED',error=traceback.format_exc())
    finally:
        for arm,proc in children:
            if proc.poll() is None:proc.send_signal(signal.SIGTERM)
        for arm,proc in children:
            try:proc.wait(timeout=150)
            except subprocess.TimeoutExpired:state['cleanup_attention']=arm
        life.save(run/'STOP_RECEIPT.json',life.stop_all())
        audit=[]
        for path in run.rglob('LAUNCH_*.json'):
            launch=read(path);cid=launch.get('stdout','').strip()
            if len(cid)!=64 or not all(c in '0123456789abcdef' for c in cid):continue
            info=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
            assert info['Id']==cid and info['Name'].startswith('/yuchen-fastrsi-'+NAME)
            row={'container':cid,'name':info['Name'],'running':info['State']['Running']}
            if row['running']:state['cleanup_attention']='Owned container still running; do not reuse resources before explicit cleanup'
            audit.append(row)
        life.save(run/'FINAL_CONTAINER_AUDIT.json',audit)
        if state.get('cleanup_attention'):state['status']='FAILED'
        state.update(finished_at=time.time(),full_wall_seconds=time.time()-started);life.save(run/'STATUS.json',state)
    if state['status']!='COMPLETED':raise SystemExit(1)

if __name__=='__main__':main()

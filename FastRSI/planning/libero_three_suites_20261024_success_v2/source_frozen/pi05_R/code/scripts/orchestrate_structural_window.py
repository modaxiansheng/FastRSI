"""Eight fixed-window development cases, four resident branches on two hosts."""
import argparse,fcntl,json,os,shutil,signal,shlex,socket,subprocess,time,traceback
from pathlib import Path
import orchestrate_full_os as life
from native_service_config import configure_qwen
from orchestrate_resident_services import gpu,configure_context,check_ports,PEER
from orchestrate_resident_full_os import BINDINGS,read,tree,freeze

NAME='structural_window_v1_20261003'
HOSTS={'187':'j03r4n13','188':'j03r4n14'}
PLACEMENT={'187':[0,1],'188':[2,3]}

def binding(arm):
    if arm not in ('N','F'):raise ValueError('Unknown arm')
    index=('N','F').index(arm)
    return {'policy_port':22100+index*100,'qwen_port':22110+index*100,'gpu_base':index*2,'mode':('fixed10','fixed5')[index]}

def require_host(host):
    assert socket.gethostname()==HOSTS[host], 'Host binding differs'

def pool(run,folder,arm,deadline):
    ports=binding(arm);services={};cfg=read(run/'MODEL_CONFIG.json');started=time.time()
    for role,receipt,port,device in [('pi05','LAUNCH_pi05_v2.json',ports['policy_port'],ports['gpu_base']),('qwen','LAUNCH_qwen.json',ports['qwen_port'],ports['gpu_base']+1)]:
        service=folder/'pool'/role;service.mkdir(parents=True);freeze(run/'code',service/'code');configure_context(service,deadline)
        argv=[x.replace(str(life.OLD),str(service)).replace('/project/runs/pilot_20261001',life.BASE) for x in read(life.OLD/receipt)['argv']]
        if role=='qwen':
            script=service/'code/scripts/qwen_agent_server.py';script.write_text(script.read_text().replace('49510',str(port)));argv=configure_qwen(argv,cfg,life)
        else:
            shutil.copytree(life.PROJECT/'models/openpi_data',service/'tokenizer_cache')
            argv=[x.replace(str(life.PROJECT/'models/openpi_data'),str(service/'tokenizer_cache')).replace('49500',str(port)) for x in argv]
            pos=argv.index(life.IMAGE);argv[pos:pos]=['-v',str(life.PROJECT)+':/project:rw']
            argv=[x.replace('PYTHONPATH=','PYTHONPATH='+life.BASE+'/code/src:') if x.startswith('PYTHONPATH=') else x for x in argv]
            argv[-1]=argv[-1].replace('python -m PhyAgentOS.runtime.policy.openpi.native_openpi_server','python '+life.BASE+'/code/scripts/resident_openpi_server.py')+' --receipt-dir /pilot/resident_receipts'
        argv=gpu(argv,device);label=str(folder.relative_to(run)).replace('/','-').lower();argv[argv.index('--name')+1]='yuchen-fastrsi-'+NAME+'-'+label+'-pool-'+role
        ix=argv.index(life.IMAGE)
        if argv[ix+1]=='timeout':argv[ix+2]=str(int(deadline-time.time()))
        life.save(service/'CODE_MANIFEST.json',tree(service/'code'));services[role]=life.launch(role,argv,service)
    life.health(ports['policy_port'],limit=360);life.health(ports['qwen_port'],limit=360)
    return services,time.time()-started


def branch(host,arm):
    require_host(host);p=life.PROJECT;run=p/'runs'/NAME;protocol=read(run/'PROTOCOL.json');deadline=protocol['deadline']
    folder=run/('host'+host)/arm;folder.mkdir(exist_ok=False);configure_context(run,deadline)
    state={'status':'RUNNING','host':host,'hostname':socket.gethostname(),'arm':arm,'pid':os.getpid(),'started_at':time.time(),'groups':[]};life.save(folder/'STATUS.json',state)
    worker=None
    def interrupted(n,f):raise KeyboardInterrupt('Branch interrupted '+str(n))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        services,state['pool_startup_seconds']=pool(run,folder,arm,deadline);configure_context(run,deadline)
        for index,global_index in enumerate(PLACEMENT[host]):
            case=protocol['cases'][global_index];pair=folder/('pair'+str(index));pair.mkdir()
            for rel in BINDINGS:shutil.copyfile(run/rel,pair/rel)
            shutil.copytree(run/'initial_experience',pair/'initial_experience');assert tree(pair/'initial_experience')==read(run/'EXPERIENCE_BINDING.json')
            life.save(pair/'PLAN.json',{'cases':[case]});ports=binding(arm)
            life.save(pair/'PROTOCOL.json',{'slot':0,'absolute_deadline':deadline,'diagnostic_trace':True,'control_kind':'feedback_window','arm_modes':protocol['arms'],'port_offset':-27400,'gpu_bases':{'N':0,'F':2},'terminal_evidence_view':True,'resident_services':{'policy_port':ports['policy_port'],'qwen_port':ports['qwen_port'],'policy_container':services['pi05'],'qwen_container':services['qwen'],'session_id':NAME+'-'+host+'-'+arm+'-'+str(index),'max_inferences':400}})
            group=pair/arm;freeze(run/'code',group/'code');shift=-27400+('N','F').index(arm)*100
            for rel in ['scripts/run_native_baseline.py','scripts/qwen_agent_server.py']:
                path=group/'code'/rel;text=path.read_text()
                for port in (49500,49502,49510,49520):text=text.replace(str(port),str(port+shift))
                path.write_text(text)
            life.save(group/'CODE_MANIFEST.json',tree(group/'code'));name=str(pair.relative_to(p/'runs'))
            with (group/'controller.log').open('x') as log:worker=subprocess.Popen(['python3',str(run/'code/scripts/run_placement_worker.py'),name,arm],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            life.save(group/'DISPATCH.json',{'pid':worker.pid,'host':host,'started_at':time.time()})
            while worker.poll() is None:
                if time.time()>deadline-30:raise TimeoutError('Branch deadline')
                time.sleep(2)
            status=read(group/'STATUS.json');state['groups'].append({'path':str(group.relative_to(run)),**status});life.save(folder/'STATUS.json',state)
            assert worker.returncode==0 and status['status']=='COMPLETED',status
        state['status']='COMPLETED'
    except BaseException:state.update(status='FAILED',error=traceback.format_exc())
    finally:
        if worker is not None and worker.poll() is None:
            worker.send_signal(signal.SIGTERM)
            try:worker.wait(timeout=90)
            except subprocess.TimeoutExpired:state['cleanup_attention']='worker cleanup not confirmed'
        life.save(folder/'POOL_STOP_RECEIPT.json',life.stop_all());state.update(finished_at=time.time(),full_wall_seconds=time.time()-state['started_at']);life.save(folder/'STATUS.json',state)
    if state['status']!='COMPLETED':raise SystemExit(1)


def preflight(host):
    require_host(host);run=life.PROJECT/'runs'/NAME;folder=run/('host'+host);folder.mkdir(exist_ok=True)
    configure_context(run,read(run/'PROTOCOL.json')['deadline']);life.gate(folder)
    check_ports([base+offset for base in (22100,22200) for offset in (0,2,10,20)])
    for receipt in ('LAUNCH_pi05_v2.json','LAUNCH_qwen.json','LAUNCH_target.json'):
        argv=read(life.OLD/receipt)['argv'];images=[x for x in argv if x.startswith('sha256:')];assert len(images)==1
        result=life.command(['docker','image','inspect',images[0]]);assert result['returncode']==0
    life.save(folder/'READY.json',{'host':host,'hostname':socket.gethostname(),'checked_at':time.time(),'status':'PASSED'})


def process_matches(pid, expected):
    path=Path('/proc')/str(pid)/'cmdline'
    if not path.exists():return False
    return path.read_bytes().decode().strip('\0').split('\0')==expected


def cleanup_host(host):
    """Reconcile only this campaign's exact process/launch receipts on this host."""
    require_host(host);run=life.PROJECT/'runs'/NAME;folder=run/('host'+host);signals=[]
    for arm in ('N','F'):
        status=folder/arm/'STATUS.json'
        if status.exists():
            row=read(status);pid=row.get('pid');expected=['python3',str(run/'code/scripts/orchestrate_structural_window.py'),'--host',host,'--arm',arm]
            if row.get('status')=='RUNNING' and pid and process_matches(pid,expected):os.kill(pid,signal.SIGTERM);signals.append(pid)
    until=time.time()+120
    while time.time()<until and any(path.exists() and read(path).get('status')=='RUNNING' for path in [folder/a/'STATUS.json' for a in ('N','F')]):time.sleep(1)
    workers=[]
    for path in folder.glob('*/pair*/*/DISPATCH.json'):
        row=read(path);pid=row['pid'];pair=path.parent.parent;arm=path.parent.name
        expected=['python3',str(run/'code/scripts/run_placement_worker.py'),str(pair.relative_to(life.PROJECT/'runs')),arm]
        if process_matches(pid,expected):
            os.kill(pid,signal.SIGTERM);signals.append(pid);workers.append((pid,expected))
    until=time.time()+45
    while time.time()<until and any(process_matches(pid,argv) for pid,argv in workers):time.sleep(1)
    outstanding_workers=[pid for pid,argv in workers if process_matches(pid,argv)]
    audit=[];extra=[]
    for path in folder.rglob('LAUNCH_*.json'):
        receipt=read(path);cid=receipt.get('stdout','').strip()
        if len(cid)!=64 or any(c not in '0123456789abcdef' for c in cid):continue
        info=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
        assert info['Id']==cid and info['Name'].startswith('/yuchen-fastrsi-'+NAME)
        if info['State']['Running']:
            role=path.stem.removeprefix('LAUNCH_');life.active.append((cid,role,path.parent));extra+=life.stop_all()
        after=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
        audit.append({'container':cid,'name':info['Name'],'running':after['State']['Running']})
    result={'host':host,'signals_sent':signals,'outstanding_workers':outstanding_workers,'extra_stops':extra,'containers':audit,'all_stopped':all(not x['running'] for x in audit),'time':time.time()}
    life.save(folder/'FINAL_CONTAINER_AUDIT.json',result)
    branches={arm:read(folder/arm/'STATUS.json') for arm in ('N','F') if (folder/arm/'STATUS.json').exists()}
    starts=[x['started_at'] for x in branches.values() if 'started_at' in x]
    finishes=[x['finished_at'] for x in branches.values() if 'finished_at' in x]
    complete=len(branches)==2 and all(x['status']=='COMPLETED' for x in branches.values())
    life.save(folder/'STATUS.json',{'host':host,'status':'COMPLETED' if complete and result['all_stopped'] and not outstanding_workers else 'FAILED','branches':branches,
        'started_at':min(starts) if starts else None,'finished_at':max(finishes) if len(finishes)==2 else None,
        'full_wall_seconds':max(finishes)-min(starts) if len(starts)==len(finishes)==2 else None,
        'scope':'Parallel branch interval on this host only; not sum of branch times'})
    assert result['all_stopped'] and not outstanding_workers
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--host',choices=list(HOSTS));ap.add_argument('--arm',choices=['N','F']);ap.add_argument('--preflight',choices=list(HOSTS));ap.add_argument('--cleanup-host',choices=list(HOSTS));args=ap.parse_args()
    if args.preflight:return preflight(args.preflight)
    if args.cleanup_host:return cleanup_host(args.cleanup_host)
    if args.host and args.arm:return branch(args.host,args.arm)
    if args.host or args.arm:raise ValueError('Host and arm must be supplied together')
    require_host('187');p=life.PROJECT;run=p/'runs'/NAME
    lock=open(p/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(p/'runs/resident_full_os_v1_20261003/STATUS.json')['status']=='COMPLETED'
    run.mkdir(exist_ok=False);start=time.time();deadline=start+2400;configure_context(run,deadline)
    state={'status':'RUNNING','started_at':start,'deadline':deadline,'children':[]};life.save(run/'STATUS.json',state)
    for rel in ('scripts','src/fastrsi','tests','configs'):freeze(p/rel,run/'code'/rel)
    life.save(run/'CODE_MANIFEST.json',tree(run/'code'))
    source=p/'runs/shared_flow_net_parallel_v2_20261002'
    for rel in BINDINGS:shutil.copyfile(source/rel,run/rel)
    shutil.copytree(source/'initial_experience',run/'initial_experience')
    protocol=read(p/'planning'/NAME/'PROTOCOL.json');assert protocol['arms']=={'N':'fixed10','F':'fixed5'} and len(protocol['cases'])==4
    protocol.update(deadline=deadline,placement=PLACEMENT,maximum_campaign_seconds=2400,status='RUNNING');life.save(run/'PROTOCOL.json',protocol)
    children=[]
    def interrupted(n,f):raise KeyboardInterrupt('Campaign interrupted '+str(n))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    script=str(run/'code/scripts/orchestrate_structural_window.py')
    try:
        preflight('187')
        result=life.command(PEER+['python3',script,'--preflight','188'],timeout=90);life.save(run/'REMOTE_PREFLIGHT.json',result);assert result['returncode']==0,result
        cpu=run/'cpu';cpu.mkdir()
        cmd='source /opt/dtk/env.sh && cd '+life.BASE+'/code/tests && python -m unittest test_resident_client test_resident_worker -v && /project/envs/os-v1/bin/python -m unittest test_native_window_attempt test_feedback_window -v'
        cid=life.launch('cpu',life.cpu_args('yuchen-fastrsi-'+NAME+'-cpu',cpu)+[life.IMAGE,'timeout','150','bash','-lc',cmd],cpu);life.wait_exit(cid,160);life.save(cpu/'STOP_RECEIPT.json',life.stop_all());state['cpu']='PASSED'
        for host in HOSTS:
            for arm in ('N','F'):
                argv=['python3',script,'--host',host,'--arm',arm]
                if host=='188':argv=PEER+argv
                with (run/(host+'_'+arm+'_controller.log')).open('x') as log:proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children.append((host,arm,proc));state['children'].append({'host':host,'arm':arm,'pid':proc.pid})
        life.save(run/'STATUS.json',state)
        while any(proc.poll() is None for _,_,proc in children):
            if time.time()>deadline:raise TimeoutError('Campaign deadline')
            time.sleep(2)
        state['branches']={host+'_'+arm:read(run/('host'+host)/arm/'STATUS.json') for host,arm,_ in children}
        assert all(proc.returncode==0 for _,_,proc in children) and all(x['status']=='COMPLETED' for x in state['branches'].values()),state['branches']
        state['status']='COMPLETED'
    except BaseException:state.update(status='FAILED',error=traceback.format_exc())
    finally:
        for host,arm,proc in children:
            if host=='187' and proc.poll() is None:proc.send_signal(signal.SIGTERM)
        for host in HOSTS:
            try:
                if host=='187':cleanup_host(host)
                else:
                    result=life.command(PEER+['python3',script,'--cleanup-host',host],timeout=220);life.save(run/'REMOTE_CLEANUP.json',result);assert result['returncode']==0,result
            except BaseException:state.setdefault('cleanup_attention',{})[host]=traceback.format_exc()
        for host,arm,proc in children:
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:state.setdefault('cleanup_attention',{})[host+'_'+arm]='Child did not exit'
        life.save(run/'STOP_RECEIPT.json',life.stop_all())
        if state.get('cleanup_attention'):state['status']='FAILED'
        state.update(finished_at=time.time(),full_wall_seconds=time.time()-start);life.save(run/'STATUS.json',state)
    if state['status']!='COMPLETED':raise SystemExit(1)

if __name__=='__main__':main()

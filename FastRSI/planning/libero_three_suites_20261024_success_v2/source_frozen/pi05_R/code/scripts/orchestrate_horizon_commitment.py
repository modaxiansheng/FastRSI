"""Bounded same-checkpoint H10W10 / H20W10 / H20W20 development diagnosis."""
import argparse, fcntl, json, os, shutil, signal, socket, subprocess, time, traceback
from pathlib import Path
import orchestrate_full_os as life
from native_service_config import configure_qwen
from orchestrate_resident_services import gpu, configure_context, check_ports
from orchestrate_resident_full_os import BINDINGS, read, tree, freeze

NAME = 'horizon_commitment_v2_20261003'
ARMS = {'A': (10,10), 'B': (20,10), 'C': (20,20)}

def binding(arm):
    i = list(ARMS).index(arm)
    return {'policy_port':22100+i*100, 'qwen_port':22110+i*100, 'gpu_base':i*2}

def validate_protocol(protocol):
    assert protocol['arms']=={k:{'inference_horizon':h,'commit_steps':w} for k,(h,w) in ARMS.items()}
    assert protocol['maximum_robot_cases']==6 and protocol['maximum_campaign_seconds']==2400
    assert protocol['sampling_steps']==10 and protocol['seed']==7
    assert [(c['task_id'],c['init_state_id']) for c in protocol['cases']]==[(0,3),(4,2)]
    assert all(c['suite']=='libero_10' and c['mode']=='development' and c['max_steps']==520 for c in protocol['cases'])

def exposure_check(root, cases):
    from horizon_exposure import scan_actual_results
    receipt=scan_actual_results(root/'runs',cases)
    if receipt['matches'] or receipt['ambiguous'] or receipt['errors']:
        raise ValueError('New-instance exposure check failed: '+repr(receipt))
    return {'checked_at':time.time(),'status':'PASSED',**receipt}

def pool(run, folder, arm, deadline):
    ports=binding(arm); services={}; cfg=read(run/'MODEL_CONFIG.json'); start=time.time()
    for role,receipt,port,device in [('pi05','LAUNCH_pi05_v2.json',ports['policy_port'],ports['gpu_base']),('qwen','LAUNCH_qwen.json',ports['qwen_port'],ports['gpu_base']+1)]:
        service=folder/'pool'/role; service.mkdir(parents=True); freeze(run/'code',service/'code'); configure_context(service,deadline)
        argv=[x.replace(str(life.OLD),str(service)).replace('/project/runs/pilot_20261001',life.BASE) for x in read(life.OLD/receipt)['argv']]
        if role=='qwen':
            script=service/'code/scripts/qwen_agent_server.py'; script.write_text(script.read_text().replace('49510',str(port))); argv=configure_qwen(argv,cfg,life)
        else:
            shutil.copytree(life.PROJECT/'models/openpi_data',service/'tokenizer_cache')
            argv=[x.replace(str(life.PROJECT/'models/openpi_data'),str(service/'tokenizer_cache')).replace('49500',str(port)) for x in argv]
            pos=argv.index(life.IMAGE); argv[pos:pos]=['-v',str(life.PROJECT)+':/project:rw']
            argv=[x.replace('PYTHONPATH=','PYTHONPATH='+life.BASE+'/code/src:') if x.startswith('PYTHONPATH=') else x for x in argv]
            argv[-1]=argv[-1].replace('python -m PhyAgentOS.runtime.policy.openpi.native_openpi_server','python '+life.BASE+'/code/scripts/resident_openpi_server.py')+' --receipt-dir /pilot/resident_receipts'
            if ARMS[arm][0]==20: argv[-1]+=' --experimental-action-horizon 20'
        argv=gpu(argv,device); argv[argv.index('--name')+1]='yuchen-fastrsi-'+NAME+'-'+arm.lower()+'-pool-'+role
        ix=argv.index(life.IMAGE)
        if argv[ix+1]=='timeout':argv[ix+2]=str(int(deadline-time.time()))
        life.save(service/'CODE_MANIFEST.json',tree(service/'code')); services[role]=life.launch(role,argv,service)
    life.health(ports['policy_port'],limit=360); life.health(ports['qwen_port'],limit=360)
    return services,time.time()-start

def branch(arm):
    root=life.PROJECT; run=root/'runs'/NAME; protocol=read(run/'PROTOCOL.json'); deadline=protocol['deadline']
    folder=run/arm; folder.mkdir(exist_ok=False); configure_context(run,deadline)
    state={'status':'RUNNING','arm':arm,'pid':os.getpid(),'started_at':time.time(),'groups':[]}; life.save(folder/'STATUS.json',state)
    worker=None
    def interrupted(n,f):raise KeyboardInterrupt('Interrupted '+str(n))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        services,state['pool_startup_seconds']=pool(run,folder,arm,deadline); configure_context(run,deadline)
        ports=binding(arm); horizon,commit=ARMS[arm]
        for index,case in enumerate(protocol['cases']):
            pair=folder/('pair'+str(index)); pair.mkdir()
            for rel in BINDINGS:shutil.copyfile(run/rel,pair/rel)
            shutil.copytree(run/'initial_experience',pair/'initial_experience')
            assert tree(pair/'initial_experience')==read(run/'EXPERIENCE_BINDING.json')
            life.save(pair/'PLAN.json',{'cases':[case]})
            shift=ports['policy_port']-49500
            life.save(pair/'PROTOCOL.json',{'slot':0,'absolute_deadline':deadline,'diagnostic_trace':True,
                'control_kind':'feedback_window','arm_modes':{'N':'fixed_commitment'},
                'port_offset':shift,'gpu_bases':{'N':ports['gpu_base']},'terminal_evidence_view':True,
                'horizon_commitments':{'N':{'inference_horizon':horizon,'commit_steps':commit}},
                'resident_services':{'policy_port':ports['policy_port'],'qwen_port':ports['qwen_port'],
                    'policy_container':services['pi05'],'qwen_container':services['qwen'],
                    'session_id':NAME+'-'+arm+'-'+str(index),'max_inferences':400,'inference_horizon':horizon}})
            group=pair/'N'; freeze(run/'code',group/'code')
            for rel in ['scripts/run_native_baseline.py','scripts/qwen_agent_server.py']:
                path=group/'code'/rel;text=path.read_text()
                for port in (49500,49502,49510,49520):text=text.replace(str(port),str(port+shift))
                path.write_text(text)
            life.save(group/'CODE_MANIFEST.json',tree(group/'code'))
            name=str(pair.relative_to(root/'runs'))
            with (group/'controller.log').open('x') as log:
                worker=subprocess.Popen(['python3',str(run/'code/scripts/run_placement_worker.py'),name,'N'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'PYTHONPATH':str(run/'code/src')})
            life.save(group/'DISPATCH.json',{'pid':worker.pid,'started_at':time.time()})
            while worker.poll() is None:
                if time.time()>deadline-30:raise TimeoutError('Campaign bound')
                time.sleep(2)
            assert (group/'STATUS.json').exists(), 'Worker failed before status publication: '+(group/'controller.log').read_text()[-4000:]
            status=read(group/'STATUS.json');state['groups'].append({'path':str(group.relative_to(run)),**status});life.save(folder/'STATUS.json',state)
            assert worker.returncode==0 and status['status']=='COMPLETED',status
        state['status']='COMPLETED'
    except BaseException:state.update(status='FAILED',error=traceback.format_exc())
    finally:
        if worker is not None and worker.poll() is None:
            worker.send_signal(signal.SIGTERM)
            try:worker.wait(timeout=90)
            except subprocess.TimeoutExpired:state['cleanup_attention']='Worker pending'
        life.save(folder/'POOL_STOP_RECEIPT.json',life.stop_all())
        state.update(finished_at=time.time(),full_wall_seconds=time.time()-state['started_at']);life.save(folder/'STATUS.json',state)
    if state['status']!='COMPLETED':raise SystemExit(1)

def cleanup(run, children):
    for arm,proc in children:
        if proc.poll() is None:proc.send_signal(signal.SIGTERM)
    pending=[]
    for arm,proc in children:
        try:proc.wait(timeout=110)
        except subprocess.TimeoutExpired:pending.append(arm)
    audit=[]
    for path in run.rglob('LAUNCH_*.json'):
        cid=read(path).get('stdout','').strip()
        if len(cid)!=64 or any(c not in '0123456789abcdef' for c in cid):continue
        info=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
        assert info['Id']==cid and info['Name'].startswith('/yuchen-fastrsi-'+NAME)
        if info['State']['Running']:
            life.active.append((cid,path.stem,path.parent));life.stop_all()
        info=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
        audit.append({'container':cid,'name':info['Name'],'running':info['State']['Running'],'exit_code':info['State']['ExitCode']})
    life.save(run/'CONTAINER_AUDIT.json',audit);assert all(not x['running'] for x in audit)
    if pending:raise RuntimeError('Branches pending cleanup: '+repr(pending))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--branch',choices=list(ARMS));args=ap.parse_args()
    assert socket.gethostname()=='j03r4n13'
    if args.branch:return branch(args.branch)
    root=life.PROJECT;run=root/'runs'/NAME
    lock=open(root/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=read(root/'planning'/NAME/'PROTOCOL.json');validate_protocol(protocol)
    exposure=exposure_check(root,protocol['cases'])
    assert read(root/'runs/horizon20_probe_v1_20261003/STATUS.json')['status']=='COMPLETED'
    run.mkdir(exist_ok=False);start=time.time();deadline=start+2400;configure_context(run,deadline)
    state={'status':'RUNNING','started_at':start,'deadline':deadline,'children':[]};life.save(run/'STATUS.json',state)
    children=[]
    def interrupted(n,f):raise KeyboardInterrupt('Campaign interrupted '+str(n))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        life.save(run/'EXPOSURE_CHECK.json',exposure)
        for rel in ('scripts','src/fastrsi','tests','configs'):freeze(root/rel,run/'code'/rel)
        life.save(run/'CODE_MANIFEST.json',tree(run/'code'))
        source=root/'runs/shared_flow_net_parallel_v2_20261002'
        for rel in BINDINGS:shutil.copyfile(source/rel,run/rel)
        shutil.copytree(source/'initial_experience',run/'initial_experience')
        assert tree(run/'initial_experience')==read(run/'EXPERIENCE_BINDING.json')
        protocol.update(status='RUNNING',deadline=deadline,placement={a:binding(a) for a in ARMS});life.save(run/'PROTOCOL.json',protocol)
        cpu=run/'cpu';cpu.mkdir();life.gate(cpu);check_ports([22100+i*100+j for i in range(3) for j in (0,2,10,20)])
        cmd='source /opt/dtk/env.sh && cd '+life.BASE+'/code/tests && python -m unittest test_experimental_horizon_policy test_resident_openpi_server test_resident_client test_resident_worker -v && /project/envs/os-v1/bin/python -m unittest test_feedback_window test_native_window_attempt test_native_isolation test_experience_admission test_horizon_commitment test_horizon_analysis test_horizon_campaign -v'
        cid=life.launch('cpu',life.cpu_args('yuchen-fastrsi-'+NAME+'-cpu',cpu)+[life.IMAGE,'timeout','180','bash','-lc',cmd],cpu)
        life.wait_exit(cid,190);life.save(cpu/'STOP_RECEIPT.json',life.stop_all());state['cpu']='PASSED'
        for arm in ARMS:
            with (run/(arm+'_controller.log')).open('x') as log:
                proc=subprocess.Popen(['python3',str(run/'code/scripts/orchestrate_horizon_commitment.py'),'--branch',arm],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append((arm,proc));state['children'].append({'arm':arm,'pid':proc.pid})
        life.save(run/'STATUS.json',state)
        while any(p.poll() is None for _,p in children):
            if time.time()>deadline:raise TimeoutError('Campaign bound')
            time.sleep(2)
        state['branches']={a:read(run/a/'STATUS.json') for a in ARMS}
        assert all(p.returncode==0 for _,p in children) and all(x['status']=='COMPLETED' for x in state['branches'].values()),state['branches']
        state['status']='COMPLETED'
    except BaseException:state.update(status='FAILED',error=traceback.format_exc())
    finally:
        try:cleanup(run,children)
        except BaseException:state.update(status='FAILED',cleanup_attention=traceback.format_exc())
        life.save(run/'STOP_RECEIPT.json',life.stop_all());state.update(finished_at=time.time(),full_wall_seconds=time.time()-start);life.save(run/'STATUS.json',state)
    if state['status']!='COMPLETED':raise SystemExit(1)

if __name__=='__main__':main()

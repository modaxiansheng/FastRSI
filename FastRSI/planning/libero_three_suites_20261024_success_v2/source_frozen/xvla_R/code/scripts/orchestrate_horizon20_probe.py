"""One-GPU same-checkpoint H20 loading/replay check; no environment execution."""
import fcntl,json,shutil,signal,socket,time,traceback,subprocess
from pathlib import Path
import orchestrate_full_os as life
from orchestrate_resident_services import configure_context,gpu
from orchestrate_resident_full_os import freeze,tree,read
NAME='horizon20_probe_v1_20261003'

def main():
    assert socket.gethostname()=='j03r4n13'
    root=life.PROJECT;run=root/'runs'/NAME
    lock=open(root/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run.mkdir(exist_ok=False);started=time.time();deadline=started+600;configure_context(run,deadline)
    state={'status':'RUNNING','started_at':started,'deadline':deadline,'scope':'Loading and saved-observation replay only; no robot or learning update'}
    life.save(run/'STATUS.json',state)
    def interrupt(n,f):raise KeyboardInterrupt('Probe interrupted '+str(n))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        for rel in ['scripts','src/fastrsi','tests']:freeze(root/rel,run/'code'/rel)
        life.save(run/'CODE_MANIFEST.json',tree(run/'code'))
        life.save(run/'PROTOCOL.json',{'action_horizon':20,'checkpoint':'same frozen pi05_libero_pytorch_fp32',
            'input_protocol':'runs/resident_services_v1_20261003/PROTOCOL.json',
            'default_changed':False,'robot_cases':0,'training':False,'sampling_steps':10,
            'purpose':'Finite output, immutable horizon config, resident reset replay consistency. No quality or speedup claim.'})
        cpu=run/'cpu';cpu.mkdir();life.gate(cpu)
        cmd='source /opt/dtk/env.sh && cd '+life.BASE+'/code/tests && python -m unittest test_experimental_horizon_policy -v'
        cid=life.launch('cpu',life.cpu_args('yuchen-fastrsi-'+NAME+'-cpu',cpu)+[life.IMAGE,'timeout','60','bash','-lc',cmd],cpu)
        life.wait_exit(cid,70);life.save(cpu/'STOP_RECEIPT.json',life.stop_all());state['cpu']='PASSED'
        shutil.copytree(root/'models/openpi_data',run/'tokenizer_cache')
        argv=read(life.OLD/'LAUNCH_pi05_v2.json')['argv']
        argv=[x.replace(str(life.OLD),str(run)).replace(str(root/'models/openpi_data'),str(run/'tokenizer_cache')) for x in argv]
        argv=gpu(argv,0);argv[argv.index('--name')+1]='yuchen-fastrsi-'+NAME+'-model'
        image=argv.index(life.IMAGE);argv[image:image]=['-v',str(root)+':/project:rw']
        argv=[x.replace('PYTHONPATH=','PYTHONPATH=/pilot/code/src:/pilot/code/scripts:') if x.startswith('PYTHONPATH=') else x for x in argv]
        image=argv.index(life.IMAGE);argv[image+2]=str(int(deadline-time.time()-30))
        argv[-1]='source /opt/dtk/env.sh && source /opt/hyhal/env.sh && exec python /pilot/code/scripts/check_experimental_horizon.py --checkpoint-dir /model --horizon 20 --protocol /project/runs/resident_services_v1_20261003/PROTOCOL.json --output /pilot/check'
        cid=life.launch('model',argv,run);life.wait_exit(cid,int(deadline-time.time()-10))
        checked=read(run/'check/RESULT.json');assert checked['status']=='PASSED',checked
        state['check_status']='PASSED';state['status']='COMPLETED'
    except BaseException:state.update(status='FAILED',error=traceback.format_exc())
    finally:
        life.save(run/'STOP_RECEIPT.json',life.stop_all())
        audit=[]
        for path in run.rglob('LAUNCH_*.json'):
            cid=read(path).get('stdout','').strip()
            if len(cid)!=64 or any(c not in '0123456789abcdef' for c in cid):continue
            info=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
            assert info['Id']==cid and info['Name'].startswith('/yuchen-fastrsi-'+NAME)
            audit.append({'id':cid,'name':info['Name'],'running':info['State']['Running']})
        life.save(run/'CONTAINER_AUDIT.json',audit)
        if any(x['running'] for x in audit):state.update(status='FAILED',cleanup_attention=True)
        state.update(finished_at=time.time(),full_wall_seconds=time.time()-started);life.save(run/'STATUS.json',state)
    if state['status']!='COMPLETED':raise SystemExit(1)
if __name__=='__main__':main()

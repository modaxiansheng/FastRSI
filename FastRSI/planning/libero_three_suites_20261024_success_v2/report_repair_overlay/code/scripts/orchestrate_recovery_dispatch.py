"""One bounded, non-executing contrast using the existing Qwen deployment."""
import fcntl
import hashlib
import json
import shutil
import signal
import socket
import time
import traceback
import orchestrate_full_os as life
from native_service_config import configure_qwen
from orchestrate_resident_services import configure_context, gpu
from orchestrate_resident_full_os import freeze, tree, read

NAME='recovery_dispatch_v1_20261004'


def main():
    assert socket.gethostname()=='j03r4n13'
    lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=life.PROJECT/'runs'/NAME;run.mkdir(exist_ok=False)
    started=time.time();deadline=started+900;configure_context(run,deadline)
    stream=run/'stream';stream.mkdir();cpu=run/'cpu';cpu.mkdir()
    state=dict(status='RUNNING',started_at=started,robot_episodes=0,persistent_updates=0)
    life.save(run/'STATUS.json',state)
    def interrupt(n,f):raise KeyboardInterrupt('Interrupted '+str(n))
    signal.signal(signal.SIGINT,interrupt);signal.signal(signal.SIGTERM,interrupt)
    try:
        for rel in ('scripts','src/fastrsi','tests','configs'):freeze(life.PROJECT/rel,run/'code'/rel)
        life.save(run/'CODE_MANIFEST.json',tree(run/'code'))
        source=life.PROJECT/'runs/stall_recovery_v1_20261004/N'
        shutil.copyfile(source/'MODEL_CONFIG.json',run/'MODEL_CONFIG.json')
        inputs=life.PROJECT/'planning'/NAME
        for rel,digest in read(inputs/'SOURCE_MANIFEST.json').items():
            path=(life.CORE/rel.split('third_party/PhyAgentOS-core/',1)[1] if rel.startswith('third_party/PhyAgentOS-core/') else life.PROJECT/rel)
            assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,rel
        # Server's actual frozen core must match the prompt source inspected locally.
        native_source=life.CORE/'PhyAgentOS/verification/service.py'
        life.save(run/'CORE_BINDING.json',dict(service_sha256=hashlib.sha256(native_source.read_bytes()).hexdigest(),
            server_sha256=hashlib.sha256((life.CORE/'PhyAgentOS/runtime/targets/remote/libero/server.py').read_bytes()).hexdigest()))
        shutil.copytree(inputs,run/'inputs')
        protocol=read(inputs/'PROTOCOL.json');protocol['deadline']=deadline;life.save(run/'PROTOCOL.json',protocol)
        life.gate(cpu)
        cmd='source /opt/dtk/env.sh && cd '+life.BASE+'/code/tests && /project/envs/os-v1/bin/python -m unittest test_recovery_dispatch_probe -v'
        cid=life.launch('cpu',life.cpu_args('yuchen-fastrsi-'+NAME+'-cpu',cpu)+[life.IMAGE,'timeout','90','bash','-lc',cmd],cpu)
        life.wait_exit(cid,100);life.save(cpu/'STOP_RECEIPT.json',life.stop_all());state['cpu_status']='PASSED'
        load=time.time();argv=read(life.OLD/'LAUNCH_qwen.json')['argv']
        argv=[x.replace(str(life.OLD),str(stream)) for x in argv]
        argv[argv.index('--name')+1]='yuchen-fastrsi-'+NAME+'-qwen'
        argv=gpu(configure_qwen(argv,read(run/'MODEL_CONFIG.json'),life),0)
        ix=argv.index(life.IMAGE);assert argv[ix+1]=='timeout';argv[ix+2]=str(int(deadline-time.time()))
        life.launch('qwen',argv,stream);life.health(49510,limit=300)
        state['model_startup_seconds']=time.time()-load;life.save(run/'STATUS.json',state)
        remaining=int(deadline-time.time()-20);assert remaining>60
        cmd='source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/replay_recovery_dispatch.py --run '+life.BASE
        cid=life.launch('runner',life.cpu_args('yuchen-fastrsi-'+NAME+'-runner',stream)+[life.IMAGE,'timeout',str(remaining),'bash','-lc',cmd],stream)
        life.wait_exit(cid,remaining+5)
        result=read(stream/'RESULT.json');count=len(list((stream/'service_calls').glob('*/REQUEST.json')))
        assert result['status']=='COMPLETED' and len(result['rows'])==12 and count==12
        state.update(status='COMPLETED',model_calls=count,replay_seconds=result['seconds'])
    except BaseException:state.update(status='FAILED',error=traceback.format_exc())
    finally:
        stopped=life.stop_all();life.save(stream/'STOP_RECEIPT.json',stopped)
        if not all(not r['running'] for r in stopped):state.update(status='FAILED',cleanup_attention=True)
        state.update(full_wall_seconds=time.time()-started,finished_at=time.time());life.save(run/'STATUS.json',state)
        print(json.dumps(state),flush=True)
    if state['status']!='COMPLETED':raise SystemExit(1)


if __name__=='__main__':main()

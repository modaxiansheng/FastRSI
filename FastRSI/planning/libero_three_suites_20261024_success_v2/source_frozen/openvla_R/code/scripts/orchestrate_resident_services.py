"""One global owner; independent resident equivalence checks on two idle hosts."""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import shlex
import socket
import subprocess
import time
import traceback
from pathlib import Path

import orchestrate_full_os as life
from native_service_config import configure_qwen

NAME = 'resident_services_v1_20261003'
PEER = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',
        '-o','StrictHostKeyChecking=yes','-p','36000','root@10.232.1.188']


def read(path):
    return json.loads(path.read_text())


def gpu(argv, index):
    return [value.split('=',1)[0]+'='+str(index)
            if value.startswith(('HIP_VISIBLE_DEVICES=','CUDA_VISIBLE_DEVICES=')) else value for value in argv]


def configure_context(folder, deadline):
    life.CAMPAIGN = folder
    life.BASE = str(folder).replace(str(life.PROJECT), '/project')
    life.deadline = deadline


def check_ports(ports):
    for port in ports:
        with socket.socket() as sock:
            assert sock.connect_ex(('127.0.0.1',port)) != 0, f'Port busy: {port}'


def part(kind):
    p = life.PROJECT; run = p/'runs'/NAME; protocol=read(run/'PROTOCOL.json')
    folder=run/kind; folder.mkdir(exist_ok=False)
    configure_context(folder,protocol['deadline'])
    state={'status':'RUNNING','host':socket.gethostname(),'pid':os.getpid(),'started_at':time.time()}
    life.save(folder/'STATUS.json',state)
    def interrupted(n,f): raise RuntimeError('Interrupted '+str(n))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        life.gate(folder)
        ports=[22110,22210] if kind=='qwen' else [22100,22200]
        check_ports(ports)
        for i,label in enumerate(['A','B']):
            service=folder/label;service.mkdir()
            shutil.copytree(run/'code',service/'code')
            configure_context(service,protocol['deadline'])
            if kind=='qwen':
                server=service/'code/scripts/qwen_agent_server.py'
                server.write_text(server.read_text().replace('49510',str(ports[i])))
                cfg=read(run/'MODEL_CONFIG.json')
                argv=read(life.OLD/'LAUNCH_qwen.json')['argv']
                argv=[x.replace(str(life.OLD),str(service)).replace('/project/runs/pilot_20261001',life.BASE) for x in argv]
                argv=configure_qwen(argv,cfg,life)
            else:
                shutil.copytree(p/'models/openpi_data',service/'tokenizer_cache')
                argv=read(life.OLD/'LAUNCH_pi05_v2.json')['argv']
                argv=[x.replace(str(life.OLD),str(service)).replace('/project/runs/pilot_20261001',life.BASE)
                      .replace(str(p/'models/openpi_data'),str(service/'tokenizer_cache')).replace('49500',str(ports[i])) for x in argv]
                pos=argv.index(life.IMAGE)
                argv[pos:pos]=['-v',str(p)+':/project:rw']
                if label=='B':
                    argv=[x.replace('PYTHONPATH=','PYTHONPATH='+life.BASE+'/code/src:') if x.startswith('PYTHONPATH=') else x for x in argv]
                    argv[-1]=argv[-1].replace('python -m PhyAgentOS.runtime.policy.openpi.native_openpi_server',
                        'python '+life.BASE+'/code/scripts/resident_openpi_server.py')+' --receipt-dir /pilot/resident_receipts'
            argv=gpu(argv,i)
            image_index=argv.index(life.IMAGE)
            if argv[image_index+1]=='timeout':argv[image_index+2]=str(int(protocol['deadline']-time.time()))
            argv[argv.index('--name')+1]='yuchen-fastrsi-'+NAME+'-'+kind+'-'+label.lower()
            life.save(service/'CODE_MANIFEST.json',{str(f.relative_to(service/'code')):hashlib.sha256(f.read_bytes()).hexdigest() for f in (service/'code').rglob('*') if f.is_file()})
            life.launch(kind+'_'+label,argv,service)
        for port in ports:life.health(port,limit=360)
        state['ready_at']=time.time();state['startup_seconds']=state['ready_at']-state['started_at']
        life.save(folder/'STATUS.json',state)
        configure_context(run,protocol['deadline'])
        # Replays execute in the base image with numpy/torch and no GPU device access.
        common=life.cpu_args('yuchen-fastrsi-'+NAME+'-'+kind+'-replay',folder)
        script='replay_resident_qwen.py' if kind=='qwen' else 'replay_resident_policy.py'
        output=life.BASE+'/'+kind+'/replay'
        remaining=int(protocol['deadline']-time.time()-30);assert remaining>60
        cmd='source /opt/dtk/env.sh && python '+life.BASE+'/code/scripts/'+script+' --protocol '+life.BASE+'/PROTOCOL.json --output '+output
        cid=life.launch(kind+'_replay',common+[life.IMAGE,'timeout',str(remaining),'bash','-lc',cmd],folder)
        life.wait_exit(cid,remaining+5)
        result=read(folder/'replay/RESULT.json')
        assert result['status'] in ('PASSED','COMPLETED'),result
        state.update(status='COMPLETED',result_status=result['status'])
    except BaseException:
        state.update(status='FAILED',error=traceback.format_exc())
    finally:
        life.save(folder/'STOP_RECEIPT.json',life.stop_all())
        state.update(finished_at=time.time(),full_wall_seconds=time.time()-state['started_at'])
        life.save(folder/'STATUS.json',state);print(json.dumps(state),flush=True)
    if state['status']!='COMPLETED': raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--part',choices=['qwen','vla']);args=parser.parse_args()
    if args.part:return part(args.part)
    p=life.PROJECT;run=p/'runs'/NAME
    lock=open(p/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run.mkdir(exist_ok=False);start=time.time();deadline=start+1400
    state={'status':'RUNNING','started_at':start,'deadline':deadline,'children':[]};life.save(run/'STATUS.json',state)
    for rel in ('scripts','src/fastrsi','tests','configs'):
        shutil.copytree(p/rel,run/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(p/'planning'/NAME/'inputs',run/'inputs')
    for rel in ('MODEL_CONFIG.json','MODEL_BINDING.json','CORE_SOURCE_BINDING.json'):
        shutil.copyfile(p/'runs/short_exit_v1_20261003'/rel,run/rel)
    protocol=read(p/'planning'/NAME/'PROTOCOL.json');protocol['deadline']=deadline;life.save(run/'PROTOCOL.json',protocol)
    life.save(run/'CODE_MANIFEST.json',{str(f.relative_to(run/'code')):hashlib.sha256(f.read_bytes()).hexdigest() for f in (run/'code').rglob('*') if f.is_file()})
    children=[]
    def interrupted(n,f):raise RuntimeError('Interrupted '+str(n))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        cpu=run/'cpu';cpu.mkdir();configure_context(run,deadline);life.gate(cpu)
        command='source /opt/dtk/env.sh && python -c "import websocket,numpy; from PhyAgentOS.runtime.policy.msgpack_numpy import packb,unpackb" && cd '+life.BASE+'/code/tests && python -m unittest test_resident_policy test_resident_qwen_replay -v'
        cid=life.launch('cpu',life.cpu_args('yuchen-fastrsi-'+NAME+'-cpu',cpu)+[life.IMAGE,'timeout','150','bash','-lc',command],cpu)
        life.wait_exit(cid,160);life.save(cpu/'STOP_RECEIPT.json',life.stop_all());state['cpu']='PASSED'
        for kind in ('vla','qwen'):
            argv=['python3',str(run/'code/scripts/orchestrate_resident_services.py'),'--part',kind]
            if kind=='qwen':argv=PEER+argv
            with (run/(kind+'_controller.log')).open('x') as log:
                proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append((kind,proc));state['children'].append({'part':kind,'pid':proc.pid,'host':'187' if kind=='vla' else '188'})
        life.save(run/'STATUS.json',state)
        while any(proc.poll() is None for _,proc in children):
            if time.time()>deadline:raise TimeoutError('Resident campaign deadline')
            time.sleep(2)
        state['parts']={kind:read(run/kind/'STATUS.json') for kind,_ in children if (run/kind/'STATUS.json').exists()}
        assert len(state['parts'])==2 and all(s['status']=='COMPLETED' for s in state['parts'].values()),state['parts']
        assert all(proc.returncode==0 for _,proc in children)
        state['status']='COMPLETED'
    except BaseException:
        state.update(status='FAILED',error=traceback.format_exc())
    finally:
        # Normally children finish and stop their exact container IDs. Timeout is
        # an attention condition; retain their individual launch/stop receipts.
        for kind,proc in children:
            saved=read(run/kind/'STATUS.json') if (run/kind/'STATUS.json').exists() else {}
            needs_cleanup=proc.poll() is None or (kind=='qwen' and saved.get('status') not in ('COMPLETED','FAILED'))
            if needs_cleanup:
                try:
                    if kind=='vla':proc.send_signal(signal.SIGTERM)
                    elif saved.get('pid'):
                        expected=['python3',str(run/'code/scripts/orchestrate_resident_services.py'),'--part','qwen']
                        code=("import os,signal;from pathlib import Path;pid="+str(int(saved['pid']))+
                              ";actual=Path('/proc/'+str(pid)+'/cmdline').read_bytes().decode().strip(chr(0)).split(chr(0));"
                              +"assert actual=="+repr(expected)+";os.kill(pid,signal.SIGTERM)")
                        cleanup=subprocess.run(PEER+['python3 -c '+shlex.quote(code)],timeout=30,capture_output=True,text=True)
                        state.setdefault('remote_cleanup',[]).append({'returncode':cleanup.returncode,'stdout':cleanup.stdout,'stderr':cleanup.stderr})
                        if cleanup.returncode:state['cleanup_attention']='remote termination not confirmed'
                    else:state['cleanup_attention']='qwen child has no PID receipt; inspect before releasing ownership'
                    proc.wait(timeout=90)
                    if kind=='qwen':
                        until=time.time()+45
                        while time.time()<until and (run/kind/'STATUS.json').exists() and read(run/kind/'STATUS.json').get('status')=='RUNNING':time.sleep(1)
                        if read(run/kind/'STATUS.json').get('status')=='RUNNING':state['cleanup_attention']='remote child still running'
                except Exception:state['cleanup_attention']=traceback.format_exc()
        life.save(run/'STOP_RECEIPT.json',life.stop_all())
        state.update(finished_at=time.time(),full_wall_seconds=time.time()-start);life.save(run/'STATUS.json',state)
        print(json.dumps(state),flush=True)
    if state['status']!='COMPLETED':raise SystemExit(1)


if __name__=='__main__': main()

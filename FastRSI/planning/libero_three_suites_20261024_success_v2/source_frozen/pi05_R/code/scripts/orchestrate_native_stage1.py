"""One bounded public-engineering iteration: decoder pairs and native closure."""
import fcntl
import hashlib
import json
import shutil
import signal
import socket
import time
import traceback
import orchestrate_full_os as life
from native_service_config import configure_qwen,model_env

def main():
    name='native_stage1_v1_20261002';campaign=life.PROJECT/'runs'/name
    lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    campaign.mkdir(exist_ok=False);life.CAMPAIGN=campaign;life.BASE='/project/runs/'+name
    started=time.time();life.deadline=started+2100;status={'status':'PREFLIGHT','started_at':started}
    def stop(signum,frame):raise RuntimeError('Interrupted '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    for rel in ['scripts','src/fastrsi','tests','configs']:
        shutil.copytree(life.PROJECT/rel,campaign/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    life.save(campaign/'CODE_MANIFEST.json',{str(p.relative_to(campaign/'code')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (campaign/'code').rglob('*') if p.is_file()})
    cfg=json.loads((life.PROJECT/'configs/native_agent_qwen3_v1.json').read_text());life.save(campaign/'MODEL_CONFIG.json',cfg)
    life.save(campaign/'MODEL_BINDING.json',json.loads((life.PROJECT/'runs/native_baseline_v5_20261002/MODEL_BINDING.json').read_text()))
    protocol=json.loads((life.PROJECT/'configs/native_stage1_v1.json').read_text());life.save(campaign/'PROTOCOL.json',protocol)
    requests=json.loads((life.PROJECT/'planning/native_stage1_v1/DECODE_REQUESTS.json').read_text())
    for row in requests:
        for file,key in [('REQUEST.json','request_sha256'),('OUTPUT_TOKEN_IDS.json','tokens_sha256')]:
            assert hashlib.sha256((life.PROJECT/row['service_call']/file).read_bytes()).hexdigest()==row[key]
    life.save(campaign/'DECODE_REQUESTS.json',requests);life.save(campaign/'ROSTER.json',protocol['roster'])
    stream=campaign/'stream';stream.mkdir();life.save(campaign/'STATUS.json',status)
    def run(role,command,limit,extra=None):
        argv=life.cpu_args('yuchen-fastrsi-'+name+'-'+role,stream)+model_env(cfg)+[
            '-e','FASTRSI_STAGE_RUN='+life.BASE,'-e','FASTRSI_NATIVE_RUN='+life.BASE+'/stream',
            '-e','FASTRSI_CONSUME_TERMINAL=1','-v',cfg['weights']+':/model:ro']+(extra or [])+[
            life.IMAGE,'timeout',str(limit),'bash','-lc','source /opt/dtk/env.sh && '+command]
        cid=life.launch(role,argv,stream);life.wait_exit(cid,limit)
    try:
        life.gate(stream)
        status['stage']='cpu_regression';life.save(campaign/'STATUS.json',status)
        tests=' '.join('test_'+n+'.py' for n in [])
        command='/project/envs/os-v1/bin/python -m unittest discover -s '+life.BASE+'/code/tests -p "test_native*.py"'
        command+=' && /project/envs/os-v1/bin/python '+life.BASE+'/code/tests/test_terminal_results.py'
        command+=' && /project/envs/os-v1/bin/python '+life.BASE+'/code/tests/test_constrained_tools.py'
        command+=' && /project/envs/qwen3-v1/bin/python '+life.BASE+'/code/scripts/check_decoder_equivalence.py'
        run('tests',command,300)
        assert json.loads((campaign/'DECODER_EQUIVALENCE.json').read_text())['status']=='PASSED'
        argv=json.loads((life.OLD/'LAUNCH_qwen.json').read_text())['argv']
        argv[argv.index('--name')+1]='yuchen-fastrsi-'+name+'-qwen'
        argv=[x.replace(str(life.OLD),str(stream)) for x in argv]
        life.launch('qwen',configure_qwen(argv,cfg,life),stream);life.health(49510)
        status.update(status='RUNNING',stage='decoder_pairs',startup_seconds=time.time()-started);life.save(campaign/'STATUS.json',status)
        run('decoder','/project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/probe_decoder_cost.py',1200)
        status['stage']='native_terminal_lifecycle';life.save(campaign/'STATUS.json',status)
        shutil.copytree(life.PROJECT/'models/openpi_data',stream/'tokenizer_cache')
        for role,receipt in [('pi05','LAUNCH_pi05_v2.json'),('target','LAUNCH_target.json')]:
            argv=json.loads((life.OLD/receipt).read_text())['argv'];argv[argv.index('--name')+1]='yuchen-fastrsi-'+name+'-'+role
            argv=[x.replace(str(life.OLD),str(stream)).replace('/project/runs/pilot_20261001',life.BASE+'/stream') for x in argv]
            if role=='pi05':argv=[x.replace(str(life.PROJECT/'models/openpi_data'),str(stream/'tokenizer_cache')) for x in argv]
            else:argv=[x.replace('/project/scripts/instrument_target.py',life.BASE+'/code/scripts/instrument_target.py').replace(':/project/scripts',':'+life.BASE+'/code/scripts') for x in argv]
            life.launch(role,argv,stream)
        life.health(49500)
        until=time.time()+120
        while True:
            try:
                with socket.create_connection(('127.0.0.1',49502),timeout=1):break
            except OSError:
                if time.time()>until:raise
                time.sleep(1)
        run('lifecycle','/project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/run_native_baseline.py',600)
        assert json.loads((stream/'RESULT.json').read_text())['status']=='COMPLETED'
        status['stage']='native_batch_end_memory';life.save(campaign/'STATUS.json',status)
        run('memory','/project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/probe_terminal_memory.py',400)
        assert json.loads((stream/'terminal_memory/RESULT.json').read_text())['status']=='PASSED'
        status.update(status='COMPLETED',stage='done')
    except BaseException:
        status.update(status='FAILED',error=traceback.format_exc())
    finally:
        life.save(stream/'STOP_RECEIPT.json',life.stop_all())
        status.update(finished_at=time.time(),full_wall_seconds=time.time()-started)
        life.save(campaign/'STATUS.json',status);print(json.dumps(status),flush=True)
    if status['status']!='COMPLETED':raise SystemExit(1)

if __name__=='__main__':main()

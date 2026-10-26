"""One CPU regression plus one real saved-request replay; no robot or learning."""
import fcntl
import hashlib
import json
import shutil
import signal
import time
import traceback
import orchestrate_full_os as life
from native_service_config import configure_qwen,model_env

def main():
    name='native_queue_receipt_v1_20261002'
    prior=life.PROJECT/'runs/native_terminal_eval_v2_20261002'
    assert json.loads((prior/'STATUS.json').read_text())['status']=='COMPLETED'
    lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    campaign=life.PROJECT/'runs'/name;campaign.mkdir(exist_ok=False)
    life.CAMPAIGN=campaign;life.BASE='/project/runs/'+name;started=time.time();life.deadline=started+480
    status={'status':'PREFLIGHT','started_at':started}
    def interrupt(signum,frame):raise RuntimeError('Interrupted '+str(signum))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    for rel in ['scripts','src/fastrsi','tests','configs']:
        shutil.copytree(life.PROJECT/rel,campaign/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    life.save(campaign/'CODE_MANIFEST.json',{str(p.relative_to(campaign/'code')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (campaign/'code').rglob('*') if p.is_file()})
    cfg=json.loads((life.PROJECT/'configs/native_agent_qwen3_v2.json').read_text());life.save(campaign/'MODEL_CONFIG.json',cfg)
    life.save(campaign/'MODEL_BINDING.json',json.loads((prior/'MODEL_BINDING.json').read_text()))
    life.save(campaign/'PROTOCOL.json',{'scope':'One post-acceptance saved-request replay','max_model_calls':1,'max_campaign_seconds':480,'robot_episodes':0,'learning':False})
    stream=campaign/'stream';stream.mkdir();life.save(campaign/'STATUS.json',status)
    try:
        life.gate(stream);tests=campaign/'tests';tests.mkdir()
        argv=life.cpu_args('yuchen-fastrsi-'+name+'-tests',tests)+[life.IMAGE,'timeout','90','bash','-lc',
            'source /opt/dtk/env.sh && /project/envs/os-v1/bin/python '+life.BASE+'/code/tests/test_native_workspace.py']
        cid=life.launch('tests',argv,tests);life.wait_exit(cid,90)
        argv=json.loads((life.OLD/'LAUNCH_qwen.json').read_text())['argv'];argv[argv.index('--name')+1]='yuchen-fastrsi-'+name+'-qwen'
        argv=[x.replace(str(life.OLD),str(stream)) for x in argv]
        life.launch('qwen',configure_qwen(argv,cfg,life),stream);life.health(49510)
        status.update(status='RUNNING',startup_seconds=time.time()-started);life.save(campaign/'STATUS.json',status)
        argv=life.cpu_args('yuchen-fastrsi-'+name+'-runner',stream)+model_env(cfg)+[
            '-e','FASTRSI_NATIVE_RUN='+life.BASE+'/stream',life.IMAGE,'timeout','250','bash','-lc',
            'source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/probe_native_queue_receipt.py']
        cid=life.launch('runner',argv,stream);life.wait_exit(cid,250)
        assert json.loads((stream/'RESULT.json').read_text())['status']=='PASSED_ONE_SAVED_INPUT'
        status['status']='COMPLETED'
    except BaseException:status.update(status='FAILED',error=traceback.format_exc())
    finally:
        life.save(stream/'STOP_RECEIPT.json',life.stop_all());status.update(finished_at=time.time(),full_wall_seconds=time.time()-started)
        life.save(campaign/'STATUS.json',status);print(json.dumps(status),flush=True)
    if status['status']!='COMPLETED':raise SystemExit(1)

if __name__=='__main__':main()

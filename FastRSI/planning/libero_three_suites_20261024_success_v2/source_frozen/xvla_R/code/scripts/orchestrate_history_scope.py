"""Bounded paired inference, one resident Qwen; original baseline must have stopped."""
import fcntl
import hashlib
import json
import shutil
import signal
import time
import traceback
import orchestrate_full_os as life


def main():
    config_path=life.PROJECT/'configs/history_scope_probe_v1.json'
    config=json.loads(config_path.read_text())
    source=life.PROJECT/'runs'/config['source_campaign']
    assert json.loads((source/'STATUS.json').read_text())['status']=='COMPLETED'
    assert all(not row['running'] for row in json.loads((source/'stream/STOP_RECEIPT.json').read_text()))
    root=life.PROJECT/'runs'/config['experiment_id'];root.mkdir(exist_ok=False)
    life.CAMPAIGN=root;life.BASE='/project/runs/'+config['experiment_id']
    start=time.time();life.deadline=start+config['campaign_timeout_seconds']
    status={'status':'PREFLIGHT','started_at':start};life.save(root/'STATUS.json',status)
    lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def interrupted(signum,frame):raise RuntimeError('Interrupted '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    shutil.copyfile(config_path,root/'PROTOCOL.json')
    for rel in ['scripts','src/fastrsi','tests','configs']:
        shutil.copytree(life.PROJECT/rel,root/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    life.save(root/'CODE_MANIFEST.json',{str(p.relative_to(root/'code')):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root/'code').rglob('*') if p.is_file()})
    prefix='yuchen-fastrsi-historyv1'
    try:
        life.gate(root)
        tests=root/'tests';tests.mkdir()
        argv=life.cpu_args(prefix+'-contracts',tests)+[life.IMAGE,'timeout','120','bash','-lc',
            'source /opt/dtk/env.sh && /project/envs/os-v1/bin/python '+life.BASE+'/code/tests/test_history_scope.py']
        cid=life.launch('contracts',argv,tests);life.wait_exit(cid,120)
        life.save(tests/'STOP_RECEIPT.json',life.stop_all())
        life.gate(root)
        argv=json.loads((life.OLD/'LAUNCH_qwen.json').read_text())['argv']
        argv[argv.index('--name')+1]=prefix+'-qwen'
        argv=[x.replace(str(life.OLD),str(root)).replace('/project/runs/pilot_20261001',life.BASE) for x in argv]
        life.launch('qwen',argv,root);life.health(49510)
        status['startup_seconds']=time.time()-start
        limit=str(config['controller_timeout_seconds'])
        argv=life.cpu_args(prefix+'-recorder',root)+[
            '-e','FASTRSI_MODEL_CALLS='+life.BASE+'/model_calls',life.IMAGE,'timeout',limit,'bash','-lc',
            'source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/qwen_recording_proxy.py']
        life.launch('recorder',argv,root);life.health(49511,limit=30)
        argv=life.cpu_args(prefix+'-runner',root)+['-e','FASTRSI_PROBE_DIR='+life.BASE,
            life.IMAGE,'timeout',limit,'bash','-lc',
            'source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/probe_history_scope.py']
        status['status']='RUNNING';life.save(root/'STATUS.json',status)
        cid=life.launch('runner',argv,root);life.wait_exit(cid,int(limit))
        result=json.loads((root/'RESULT.json').read_text())
        assert result['status']=='COMPLETED' and len(result['calls'])==config['max_calls']
        status.update(status='COMPLETED',controller_seconds=result['elapsed_seconds'])
    except BaseException:
        status.update(status='FAILED',error=traceback.format_exc())
    finally:
        life.save(root/'STOP_RECEIPT.json',life.stop_all())
        status.update(finished_at=time.time(),full_wall_seconds=time.time()-start)
        life.save(root/'STATUS.json',status)
        print(json.dumps(status),flush=True)
    if status['status']!='COMPLETED':raise SystemExit(1)


if __name__=='__main__':main()

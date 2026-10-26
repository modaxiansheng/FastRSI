"""One bounded baseline stream; reuse audited service lifecycle without rerunning old campaigns."""
import argparse
import fcntl
import hashlib
import json
import shutil
import signal
import socket
import time
import traceback
import orchestrate_full_os as life


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='opportunity_screen_v1.json')
    args=parser.parse_args()
    assert '/' not in args.config and args.config.endswith('.json')
    config_path=life.PROJECT/'configs'/args.config
    config=json.loads(config_path.read_text())
    campaign=life.PROJECT/'runs'/config['experiment_id']
    campaign.mkdir(exist_ok=False)
    life.CAMPAIGN=campaign
    life.BASE='/project/runs/'+config['experiment_id']
    life.deadline=time.time()+config['campaign_timeout_seconds']
    started=time.time()
    status={'status':'PREFLIGHT','started_at':started};life.save(campaign/'STATUS.json',status)
    folder=campaign/'stream';folder.mkdir()
    lock=open(life.PROJECT/'runs/full_os_v1.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def interrupted(signum,frame):raise RuntimeError('Interrupted '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    shutil.copyfile(config_path,campaign/'PROTOCOL.json')
    for rel in ['scripts','src/fastrsi','tests','configs']:
        shutil.copytree(life.PROJECT/rel,campaign/'code'/rel,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    life.save(campaign/'CODE_MANIFEST.json',{str(p.relative_to(campaign/'code')):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in (campaign/'code').rglob('*') if p.is_file()})
    prefix=config.get('container_prefix','yuchen-fastrsi-opportunityv1')
    assert prefix.startswith('yuchen-fastrsi-')
    try:
        life.gate(folder)
        tests=campaign/'tests';tests.mkdir()
        argv=life.cpu_args(prefix+'-contracts',tests)+[life.IMAGE,'timeout','180','bash','-lc',
            'source /opt/dtk/env.sh && /project/envs/os-v1/bin/python '+life.BASE+'/code/tests/test_round_contracts.py']
        cid=life.launch('contracts',argv,tests);life.wait_exit(cid,180)
        life.save(tests/'STOP_RECEIPT.json',life.stop_all())
        life.gate(folder)
        shutil.copytree(life.PROJECT/'models/openpi_data',folder/'tokenizer_cache')
        containers={}
        for role,receipt in [('pi05','LAUNCH_pi05_v2.json'),('qwen','LAUNCH_qwen.json'),('target','LAUNCH_target.json')]:
            argv=json.loads((life.OLD/receipt).read_text())['argv']
            argv[argv.index('--name')+1]=prefix+'-'+role
            argv=[x.replace(str(life.OLD),str(folder)).replace('/project/runs/pilot_20261001',life.BASE+'/stream') for x in argv]
            if role=='pi05':
                argv=[x.replace(str(life.PROJECT/'models/openpi_data'),str(folder/'tokenizer_cache')) for x in argv]
            if role=='target':
                argv=[x.replace('/project/scripts/instrument_target.py',life.BASE+'/code/scripts/instrument_target.py')
                      .replace(':/project/scripts',':'+life.BASE+'/code/scripts') for x in argv]
            containers[role]=life.launch(role,argv,folder)
        life.health(49500);life.health(49510)
        until=min(life.deadline,time.time()+120)
        while True:
            try:
                with socket.create_connection(('127.0.0.1',49502),timeout=1):break
            except OSError:
                if time.time()>until:raise
                time.sleep(1)
        status['startup_seconds']=time.time()-started
        # Read language from the same frozen LIBERO package used by the target.
        assert config['suite'] in ('libero_spatial','libero_10')
        code=('import json; from libero.libero import benchmark; '
              'suite=benchmark.get_benchmark_dict()['+repr(config['suite'])+'](); '
              'print("FASTRSI_ROSTER="+json.dumps([{ "task_id":t,"init_state_id":i,'
              '"task_description":suite.get_task(t).language} '
              'for t in '+repr(config['task_ids'])+' for i in '+repr(config['init_state_ids'])+']))')
        receipt=life.command(['docker','exec',containers['target'],'python','-c',code],timeout=60)
        life.save(campaign/'ROSTER_SOURCE.json',receipt)
        assert receipt['returncode']==0,receipt['stderr']
        lines=[x for x in receipt['stdout'].splitlines() if x.startswith('FASTRSI_ROSTER=')]
        assert len(lines)==1
        life.save(campaign/'ROSTER.json',json.loads(lines[0].split('=',1)[1]))
        limit=str(config['controller_timeout_seconds'])
        argv=life.cpu_args(prefix+'-recorder',folder)+[
            '-e','FASTRSI_MODEL_CALLS='+life.BASE+'/stream/model_calls',
            life.IMAGE,'timeout',limit,'bash','-lc',
            'source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/qwen_recording_proxy.py']
        life.launch('recorder',argv,folder);life.health(49511,limit=30)
        argv=life.cpu_args(prefix+'-runner',folder)+[
            '-e','FASTRSI_ROUND_DIR='+life.BASE+'/stream','-e','FASTRSI_ARM=SCREEN','-e','FASTRSI_ROUND=1',
            life.IMAGE,'timeout',limit,'bash','-lc',
            'source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/run_opportunity_screen.py']
        status['status']='RUNNING';life.save(campaign/'STATUS.json',status)
        cid=life.launch('runner',argv,folder);life.wait_exit(cid,int(limit))
        result=json.loads((folder/'RESULT.json').read_text())
        assert result['status']=='COMPLETED' and len(result['cases'])==config['max_cases']
        status.update(status='COMPLETED',controller_seconds=result['elapsed_seconds'])
    except BaseException:
        status.update(status='FAILED',error=traceback.format_exc())
    finally:
        life.save(folder/'STOP_RECEIPT.json',life.stop_all())
        status.update(finished_at=time.time(),full_wall_seconds=time.time()-started)
        life.save(campaign/'STATUS.json',status)
        print(json.dumps(status),flush=True)
    if status['status']!='COMPLETED':raise SystemExit(1)


if __name__=='__main__':main()

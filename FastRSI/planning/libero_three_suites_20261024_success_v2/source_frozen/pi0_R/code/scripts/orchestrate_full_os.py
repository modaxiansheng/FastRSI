"""Single bounded campaign, fresh services per group, finally-stop by exact ID."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import time
import traceback
from urllib.request import build_opener,ProxyHandler

PROJECT=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
CAMPAIGN=PROJECT/'runs/full_os_v2_20261001'
OLD=PROJECT/'runs/pilot_20261001'
CORE=PROJECT.parent/'frozen_archive/extra-freeze-20260929T145255Z/frozen/assets-20260929-v1/dependencies/PhyAgentOS-core'
IMAGE='sha256:c98dca2210774e2311a0780146956ef3df3c6f0bb099b6bc671ab31befc42a81'
BASE='/project/runs/full_os_v2_20261001'
started=time.time()
deadline=started+5400
active=[]
status={'status':'PREFLIGHT','started_at':started,'groups':[]}


def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str))


def command(argv,timeout=60):
    p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
    return {'argv':argv,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}


def gate(folder):
    containers=command(['docker','ps','--format','{{.ID}} {{.Names}}'])
    hcu=command(['bash','-lc','source /opt/dtk/env.sh; source /opt/hyhal/env.sh; hy-smi --showuse --showmemuse'])
    holders=command(['bash','-lc','fuser -v /dev/kfd /dev/mkfd /dev/dri/render* 2>&1 || true'])
    save(folder/'PREFLIGHT.json',{'time':time.time(),'containers':containers,'hcu':hcu,'holders':holders})
    mem=re.findall(r'HCU\[(\d+)\].*?HCU memory use \(%\):\s*([0-9.]+)',hcu['stdout'])
    assert hcu['returncode']==0 and len(mem)==8 and all(float(v)==0 for _,v in mem),'Resource gate unavailable/busy'
    assert not containers['stdout'].strip(),'Existing container: stop and inspect, never launch a duplicate'
    for port in [49500,49502,49510,49511,49520]:
        with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',port))!=0,'Port busy'


def launch(role,argv,folder):
    receipt=command(argv)
    receipt['launched_at']=time.time()
    save(folder/('LAUNCH_'+role+'.json'),receipt)
    assert receipt['returncode']==0,receipt['stderr']
    cid=receipt['stdout'].strip()
    info=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
    assert info['Id']==cid and info['Name'].startswith('/yuchen-fastrsi-')
    active.append((cid,role,folder))
    return cid


def stop_all():
    rows=[]
    while active:
        cid,role,folder=active.pop()
        info=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
        assert info['Id']==cid and info['Name'].startswith('/yuchen-fastrsi-')
        if info['State']['Running']:command(['docker','stop','-t','10',cid],timeout=40)
        logs=subprocess.run(['docker','logs',cid],capture_output=True)
        (folder/(role+'.log')).write_bytes(logs.stdout+logs.stderr)
        after=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
        rows.append({'container':cid,'role':role,'name':info['Name'],'created':info['Created'],
                     'started':info['State']['StartedAt'],'finished':after['State']['FinishedAt'],
                     'running':after['State']['Running'],'exit_code':after['State']['ExitCode']})
    return rows


def health(port,path='/healthz',limit=240):
    opener=build_opener(ProxyHandler({}));until=min(deadline,time.time()+limit)
    while time.time()<until:
        try:
            with opener.open('http://127.0.0.1:'+str(port)+path,timeout=2) as response:
                if response.status==200:return
        except Exception:pass
        time.sleep(1)
    raise RuntimeError('Service readiness failed: '+str(port))


def cpu_args(name,folder):
    cache=PROJECT/'models/tiktoken_cache_v1'
    asset=cache/'9b5ad71b2ce5302211f9c61530b329a4922fc6a4'
    assert hashlib.sha256(asset.read_bytes()).hexdigest()=='223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7','Native token-count cache missing or changed'
    return ['docker','run','-d','--name',name,'--network','host','--cpus','8','--memory','16g',
            '-e','PYTHONDONTWRITEBYTECODE=1','-e','TIKTOKEN_CACHE_DIR=/tiktoken-cache',
            '-v',str(cache)+':/tiktoken-cache:ro',
            '-e',f'PYTHONPATH=/core:{BASE}/code/scripts:{BASE}/code/src:/project/envs/structured-decode-v1',
            '-v',str(PROJECT)+':/project:rw','-v',str(CORE)+':/core:ro',
            '-v',str(CAMPAIGN/'code/scripts/engine_overlay.py')+':/core/PhyAgentOS/verification/engine.py:ro',
            '-e','FASTRSI_TRACE='+str(folder/'OS_COST_EVENTS.jsonl').replace(str(PROJECT),'/project')]


def wait_exit(cid,limit=1200):
    until=min(deadline,time.time()+limit)
    while time.time()<until:
        state=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]['State']
        if not state['Running']:
            assert state['ExitCode']==0,'Container failed '+cid
            return
        time.sleep(2)
    raise TimeoutError('Bounded runtime expired')


def group(arm,round_id,smoke=False):
    folder=CAMPAIGN/arm/('round_'+str(round_id));folder.mkdir(parents=True,exist_ok=False)
    group_started=time.time();prefix='yuchen-fastrsi-fullv2-'+arm.lower()+'-r'+str(round_id)
    current={'arm':arm,'round':round_id,'status':'RUNNING','started_at':group_started}
    status['current']=current;save(CAMPAIGN/'STATUS.json',status)
    try:
        gate(folder)
        shutil.copytree(PROJECT/'models/openpi_data',folder/'tokenizer_cache')
        for role,receipt in [('pi05','LAUNCH_pi05_v2.json'),('qwen','LAUNCH_qwen.json'),('target','LAUNCH_target.json')]:
            argv=json.loads((OLD/receipt).read_text())['argv']
            argv[argv.index('--name')+1]=prefix+'-'+role
            argv=[x.replace(str(OLD),str(folder)).replace('/project/runs/pilot_20261001',str(folder).replace(str(PROJECT),'/project')) for x in argv]
            if role=='pi05':argv=[x.replace(str(PROJECT/'models/openpi_data'),str(folder/'tokenizer_cache')) for x in argv]
            if role=='target':
                argv=[x.replace('/project/scripts/instrument_target.py',BASE+'/code/scripts/instrument_target.py').replace(':/project/scripts',':'+BASE+'/code/scripts') for x in argv]
            launch(role,argv,folder)
        health(49500)
        health(49510)
        # TargetWS speaks WebSocket, so readiness is a TCP gate; execution validates protocol.
        until=time.time()+120
        while True:
            try:
                with socket.create_connection(('127.0.0.1',49502),timeout=1):break
            except OSError:
                if time.time()>until:raise
                time.sleep(1)
        ready=time.time()
        common=cpu_args(prefix+'-recorder',folder)
        common+=['-e','FASTRSI_MODEL_CALLS='+str(folder/'model_calls').replace(str(PROJECT),'/project'),
                 IMAGE,'timeout','1200','bash','-lc','source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+BASE+'/code/scripts/qwen_recording_proxy.py']
        launch('recorder',common,folder);health(49511,limit=30)
        argv=cpu_args(prefix+'-runner',folder)+['-e','FASTRSI_ROUND_DIR='+str(folder).replace(str(PROJECT),'/project'),
               '-e','FASTRSI_ARM='+arm,'-e','FASTRSI_ROUND='+str(round_id),'-e','FASTRSI_SMOKE='+str(int(smoke)),
               IMAGE,'timeout','1200','bash','-lc','source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+BASE+'/code/scripts/run_round.py']
        cid=launch('runner',argv,folder);wait_exit(cid)
        result=json.loads((folder/'RESULT.json').read_text());assert result['status']=='PASSED'
        if smoke:
            assert len(result['cases'])==2 and all(c['environment_success'] for c in result['cases'])
            assert result['cases'][1]['evaluation_updates_blocked']
            events=[json.loads(s) for s in (folder/'OS_COST_EVENTS.jsonl').read_text().splitlines()]
            assert any(x.get('event')=='decision_input' and x.get('mode')=='evaluation' and x.get('lessons_present') for x in events)
        current.update(status='PASSED',startup_seconds=ready-group_started,result_seconds=result['elapsed_seconds'])
    except Exception:
        current.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        stopped=stop_all();save(folder/'STOP_RECEIPT.json',stopped)
        current['complete_wall_seconds']=time.time()-group_started
        save(folder/'GROUP_COST.json',current)
        status['groups'].append(current);save(CAMPAIGN/'STATUS.json',status)


def main():
    CAMPAIGN.mkdir(exist_ok=False)
    lock=open(PROJECT/'runs/full_os_v1.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def interrupted(signum,frame):raise RuntimeError('Campaign interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    for rel in ['scripts','src/fastrsi','tests','configs']:
        shutil.copytree(PROJECT/rel,CAMPAIGN/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    manifest={str(p.relative_to(CAMPAIGN/'code')):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (CAMPAIGN/'code').rglob('*') if p.is_file()}
    save(CAMPAIGN/'CODE_MANIFEST.json',manifest)
    try:
        tests=CAMPAIGN/'tests';tests.mkdir()
        argv=cpu_args('yuchen-fastrsi-fullv2-contracts',tests)+[IMAGE,'timeout','180','bash','-lc',
            'source /opt/dtk/env.sh && /project/envs/os-v1/bin/python '+BASE+'/code/tests/test_round_contracts.py']
        cid=launch('contracts',argv,tests);wait_exit(cid,180)
        save(tests/'STOP_RECEIPT.json',stop_all())
        group('SMOKE',1,True)
        status['status']='BOUNDED_THREE_ARM_RUNNING';save(CAMPAIGN/'STATUS.json',status)
        for r,arms in [(1,['A','B','C']),(2,['C','A','B'])]:
            for arm in arms:
                assert time.time()<deadline
                group(arm,r)
        status['status']='COMPLETED'
    except Exception:
        status.update(status='FAILED',error=traceback.format_exc())
    finally:
        remaining=stop_all()
        status['remaining_stopped']=remaining;status['finished_at']=time.time()
        status['campaign_wall_seconds']=time.time()-started
        save(CAMPAIGN/'STATUS.json',status)
        print(json.dumps(status),flush=True)


if __name__=='__main__':main()

"""Bounded protocol/isolation gates before any native baseline development."""
import argparse
import fcntl
import hashlib
import json
import shutil
import signal
import time
import traceback
import orchestrate_full_os as life
from native_service_config import configure_qwen, model_env

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='native_protocol_v22_20261002')
    parser.add_argument('--model-config',default='configs/native_agent_qwen3_v1.json');args=parser.parse_args()
    cfg=json.loads((life.PROJECT/args.model_config).read_text())
    model_binding=json.loads((life.PROJECT/'planning/native_qwen3_v1/MODEL_BINDING.json').read_text())
    assert model_binding['status']=='VERIFIED'
    name=args.run_id
    assert name.startswith('native_protocol_') and all(c.isalnum() or c=='_' for c in name)
    campaign=life.PROJECT/'runs'/name;campaign.mkdir(exist_ok=False)
    life.CAMPAIGN=campaign;life.BASE='/project/runs/'+name;life.deadline=time.time()+1200
    started=time.time();status={'status':'PREFLIGHT','started_at':started}
    lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def stop(signum,frame):raise RuntimeError('Interrupted '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    for rel in ['scripts','src/fastrsi','tests','configs']:
        shutil.copytree(life.PROJECT/rel,campaign/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    life.save(campaign/'CODE_MANIFEST.json',{str(p.relative_to(campaign/'code')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (campaign/'code').rglob('*') if p.is_file()})
    life.save(campaign/'PROTOCOL.json',{'scope':'native Agent tool protocol and evaluation isolation acceptance','max_wall_seconds':1200,'max_agent_iterations':6,'planning_max_iterations':10,'minimal_probe_max_tokens':1024,'planning_max_tokens':2048,'context_window_tokens':65536,'models':[cfg['model_id']],'robot_episodes':0,'laya':False,'common_tool_extension':'append_yaml_item (user approved); no learning rule change'})
    life.save(campaign/'MODEL_CONFIG.json',cfg)
    life.save(campaign/'MODEL_BINDING.json',model_binding)
    tests=campaign/'tests';tests.mkdir();stream=campaign/'stream';stream.mkdir()
    try:
        life.gate(stream)
        argv=life.cpu_args('yuchen-fastrsi-'+name+'-tests',tests)+[life.IMAGE,'timeout','180','bash','-lc',
          'source /opt/dtk/env.sh && /project/envs/os-v1/bin/python '+life.BASE+'/code/tests/test_constrained_tools.py && /project/envs/os-v1/bin/python '+life.BASE+'/code/tests/test_tool_protocol.py && /project/envs/os-v1/bin/python '+life.BASE+'/code/tests/test_file_append.py && /project/envs/os-v1/bin/python '+life.BASE+'/code/tests/test_native_isolation.py && /project/envs/os-v1/bin/python '+life.BASE+'/code/tests/test_native_workspace.py && /project/envs/os-v1/bin/python '+life.BASE+'/code/tests/test_verifier_schema_contract.py && /project/envs/os-v1/bin/python '+life.BASE+'/code/tests/test_round_contracts.py']
        cid=life.launch('tests',argv,tests);life.wait_exit(cid,180)
        life.save(tests/'STOP_RECEIPT.json',life.stop_all())
        life.gate(stream)
        argv=json.loads((life.OLD/'LAUNCH_qwen.json').read_text())['argv']
        argv[argv.index('--name')+1]='yuchen-fastrsi-'+name+'-qwen'
        argv=[x.replace(str(life.OLD),str(stream)) for x in argv]
        argv=configure_qwen(argv,cfg,life)
        life.launch('qwen',argv,stream);life.health(49510,limit=240)
        status.update(status='RUNNING',startup_seconds=time.time()-started)
        life.save(campaign/'STATUS.json',status)
        argv=life.cpu_args('yuchen-fastrsi-'+name+'-runner',stream)+model_env(cfg)+['-e','FASTRSI_NATIVE_RUN='+life.BASE+'/stream',life.IMAGE,'timeout','600','bash','-lc',
          'source /opt/dtk/env.sh && /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/probe_history_replay.py && /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/probe_native_agent.py && /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/probe_native_verifier.py']
        cid=life.launch('runner',argv,stream);life.wait_exit(cid,600)
        assert json.loads((stream/'failed_request_replay/RESULT.json').read_text())['status']=='PASSED'
        assert json.loads((stream/'planning_long_history_probe/RESULT.json').read_text())['status']=='PASSED'
        assert json.loads((stream/'PROTOCOL_AGENT_RESULT.json').read_text())['status']=='PASSED'
        assert json.loads((stream/'planning_probe/RESULT.json').read_text())['status']=='PASSED'
        assert json.loads((stream/'planning_history_probe/RESULT.json').read_text())['status']=='PASSED'
        assert json.loads((stream/'VERIFIER_REPLAY_RESULT.json').read_text())['status']=='PASSED'
        status['status']='COMPLETED'
    except BaseException:
        status.update(status='FAILED',error=traceback.format_exc())
    finally:
        life.save(stream/'STOP_RECEIPT.json',life.stop_all())
        status.update(finished_at=time.time(),full_wall_seconds=time.time()-started)
        life.save(campaign/'STATUS.json',status);print(json.dumps(status),flush=True)
    if status['status']!='COMPLETED':raise SystemExit(1)

if __name__=='__main__':main()

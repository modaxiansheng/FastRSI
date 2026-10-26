"""Bounded native memory rollover probe after the real baseline completes."""
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
    name = 'native_memory_v3_20261002'
    gate = life.PROJECT/'runs/native_baseline_v5_20261002'
    assert json.loads((gate/'STATUS.json').read_text())['status'] == 'COMPLETED'
    cfg = json.loads((gate/'MODEL_CONFIG.json').read_text())
    tested = json.loads((gate/'CODE_MANIFEST.json').read_text())
    regression=life.PROJECT/'runs/native_memory_budget_regression_v2_20261002'
    assert json.loads((regression/'STATUS.json').read_text())['status']=='COMPLETED'
    repaired=json.loads((regression/'CODE_MANIFEST.json').read_text())
    for rel in ['scripts/qwen_agent_server.py', 'scripts/native_service_config.py',
                'scripts/run_native_baseline.py', 'src/fastrsi/native_binding.py',
                'src/fastrsi/constrained_tools.py', 'src/fastrsi/tool_protocol.py', 'src/fastrsi/file_append.py']:
        expected=repaired[rel] if rel in ('scripts/qwen_agent_server.py','scripts/run_native_baseline.py') else tested[rel]
        assert hashlib.sha256((life.PROJECT/rel).read_bytes()).hexdigest() == expected
    assert hashlib.sha256((life.PROJECT/'src/fastrsi/decoding_policy.py').read_bytes()).hexdigest()==repaired['src/fastrsi/decoding_policy.py']
    campaign = life.PROJECT/'runs'/name; campaign.mkdir(exist_ok=False)
    life.CAMPAIGN = campaign; life.BASE = '/project/runs/'+name; life.deadline = time.time()+1200
    started = time.time(); status = {'status': 'PREFLIGHT', 'started_at': started}
    lock = open(life.PROJECT/'runs/full_os_v1.lock', 'a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    def stop(signum, frame):
        raise RuntimeError('Interrupted '+str(signum))
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    for rel in ['scripts', 'src/fastrsi', 'tests', 'configs']:
        shutil.copytree(life.PROJECT/rel, campaign/'code'/rel, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    life.save(campaign/'CODE_MANIFEST.json', {str(p.relative_to(campaign/'code')): hashlib.sha256(p.read_bytes()).hexdigest() for p in (campaign/'code').rglob('*') if p.is_file()})
    life.save(campaign/'MODEL_CONFIG.json', cfg)
    life.save(campaign/'MODEL_BINDING.json', json.loads((gate/'MODEL_BINDING.json').read_text()))
    life.save(campaign/'PROTOCOL.json', {'source': str(gate), 'robot_episodes': 0,
        'max_wall_seconds': 1200, 'max_runner_seconds': 900,
        'memory_max_tokens': 4096, 'planning_max_tokens': 2048,
        'memory_temperature':0.7,'sampling_seed':0,'planning_temperature':0,
        'memory_budget_reason': 'Use native provider defaults for complete archival; service now honors temperature and retains checkpoint top-p/top-k. Planning remains greedy. v1 and v2 greedy archival hit their 2048/4096 limits.',
        'trigger': 'One original /new after the completed two-instance development batch',
        'memory_threshold_changed': False, 'forced_per_round_consolidation': False,
        'scope': 'Memory lifecycle acceptance, not effectiveness comparison; all cost recorded separately'})
    stream = campaign/'stream'; stream.mkdir(); life.save(campaign/'STATUS.json', status)
    try:
        life.gate(stream)
        argv = json.loads((life.OLD/'LAUNCH_qwen.json').read_text())['argv']
        argv[argv.index('--name')+1] = 'yuchen-fastrsi-'+name+'-qwen'
        argv = [x.replace(str(life.OLD), str(stream)) for x in argv]
        life.launch('qwen', configure_qwen(argv, cfg, life), stream); life.health(49510, limit=240)
        status.update(status='RUNNING', startup_seconds=time.time()-started); life.save(campaign/'STATUS.json', status)
        argv = life.cpu_args('yuchen-fastrsi-'+name+'-runner', stream)+model_env(cfg)+[
            '-e', 'FASTRSI_NATIVE_RUN='+life.BASE+'/stream', life.IMAGE, 'timeout', '900', 'bash', '-lc',
            'source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/probe_native_memory.py']
        cid = life.launch('runner', argv, stream); life.wait_exit(cid, 900)
        assert json.loads((stream/'RESULT.json').read_text())['status'] == 'COMPLETED'
        status['status'] = 'COMPLETED'
    except BaseException:
        status.update(status='FAILED', error=traceback.format_exc())
    finally:
        life.save(stream/'STOP_RECEIPT.json', life.stop_all())
        status.update(finished_at=time.time(), full_wall_seconds=time.time()-started)
        life.save(campaign/'STATUS.json', status); print(json.dumps(status), flush=True)
    if status['status'] != 'COMPLETED':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

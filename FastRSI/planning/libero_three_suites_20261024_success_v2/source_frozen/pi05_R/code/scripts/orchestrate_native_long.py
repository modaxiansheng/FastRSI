"""Four predeclared long-task native development cases; no training or efficacy comparison.
"""
import fcntl
import argparse
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
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='native_long_v1_20261002');args=parser.parse_args()
    name=args.run_id
    assert name.startswith('native_long_') and all(c.isalnum() or c=='_' for c in name)
    memory_run=life.PROJECT/'runs/native_workflow_v2_20261002'
    memory=json.loads((memory_run/'stream/terminal_memory/RESULT.json').read_text())
    assert memory['status']=='PASSED' and memory['source_unchanged_after'] is True
    workflow=life.PROJECT/'runs/native_workflow_v3_20261002'
    assert json.loads((workflow/'STATUS.json').read_text())['status']=='COMPLETED'
    decoder=life.PROJECT/'runs/native_stage1_v1_20261002'
    assert all(x['all_output_tokens_equal'] for x in json.loads((decoder/'DECODER_RESULT.json').read_text())['summary'])
    tested=json.loads((workflow/'CODE_MANIFEST.json').read_text())
    for rel in ['scripts/qwen_agent_server.py','scripts/native_service_config.py','src/fastrsi/constrained_tools.py','src/fastrsi/decoding_policy.py','src/fastrsi/terminal_results.py','src/fastrsi/native_binding.py']:
        assert hashlib.sha256((life.PROJECT/rel).read_bytes()).hexdigest()==tested[rel]
    lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    campaign=life.PROJECT/'runs'/name;campaign.mkdir(exist_ok=False)
    life.CAMPAIGN=campaign;life.BASE='/project/runs/'+name
    started=time.time();life.deadline=started+2400;status={'status':'PREFLIGHT','started_at':started}
    def interrupt(signum,frame):raise RuntimeError('Interrupted '+str(signum))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    for rel in ['scripts','src/fastrsi','tests','configs']:
        shutil.copytree(life.PROJECT/rel,campaign/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    life.save(campaign/'CODE_MANIFEST.json',{str(p.relative_to(campaign/'code')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (campaign/'code').rglob('*') if p.is_file()})
    cfg=json.loads((life.PROJECT/'configs/native_agent_qwen3_v2.json').read_text());life.save(campaign/'MODEL_CONFIG.json',cfg)
    life.save(campaign/'MODEL_BINDING.json',json.loads((memory_run/'MODEL_BINDING.json').read_text()))
    core_names=['agent/loop.py','agent/context.py','agent/memory.py','agent/skills.py','agent/session_verifier.py','runtime/watchdog/result_writer.py','skills/benchmarking/SKILL.md']
    life.save(campaign/'CORE_SOURCE_BINDING.json',{p:hashlib.sha256((life.CORE/'PhyAgentOS'/p).read_bytes()).hexdigest() for p in core_names})
    old_roster=life.PROJECT/'runs/opportunity_long_v1_20261002/ROSTER.json'
    cases=[{**c,'id':f'native_long_t{c["task_id"]}_i{c["init_state_id"]}',
        'mode':'development','suite':'libero_10','max_steps':520,'require_discovery_reads':True,'require_clean_model_calls':True}
        for c in json.loads(old_roster.read_text())]
    assert [(c['task_id'],c['init_state_id']) for c in cases]==[(0,8),(3,8),(6,8),(9,8)]
    life.save(campaign/'ROSTER.json',cases)
    life.save(campaign/'EXPERIENCE_BINDING.json',{'initial':'empty','cross_case':'complete admitted workspace; conflicting candidate retained outside next-session source',
        'seen_scope':'These instances were previously executed through the subchain; not unseen tests',
        'roster_source_sha256':hashlib.sha256(old_roster.read_bytes()).hexdigest()})
    life.save(campaign/'PROTOCOL.json',{'scope':'Native development lifecycle and cost screen, not A/B/C',
        'max_cases':4,'max_attempts':12,'max_steps_per_attempt':520,'max_campaign_seconds':2400,
        'agent_max_iterations':10,'memory_trigger_threshold':65536,'workflow':'unguided native reads',
        'experience':'Empty initial workspace; native feedback retained, only admitted workspace carried forward',
        'experience_admission':'guarded_v1; deterministic common A/B/C infrastructure; historical raw run remains separate',
        'native_terminal_report':True,'forced_consolidation':False,'laya':False,'training':False,
        'stop':'Stop on chain/protocol failure or time budget; retain ordinary task failures and all costs',
        'decision_freedom':'Existing task_description only; costs screened before expanding legal controls',
        'not_claimed':['Learning benefit','Unseen task generalization','Final success noninferiority']})
    stream=campaign/'stream';stream.mkdir();life.save(campaign/'STATUS.json',status)
    try:
        life.gate(stream)
        tests=campaign/'tests';tests.mkdir()
        argv=life.cpu_args('yuchen-fastrsi-'+name+'-tests',tests)+[life.IMAGE,'timeout','90','bash','-lc',
            'source /opt/dtk/env.sh && /project/envs/os-v1/bin/python '+life.BASE+'/code/tests/test_native_workspace.py && /project/envs/os-v1/bin/python '+life.BASE+'/code/tests/test_terminal_results.py']
        cid=life.launch('tests',argv,tests);life.wait_exit(cid,90)
        shutil.copytree(life.PROJECT/'models/openpi_data',stream/'tokenizer_cache')
        containers={}
        for role,receipt in [('pi05','LAUNCH_pi05_v2.json'),('qwen','LAUNCH_qwen.json'),('target','LAUNCH_target.json')]:
            argv=json.loads((life.OLD/receipt).read_text())['argv'];argv[argv.index('--name')+1]='yuchen-fastrsi-'+name+'-'+role
            argv=[x.replace(str(life.OLD),str(stream)).replace('/project/runs/pilot_20261001',life.BASE+'/stream') for x in argv]
            if role=='pi05':argv=[x.replace(str(life.PROJECT/'models/openpi_data'),str(stream/'tokenizer_cache')) for x in argv]
            if role=='target':argv=[x.replace('/project/scripts/instrument_target.py',life.BASE+'/code/scripts/instrument_target.py').replace(':/project/scripts',':'+life.BASE+'/code/scripts') for x in argv]
            if role=='qwen':argv=configure_qwen(argv,cfg,life)
            containers[role]=life.launch(role,argv,stream)
        life.health(49500);life.health(49510)
        until=time.time()+120
        while True:
            try:
                with socket.create_connection(('127.0.0.1',49502),timeout=1):break
            except OSError:
                if time.time()>until:raise
                time.sleep(1)
        code="import json; from libero.libero import benchmark; s=benchmark.get_benchmark_dict()['libero_10'](); print('ROSTER='+json.dumps([s.get_task(i).language for i in [0,3,6,9]]))"
        receipt=life.command(['docker','exec',containers['target'],'python','-c',code],timeout=60)
        life.save(campaign/'ROSTER_SOURCE_LIVE.json',receipt)
        assert receipt['returncode']==0
        actual=json.loads(next(x[7:] for x in receipt['stdout'].splitlines() if x.startswith('ROSTER=')))
        assert actual==[c['task_description'] for c in cases]
        status.update(status='RUNNING',startup_seconds=time.time()-started);life.save(campaign/'STATUS.json',status)
        argv=life.cpu_args('yuchen-fastrsi-'+name+'-runner',stream)+model_env(cfg)+[
            '-e','FASTRSI_NATIVE_RUN='+life.BASE+'/stream','-e','FASTRSI_CONSUME_TERMINAL=1',
            '-e','FASTRSI_EXPERIENCE_ADMISSION=guarded_v1',
            life.IMAGE,'timeout','2100','bash','-lc','source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/run_native_baseline.py']
        cid=life.launch('runner',argv,stream);life.wait_exit(cid,2100)
        result=json.loads((stream/'RESULT.json').read_text());assert result['status']=='COMPLETED'
        assert len(result['cases'])==4
        for row in result['cases']:
            assert row['terminal_report']['status']=='PASSED'
            assert row['chain_status']=='PASSED'
        requests=list(stream.glob('service_calls/*/REQUEST.json'))
        receipts=list(stream.glob('service_calls/*/RECEIPT.json'))
        assert len(requests)==len(receipts),'Incomplete model request'
        assert all(json.loads(p.read_text())['status']=='OK' for p in receipts),'Model protocol rejection'
        status.update(status='COMPLETED',controller_seconds=result['seconds'])
    except BaseException:
        status.update(status='FAILED',error=traceback.format_exc())
    finally:
        life.save(stream/'STOP_RECEIPT.json',life.stop_all());status.update(finished_at=time.time(),full_wall_seconds=time.time()-started)
        life.save(campaign/'STATUS.json',status);print(json.dumps(status),flush=True)
    if status['status']!='COMPLETED':raise SystemExit(1)


if __name__=='__main__':main()

"""One fixed-experience evaluation of the repaired native terminal lifecycle.

No development, forced consolidation, routing training or efficacy comparison.
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
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='native_terminal_eval_v2_20261002');args=parser.parse_args()
    name=args.run_id
    assert name.startswith('native_terminal_eval_') and all(c.isalnum() or c=='_' for c in name)
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
    started=time.time();life.deadline=started+1500;status={'status':'PREFLIGHT','started_at':started}
    def interrupt(signum,frame):raise RuntimeError('Interrupted '+str(signum))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    for rel in ['scripts','src/fastrsi','tests','configs']:
        shutil.copytree(life.PROJECT/rel,campaign/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    life.save(campaign/'CODE_MANIFEST.json',{str(p.relative_to(campaign/'code')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (campaign/'code').rglob('*') if p.is_file()})
    cfg=json.loads((life.PROJECT/'configs/native_agent_qwen3_v2.json').read_text());life.save(campaign/'MODEL_CONFIG.json',cfg)
    life.save(campaign/'MODEL_BINDING.json',json.loads((memory_run/'MODEL_BINDING.json').read_text()))
    core_names=['agent/loop.py','agent/context.py','agent/memory.py','agent/skills.py','agent/session_verifier.py','runtime/watchdog/result_writer.py','skills/benchmarking/SKILL.md']
    life.save(campaign/'CORE_SOURCE_BINDING.json',{p:hashlib.sha256((life.CORE/'PhyAgentOS'/p).read_bytes()).hexdigest() for p in core_names})
    case={'id':'eval_terminal_t4_i23','mode':'evaluation','conversation_key':'cli:evaluation','task_id':4,'init_state_id':23,
          'task_description':'pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate','require_discovery_reads':True}
    life.save(campaign/'ROSTER.json',[case])
    source=memory_run/'stream/terminal_memory/workspace'
    source_hashes={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
    life.save(campaign/'EXPERIENCE_BINDING.json',{'source':str(source),'hashes':source_hashes,'read_only_source':True})
    life.save(campaign/'PROTOCOL.json',{'scope':'One known-instance full native evaluation and report-guard integration, not A/B/C or success-rate estimation',
        'max_cases':1,'max_attempts':3,'max_steps_per_attempt':220,'max_campaign_seconds':1500,
        'agent_max_iterations':10,'memory_trigger_threshold':65536,'workflow':'unguided native reads',
        'experience':'Fixed development memory clone; task context retained; all persistent learning disabled through close and background callbacks',
        'native_terminal_report':True,'forced_consolidation':False,'laya':False,'training':False})
    stream=campaign/'stream';stream.mkdir();life.save(campaign/'STATUS.json',status)
    try:
        life.gate(stream)
        tests=campaign/'tests';tests.mkdir()
        argv=life.cpu_args('yuchen-fastrsi-'+name+'-tests',tests)+[life.IMAGE,'timeout','90','bash','-lc',
            'source /opt/dtk/env.sh && /project/envs/os-v1/bin/python '+life.BASE+'/code/tests/test_native_workspace.py && /project/envs/os-v1/bin/python '+life.BASE+'/code/tests/test_terminal_results.py']
        cid=life.launch('tests',argv,tests);life.wait_exit(cid,90)
        shutil.copytree(life.PROJECT/'models/openpi_data',stream/'tokenizer_cache')
        for role,receipt in [('pi05','LAUNCH_pi05_v2.json'),('qwen','LAUNCH_qwen.json'),('target','LAUNCH_target.json')]:
            argv=json.loads((life.OLD/receipt).read_text())['argv'];argv[argv.index('--name')+1]='yuchen-fastrsi-'+name+'-'+role
            argv=[x.replace(str(life.OLD),str(stream)).replace('/project/runs/pilot_20261001',life.BASE+'/stream') for x in argv]
            if role=='pi05':argv=[x.replace(str(life.PROJECT/'models/openpi_data'),str(stream/'tokenizer_cache')) for x in argv]
            if role=='target':argv=[x.replace('/project/scripts/instrument_target.py',life.BASE+'/code/scripts/instrument_target.py').replace(':/project/scripts',':'+life.BASE+'/code/scripts') for x in argv]
            if role=='qwen':argv=configure_qwen(argv,cfg,life)
            life.launch(role,argv,stream)
        life.health(49500);life.health(49510)
        until=time.time()+120
        while True:
            try:
                with socket.create_connection(('127.0.0.1',49502),timeout=1):break
            except OSError:
                if time.time()>until:raise
                time.sleep(1)
        status.update(status='RUNNING',startup_seconds=time.time()-started);life.save(campaign/'STATUS.json',status)
        argv=life.cpu_args('yuchen-fastrsi-'+name+'-runner',stream)+model_env(cfg)+[
            '-e','FASTRSI_NATIVE_RUN='+life.BASE+'/stream','-e','FASTRSI_CONSUME_TERMINAL=1',
            '-e','FASTRSI_INITIAL_EXPERIENCE=/project/runs/native_workflow_v2_20261002/stream/terminal_memory/workspace',
            life.IMAGE,'timeout','1100','bash','-lc','source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/run_native_baseline.py']
        cid=life.launch('runner',argv,stream);life.wait_exit(cid,1100)
        result=json.loads((stream/'RESULT.json').read_text());assert result['status']=='COMPLETED'
        row=result['cases'][0];assert row['terminal_report']['status']=='PASSED'
        assert row['experience_before']==row['experience_after']
        assert source_hashes=={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
        status.update(status='COMPLETED',controller_seconds=result['seconds'])
    except BaseException:
        status.update(status='FAILED',error=traceback.format_exc())
    finally:
        life.save(stream/'STOP_RECEIPT.json',life.stop_all());status.update(finished_at=time.time(),full_wall_seconds=time.time()-started)
        life.save(campaign/'STATUS.json',status);print(json.dumps(status),flush=True)
    if status['status']!='COMPLETED':raise SystemExit(1)


if __name__=='__main__':main()

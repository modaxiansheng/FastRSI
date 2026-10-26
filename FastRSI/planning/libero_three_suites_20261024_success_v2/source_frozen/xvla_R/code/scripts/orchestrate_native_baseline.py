"""Two development instances and one fixed-experience evaluation; no A/B/C."""
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
from native_service_config import configure_qwen, model_env

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='native_baseline_v6_20261002')
    parser.add_argument('--protocol-gate',default='native_protocol_v22_20261002')
    parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    assert args.run_id.startswith('native_baseline_') and all(c.isalnum() or c=='_' for c in args.run_id)
    gate=life.PROJECT/'runs'/args.protocol_gate/'STATUS.json'
    assert json.loads(gate.read_text())['status']=='COMPLETED','Real-model protocol gate not passed'
    cfg=json.loads((gate.parent/'MODEL_CONFIG.json').read_text())
    tested=json.loads((gate.parent/'CODE_MANIFEST.json').read_text())
    budget_gate=None
    for rel in ['src/fastrsi/decoding_policy.py','src/fastrsi/constrained_tools.py','tests/test_constrained_tools.py','src/fastrsi/file_append.py','tests/test_file_append.py','scripts/probe_history_replay.py','scripts/probe_native_agent.py','src/fastrsi/tool_protocol.py','scripts/qwen_agent_server.py','src/fastrsi/native_binding.py','tests/test_native_isolation.py','scripts/native_service_config.py','scripts/run_native_baseline.py','tests/test_native_workspace.py','src/fastrsi/verification_engine.py','tests/test_verifier_schema_contract.py']:
        digest=hashlib.sha256((life.PROJECT/rel).read_bytes()).hexdigest()
        if rel in ('scripts/run_native_baseline.py','scripts/qwen_agent_server.py','src/fastrsi/decoding_policy.py') and digest!=tested.get(rel):
            # The post-v22 request change is native archival generation settings.
            # Combine a real MemoryStore dispatch regression with the real
            # model's completed /new at that exact budget. Planning is unchanged.
            regression=life.PROJECT/'runs/native_memory_budget_regression_v2_20261002'
            memory=life.PROJECT/'runs/native_memory_v3_20261002'
            assert json.loads((regression/'STATUS.json').read_text())['status']=='COMPLETED'
            assert json.loads((memory/'STATUS.json').read_text())['status']=='COMPLETED'
            assert json.loads((memory/'stream/rollover/RESULT.json').read_text())['status']=='PASSED'
            verified=json.loads((regression/'CODE_MANIFEST.json').read_text())
            assert digest==verified[rel]
            assert hashlib.sha256((life.PROJECT/'tests/test_native_memory_budget.py').read_bytes()).hexdigest()==verified['tests/test_native_memory_budget.py']
            if budget_gate is None:
                budget_gate={'regression':str(regression),'real_model':str(memory),'covered_code':{},
                    'scope':'Native save_memory uses 4096 and temperature 0.7; the service honors positive temperature with checkpoint sampling settings and seed 0. Planning remains 2048 and greedy. No memory trigger changed.'}
            budget_gate['covered_code'][rel]=digest
        else:
            assert digest==tested[rel], 'Tool service changed after real-model gate'
    if args.check_only:
        print(json.dumps({'status':'PASSED','protocol_gate':str(gate),'memory_budget_gate':budget_gate,
            'scope':'Current common code is bound to completed protocol, native baseline and memory repair evidence; no workload started.'}),flush=True)
        return
    campaign=life.PROJECT/'runs'/args.run_id;campaign.mkdir(exist_ok=False)
    life.CAMPAIGN=campaign;life.BASE='/project/runs/'+args.run_id;life.deadline=time.time()+2100
    started=time.time();status={'status':'PREFLIGHT','started_at':started}
    lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def stop(signum,frame):raise RuntimeError('Interrupted '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    for rel in ['scripts','src/fastrsi','tests','configs']:
        shutil.copytree(life.PROJECT/rel,campaign/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    life.save(campaign/'CODE_MANIFEST.json',{str(p.relative_to(campaign/'code')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (campaign/'code').rglob('*') if p.is_file()})
    life.save(campaign/'MODEL_CONFIG.json',cfg)
    life.save(campaign/'MODEL_BINDING.json',json.loads((gate.parent/'MODEL_BINDING.json').read_text()))
    core_files=['PhyAgentOS/agent/loop.py','PhyAgentOS/agent/context.py','PhyAgentOS/agent/memory.py',
                'PhyAgentOS/agent/skills.py','PhyAgentOS/agent/session_verifier.py','PhyAgentOS/cli/commands.py',
                'PhyAgentOS/runtime/watchdog/result_writer.py','PhyAgentOS/runtime/workspace/manager.py',
                'PhyAgentOS/templates/TARGETS.md','PhyAgentOS/templates/SKILLRUNTIME.md','PhyAgentOS/skills/benchmarking/SKILL.md']
    life.save(campaign/'CORE_SOURCE_BINDING.json',{name:hashlib.sha256((life.CORE/name).read_bytes()).hexdigest() for name in core_files})
    source=life.PROJECT.parent/'evaluation/table4-full-20260929-v1/output/pi05-libero_spatial-t04/libero_spatial-t4-i20/RESULT.json'
    historical=json.loads(source.read_text());language=historical['attempts'][0]['task_description']
    roster=[dict(id='dev1_t4_i20',mode='development',task_id=4,init_state_id=20,task_description=language),
            dict(id='dev2_t4_i23',mode='development',task_id=4,init_state_id=23,task_description=language),
            dict(id='eval_t4_i26',mode='evaluation',task_id=4,init_state_id=26,task_description=language)]
    life.save(campaign/'ROSTER.json',roster)
    life.save(campaign/'PROTOCOL.json',{'scope':'native Agent binding; not efficacy comparison',
        'max_cases':3,'max_attempts_per_case':3,'max_steps_per_attempt':220,'max_wall_seconds':2100,
        'context_window_tokens':65536,'agent_max_iterations':10,'agent_max_tokens':2048,'verifier_max_tokens':384,
        'memory_max_tokens':4096,'memory_temperature':0.7,'sampling_seed':0,'memory_budget_gate':budget_gate,
        'selection':'Two historically difficult spatial development instances, fixed before this run; evaluation is an isolation probe, not unseen final testing',
        'language_source':str(source),'language_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'protocol_gate':str(gate),'protocol_gate_sha256':hashlib.sha256(gate.read_bytes()).hexdigest(),
        'laya':False,'training':False,'extra_reflection_or_forced_consolidation':False,'lifecycle':'native AgentLoop with background verifier; validated queue dispatch starts watchdog before the Agent turn ends; await terminal runtime after the turn'})
    stream=campaign/'stream';stream.mkdir()
    try:
        life.gate(stream)
        tests=campaign/'tests';tests.mkdir()
        argv=life.cpu_args('yuchen-fastrsi-'+args.run_id+'-tests',tests)+[life.IMAGE,'timeout','180','bash','-lc',
          'source /opt/dtk/env.sh && /project/envs/os-v1/bin/python '+life.BASE+'/code/tests/test_native_workspace.py']
        cid=life.launch('tests',argv,tests);life.wait_exit(cid,180)
        life.save(tests/'STOP_RECEIPT.json',life.stop_all())
        life.gate(stream)
        shutil.copytree(life.PROJECT/'models/openpi_data',stream/'tokenizer_cache')
        for role,receipt in [('pi05','LAUNCH_pi05_v2.json'),('qwen','LAUNCH_qwen.json'),('target','LAUNCH_target.json')]:
            argv=json.loads((life.OLD/receipt).read_text())['argv']
            argv[argv.index('--name')+1]='yuchen-fastrsi-'+args.run_id+'-'+role
            argv=[x.replace(str(life.OLD),str(stream)).replace('/project/runs/pilot_20261001',life.BASE+'/stream') for x in argv]
            if role=='pi05':argv=[x.replace(str(life.PROJECT/'models/openpi_data'),str(stream/'tokenizer_cache')) for x in argv]
            if role=='target':
                argv=[x.replace('/project/scripts/instrument_target.py',life.BASE+'/code/scripts/instrument_target.py').replace(':/project/scripts',':'+life.BASE+'/code/scripts') for x in argv]
            if role=='qwen':
                argv=configure_qwen(argv,cfg,life)
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
        argv=life.cpu_args('yuchen-fastrsi-'+args.run_id+'-runner',stream)+model_env(cfg)+[
            '-e','FASTRSI_NATIVE_RUN='+life.BASE+'/stream',life.IMAGE,'timeout','1700','bash','-lc',
            'source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/run_native_baseline.py']
        cid=life.launch('runner',argv,stream);life.wait_exit(cid,1700)
        result=json.loads((stream/'RESULT.json').read_text());assert result['status']=='COMPLETED'
        status.update(status='COMPLETED',controller_seconds=result['seconds'])
    except BaseException:
        status.update(status='FAILED',error=traceback.format_exc())
    finally:
        life.save(stream/'STOP_RECEIPT.json',life.stop_all())
        status.update(finished_at=time.time(),full_wall_seconds=time.time()-started)
        life.save(campaign/'STATUS.json',status);print(json.dumps(status),flush=True)
    if status['status']!='COMPLETED':raise SystemExit(1)

if __name__=='__main__':main()

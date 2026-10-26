"""A bounded slice of one frozen native development stream; no ABC or training."""
import argparse,fcntl,hashlib,json,shutil,signal,socket,sys,time,traceback
from pathlib import Path
import orchestrate_full_os as life
from native_service_config import configure_qwen,model_env
sys.path.insert(0,str(life.PROJECT/'src'))
from fastrsi.experience_admission import validate_snapshot

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def tree(root):return {str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',default='representative_native_v1_20261002');ap.add_argument('--stream',type=int,default=0);ap.add_argument('--roots',type=int,default=3);args=ap.parse_args()
    assert args.campaign.startswith('representative_native_') and args.campaign.replace('_','').isalnum()
    assert args.stream in range(3) and 1<=args.roots<=5
    lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    master=life.PROJECT/'runs'/args.campaign;planning=life.PROJECT/'planning/representative_coverage_v1'
    if not master.exists():
        assert load(life.PROJECT/'runs/verifier_labels_v2_20261002/STATUS.json')['status']=='COMPLETED'
        master.mkdir();started=time.time()
        for rel in ['scripts','src/fastrsi','tests','configs']:shutil.copytree(life.PROJECT/rel,master/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        life.save(master/'CODE_MANIFEST.json',tree(master/'code'))
        for rel in ['PROPOSED_ROSTER.json','RESERVED_SPLIT.json','TASK_BINDING.json','RUNTIME_TASKS.json','EXECUTED_EXPOSURE_V2.json']:shutil.copyfile(planning/rel,master/rel)
        prior=life.PROJECT/'runs/session_flow_v1_20261002'
        for rel in ['MODEL_CONFIG.json','MODEL_BINDING.json']:shutil.copyfile(prior/rel,master/rel)
        source=life.PROJECT/'runs/native_admission_v1_20261002/stream/recovered_clean/accepted_experience'
        validate_snapshot(source);shutil.copytree(source,master/'initial_experience');validate_snapshot(master/'initial_experience')
        core=['agent/loop.py','agent/context.py','agent/memory.py','agent/session_verifier.py','verification/service.py','runtime/watchdog/result_writer.py','skills/benchmarking/SKILL.md']
        life.save(master/'CORE_SOURCE_BINDING.json',{r:digest(life.CORE/'PhyAgentOS'/r) for r in core})
        life.save(master/'PROTOCOL.json',{'scope':'S1 full native development baseline screening; no efficacy ABC','created_at':started,
          'campaign_absolute_deadline':started+28800,'maximum_root_attempts':60,'planned_roots':30,'extension_to_50':'Not scheduled; balanced extension requires pre-outcome roster and reason',
          'maximum_environment_executions':180,'maximum_per_root':3,'max_batch_wall_seconds':2700,'max_batch_roots':5,
          'technical_retries':'None automatically; any retry keeps original identity and costs, consumes shared cap',
          'schedule':'Three independent streams, each in frozen task order. Partial batches resume only from admitted source; failure stops further batches',
          'roster_sha256':digest(master/'PROPOSED_ROSTER.json'),'reserved_split_sha256':digest(master/'RESERVED_SPLIT.json'),
          'initial_seal_sha256':digest(master/'initial_experience/.FASTRSI_ADMITTED.json'),
          'lifecycle':'One fresh set of services per batch, resident across roots; include idle waits, acquisition, all failures and shutdown in master elapsed wall',
          'native_memory_trigger':'Unchanged; no forced consolidation; independent initial copies per stream','public_repairs':'Same tool protocol, admission and explicit initial/final image metadata for all roots',
          'session_constructor':'Native AgentLoop; public mechanical prototype remains available but is not enabled in S1. No learned controller is being compared',
          'evaluation':'No final test pool runs here. Development feedback allowed; any later independent evaluation must freeze persistence through close/background',
          'generation':'Same model config, planning 2048, verification 384, native archival budget/trigger unchanged','max_agent_iterations':10,
          'resources':'187, fixed existing image, pi05 card0/Qwen card1, each8 CPU quota, serial roots, torch/environment seed7',
          'stop':'First resource/budget/isolation/protocol failure; ordinary task failures retained and admission quarantines conflicts; never retry to wash out failures',
          'costs':'Complete master elapsed wall and per-batch wall; model spans, token, capacity-time and environment interactions separately, no overlap summation',
          'not_claimed':['2pp noninferiority','learned benefit','30% cost reduction','unseen-to-policy data']})
        life.save(master/'STATE.json',{'status':'READY','started_at':started,'roots_reserved':0,'batches':[],
          'streams':{str(s):{'offset':0,'accepted_source':str(master/'initial_experience')} for s in range(3)}})
    state=load(master/'STATE.json');protocol=load(master/'PROTOCOL.json')
    assert state['status']=='READY','Existing running/failed campaign requires inspection; no duplicate or reset'
    assert time.time()<protocol['campaign_absolute_deadline'],'Global wall budget exhausted'
    assert digest(master/'PROPOSED_ROSTER.json')==protocol['roster_sha256']
    assert tree(master/'code')==load(master/'CODE_MANIFEST.json')
    code_source=master/'code';code_manifest=master/'CODE_MANIFEST.json'
    revision=state.get('active_protocol_revision')
    if revision:
        assert revision in ('report_path_v2','terminal_reads_v3'), 'Unknown protocol revision'
        rev=master/'protocol_revisions'/revision
        binding=load(rev/'BINDING.json')
        assert binding['original_protocol_sha256']==digest(master/'PROTOCOL.json')
        code_source=rev/'code';code_manifest=rev/'CODE_MANIFEST.json'
        assert tree(code_source)==load(code_manifest), 'Revision source changed'
        assert binding['code_manifest_sha256']==digest(code_manifest)
    stream_state=state['streams'][str(args.stream)];offset=stream_state['offset']
    cases=load(master/'PROPOSED_ROSTER.json')['streams'][args.stream]['cases'][offset:offset+args.roots]
    assert cases and state['roots_reserved']+len(cases)<=60
    accepted=Path(stream_state['accepted_source']);validate_snapshot(accepted);parent_hash=tree(accepted)
    batch_id=len(state['batches']);run=master/f'batch_{batch_id:02d}_s{args.stream}';run.mkdir(exist_ok=False)
    start=time.time();life.CAMPAIGN=run;life.BASE=str(run).replace(str(life.PROJECT),'/project');life.deadline=min(start+2700,protocol['campaign_absolute_deadline'])
    shutil.copytree(code_source,run/'code');shutil.copyfile(code_manifest,run/'CODE_MANIFEST.json')
    if revision:shutil.copyfile(rev/'BINDING.json',run/'PROTOCOL_REVISION.json')
    for rel in ['MODEL_CONFIG.json','MODEL_BINDING.json','CORE_SOURCE_BINDING.json','PROTOCOL.json']:shutil.copyfile(master/rel,run/rel)
    life.save(run/'ROSTER.json',cases);life.save(run/'EXPERIENCE_BINDING.json',{'accepted_source':str(accepted),'source_files':parent_hash,'stream':args.stream,'offset':offset})
    status={'status':'RUNNING','started_at':start,'stream':args.stream,'offset':offset,'root_reservations':len(cases),'protocol_revision':revision or 'initial'}
    state.update(status='RUNNING',roots_reserved=state['roots_reserved']+len(cases));state['batches'].append({'id':batch_id,'path':str(run),'status':'RUNNING','started_at':start,'roots_reserved':len(cases)});life.save(master/'STATE.json',state);life.save(run/'STATUS.json',status)
    def interrupt(n,f):raise RuntimeError('Interrupted '+str(n))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    stream=run/'stream';stream.mkdir();cpu=run/'cpu';cpu.mkdir()
    try:
        life.gate(cpu)
        tests=['test_native_workspace.py','test_native_isolation.py','test_terminal_results.py']
        argv=life.cpu_args('yuchen-fastrsi-'+args.campaign+f'-b{batch_id}-cpu',cpu)+['-e','FASTRSI_NATIVE_RUN=/tmp/fastrsi-screen-cpu/stream',life.IMAGE,'timeout','180','bash','-lc','source /opt/dtk/env.sh && '+' && '.join('/project/envs/os-v1/bin/python '+life.BASE+'/code/tests/'+t for t in tests)]
        cid=life.launch('cpu',argv,cpu);life.wait_exit(cid,190);life.save(cpu/'STOP_RECEIPT.json',life.stop_all());status['cpu_status']='PASSED';life.save(run/'STATUS.json',status)
        life.gate(stream);shutil.copytree(life.PROJECT/'models/openpi_data',stream/'tokenizer_cache');cfg=load(run/'MODEL_CONFIG.json');containers={}
        for role,receipt in [('pi05','LAUNCH_pi05_v2.json'),('qwen','LAUNCH_qwen.json'),('target','LAUNCH_target.json')]:
            argv=load(life.OLD/receipt)['argv'];argv[argv.index('--name')+1]='yuchen-fastrsi-'+args.campaign+f'-b{batch_id}-'+role
            argv=[x.replace(str(life.OLD),str(stream)).replace('/project/runs/pilot_20261001',life.BASE+'/stream') for x in argv]
            if role=='pi05':
                argv=[x.replace(str(life.PROJECT/'models/openpi_data'),str(stream/'tokenizer_cache')) for x in argv];assert '--torch-seed 7' in argv[-1]
            if role=='qwen':argv=configure_qwen(argv,cfg,life)
            if role=='target':argv=[x.replace('/project/scripts/instrument_target.py',life.BASE+'/code/scripts/instrument_target.py').replace(':/project/scripts',':'+life.BASE+'/code/scripts') for x in argv]
            containers[role]=life.launch(role,argv,stream)
        life.health(49500);life.health(49510)
        until=min(life.deadline,time.time()+120)
        while True:
            try:
                with socket.create_connection(('127.0.0.1',49502),timeout=1):break
            except OSError:
                if time.time()>until:raise
                time.sleep(1)
        sdk="import json; from libero.libero import benchmark; s=benchmark.get_benchmark_dict()['libero_10'](); print('TASKS='+json.dumps([{'language':s.get_task(i).language,'init_count':len(s.get_task_init_states(i))} for i in range(10)]))"
        receipt=life.command(['docker','exec',containers['target'],'python','-c',sdk],timeout=60);life.save(run/'SDK_TASK_BINDING.json',receipt);assert receipt['returncode']==0
        actual=json.loads(next(x[6:] for x in receipt['stdout'].splitlines() if x.startswith('TASKS=')))
        for c in cases:assert c['task_description']==actual[c['task_id']]['language'] and c['init_state_id']<actual[c['task_id']]['init_count']
        status['startup_seconds']=time.time()-start;life.save(run/'STATUS.json',status)
        remaining=int(life.deadline-time.time()-30);assert remaining>60
        source=str(accepted).replace(str(life.PROJECT),'/project')
        argv=life.cpu_args('yuchen-fastrsi-'+args.campaign+f'-b{batch_id}-runner',stream)+model_env(cfg)+['-e','FASTRSI_NATIVE_RUN='+life.BASE+'/stream','-e','FASTRSI_CONSUME_TERMINAL=1','-e','FASTRSI_EXPERIENCE_ADMISSION=guarded_v1','-e','FASTRSI_INITIAL_EXPERIENCE='+source,
             life.IMAGE,'timeout',str(remaining),'bash','-lc','source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/run_native_baseline.py']
        cid=life.launch('runner',argv,stream);life.wait_exit(cid,remaining+5)
        result=load(stream/'RESULT.json');assert result['status']=='COMPLETED' and len(result['cases'])==len(cases)
        executions=sum(len(row['outcomes']) for row in result['cases']);assert executions<=3*len(cases)
        for row in result['cases']:assert row['chain_status']=='PASSED' and row['terminal_report']['status']=='PASSED'
        assert all((p.parent/'RECEIPT.json').exists() for p in stream.glob('service_calls/*/REQUEST.json'))
        next_source=result['cases'][-1]['experience_admission']['next_source'];assert next_source
        next_source=Path(next_source.replace('/project',str(life.PROJECT),1));validate_snapshot(next_source)
        assert tree(accepted)==parent_hash,'Initial accepted source changed'
        stream_state.update(offset=offset+len(cases),accepted_source=str(next_source))
        status.update(status='COMPLETED',controller_seconds=result['seconds'],environment_executions=executions,case_results=[{'id':r['case']['id'],'environment_success':r['environment_success'],'verdicts':[o.get('verdict') for o in r['outcomes']],'admission':r['experience_admission']['status']} for r in result['cases']])
    except BaseException:status.update(status='FAILED',error=traceback.format_exc())
    finally:
        life.save(stream/'STOP_RECEIPT.json',life.stop_all());status.update(finished_at=time.time(),full_wall_seconds=time.time()-start);life.save(run/'STATUS.json',status)
        state['batches'][-1].update(status=status['status'],finished_at=status['finished_at'],full_wall_seconds=status['full_wall_seconds'])
        state['status']='READY' if status['status']=='COMPLETED' else 'FAILED'
        state['elapsed_master_wall_seconds']=time.time()-state['started_at'];life.save(master/'STATE.json',state)
        print(json.dumps(status),flush=True)
    if status['status']!='COMPLETED':raise SystemExit(1)

if __name__=='__main__':main()

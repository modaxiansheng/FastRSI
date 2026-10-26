"""Bounded shared Qwen service; five saved-report checks, no VLA or target service."""
import fcntl,hashlib,json,shutil,signal,time,traceback
from pathlib import Path
import orchestrate_full_os as life
from native_service_config import configure_qwen,model_env
NAME='report_evidence_v2_20261003'
def main():
 p=life.PROJECT;run=p/'runs'/NAME;lock=open(p/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 run.mkdir(exist_ok=False);life.CAMPAIGN=run;life.BASE='/project/runs/'+NAME;start=time.time();life.deadline=start+850
 for rel in ('scripts','src/fastrsi','tests','configs'):shutil.copytree(p/rel,run/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for rel in ('scripts/qwen_agent_server.py','scripts/run_native_baseline.py'):
  f=run/'code'/rel;f.write_text(f.read_text().replace('49510','19510'))
 life.save(run/'CODE_MANIFEST.json',{str(f.relative_to(run/'code')):hashlib.sha256(f.read_bytes()).hexdigest() for f in (run/'code').rglob('*') if f.is_file()})
 src=p/'runs/task_window_v1_20261003'
 for rel in ('MODEL_CONFIG.json','MODEL_BINDING.json','CORE_SOURCE_BINDING.json'):shutil.copyfile(src/rel,run/rel)
 jobs=[]
 for name,source,view in [('N_raw','task_window_v1_20261003/pair0/N/stream/net_t4_i5',False),('N_view','task_window_v1_20261003/pair0/N/stream/net_t4_i5',True),('F_view','task_window_v1_20261003/pair0/F/stream/net_t4_i5',True),('F_raw','task_window_v1_20261003/pair0/F/stream/net_t4_i5',False),('failure_view','feedback_window_known_v1_20261003/pair0/N/stream/net_t4_i31',True)]:jobs.append({'name':name,'source':'/project/runs/'+source,'view':view})
 life.save(run/'PROTOCOL.json',{'scope':'Shared deterministic reporting repair, report-only replay; no FastRSI learning efficiency claim','jobs':jobs,'deadline':life.deadline,'maximum_wall_seconds':850,'resources':'One shared Qwen GPU, sequential raw/view and reversed view/raw order; startup billed separately','robot_executions':0,'verifier_calls':0,'learning':'Fixed experience with native task-local prefix; original report turn excluded; no admission','promotion':'Opt-in for common future protocols only after actual report checks; canonical files and all chunk facts retained'})
 state={'status':'RUNNING','started_at':start};life.save(run/'STATUS.json',state)
 def interrupt(n,f):raise RuntimeError('Interrupted '+str(n))
 signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
 cpu=run/'cpu';cpu.mkdir();stream=run/'stream';stream.mkdir()
 try:
  life.gate(cpu)
  argv=life.cpu_args('yuchen-fastrsi-'+NAME+'-cpu',cpu)+['-e','FASTRSI_NATIVE_RUN=/tmp/report-view/stream',life.IMAGE,'timeout','150','bash','-lc','source /opt/dtk/env.sh && cd '+life.BASE+'/code/tests && /project/envs/os-v1/bin/python -m unittest test_report_evidence test_terminal_evidence_view test_terminal_results test_terminal_prefetch test_native_isolation test_native_memory_budget -v']
  cid=life.launch('cpu',argv,cpu);life.wait_exit(cid,160);life.save(cpu/'STOP_RECEIPT.json',life.stop_all());state['cpu']='PASSED';life.save(run/'STATUS.json',state)
  life.gate(stream);cfg=json.loads((run/'MODEL_CONFIG.json').read_text());argv=json.loads((life.OLD/'LAUNCH_qwen.json').read_text())['argv'];argv[argv.index('--name')+1]='yuchen-fastrsi-'+NAME+'-qwen'
  argv=[x.replace(str(life.OLD),str(stream)).replace('/project/runs/pilot_20261001',life.BASE+'/stream').replace('49510','19510') for x in argv]
  life.launch('qwen',configure_qwen(argv,cfg,life),stream);life.health(19510)
  remaining=int(life.deadline-time.time()-30);assert remaining>60
  argv=life.cpu_args('yuchen-fastrsi-'+NAME+'-runner',stream)+model_env(cfg)+['-e','FASTRSI_NATIVE_RUN='+life.BASE+'/stream',life.IMAGE,'timeout',str(remaining),'bash','-lc','source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/replay_evidence_view.py']
  cid=life.launch('runner',argv,stream);life.wait_exit(cid,remaining+5)
  result=json.loads((stream/'RESULT.json').read_text());assert result['status']=='COMPLETED';state['status']='COMPLETED'
 except BaseException:state.update(status='FAILED',error=traceback.format_exc())
 finally:
  life.save(stream/'STOP_RECEIPT.json',life.stop_all());state.update(full_wall_seconds=time.time()-start,finished_at=time.time());life.save(run/'STATUS.json',state);print(json.dumps(state),flush=True)
 if state['status']!='COMPLETED':raise SystemExit(1)
if __name__=='__main__':main()

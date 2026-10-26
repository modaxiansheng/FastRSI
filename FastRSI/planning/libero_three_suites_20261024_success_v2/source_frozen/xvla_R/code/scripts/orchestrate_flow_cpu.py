"""Bounded regression for the shared Session flow; never starts models."""
import argparse,fcntl,hashlib,json,shutil,sys,time,traceback
from pathlib import Path
import orchestrate_full_os as life

def main():
 ap=argparse.ArgumentParser();ap.add_argument('name');ap.add_argument('--pagination',action='store_true');ap.add_argument('--terminal',action='store_true');ap.add_argument('--master');args=ap.parse_args()
 name=args.name;assert name.replace('_','').isalnum()
 tests=['test_session_flow.py','test_native_workspace.py']
 if args.pagination:tests=['test_paged_reads.py']+tests+['test_native_isolation.py']
 if args.terminal:tests+=['test_terminal_results.py']
 lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 run=life.PROJECT/'runs'/name;run.mkdir(exist_ok=False);life.CAMPAIGN=run;life.BASE='/project/runs/'+name
 started=time.time();life.deadline=started+240
 if args.master:
  assert args.master.startswith('representative_native_') and args.master.replace('_','').isalnum()
  master=life.PROJECT/'runs'/args.master;protocol=json.loads((master/'PROTOCOL.json').read_text())
  assert json.loads((master/'STATE.json').read_text())['status']!='RUNNING'
  life.deadline=min(life.deadline,protocol['campaign_absolute_deadline']);assert life.deadline-started>200
 status={'status':'RUNNING','started_at':started,'scope':'CPU bounded pagination and native integration regression' if args.pagination else 'CPU partial-write resume regression only','master_budget':args.master}
 for rel in ['scripts','src/fastrsi','tests','configs']:shutil.copytree(life.PROJECT/rel,run/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 life.save(run/'CODE_MANIFEST.json',{str(p.relative_to(run/'code')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'code').rglob('*') if p.is_file()})
 life.save(run/'PROTOCOL.json',{'max_seconds':240,'model_services':0,'robot_episodes':0,'tests':tests,'costs':'All launch, tests and shutdown counted; no runtime-quality or learning claim'})
 folder=run/'cpu';folder.mkdir();life.save(run/'STATUS.json',status)
 try:
  life.gate(folder)
  argv=life.cpu_args('yuchen-fastrsi-'+name+'-cpu',folder)+['-e','FASTRSI_NATIVE_RUN=/tmp/fastrsi-flow-cpu2/stream','-e','FASTRSI_EXPERIENCE_ADMISSION=guarded_v1',life.IMAGE,'timeout','180','bash','-lc','source /opt/dtk/env.sh && '+ ' && '.join('/project/envs/os-v1/bin/python '+life.BASE+'/code/tests/'+n for n in tests)]
  cid=life.launch('cpu',argv,folder);life.wait_exit(cid,190);status['status']='COMPLETED'
 except BaseException:status.update(status='FAILED',error=traceback.format_exc())
 finally:
  life.save(folder/'STOP_RECEIPT.json',life.stop_all());status.update(finished_at=time.time(),full_wall_seconds=time.time()-started);life.save(run/'STATUS.json',status);print(json.dumps(status),flush=True)
 if status['status']!='COMPLETED':raise SystemExit(1)
if __name__=='__main__':main()

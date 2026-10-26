"""Resident single-card feature extraction while independent robot jobs execute."""
import fcntl,json,time,traceback
import orchestrate_full_os as life
NAME='phase_learning_v1_20261005';REV='17f0b0ad64efb65d273590632833508766b2aae6'
def main():
 run=life.PROJECT/'runs'/NAME;lock=open(life.PROJECT/'runs/clef_validity.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (run/'STATUS.json').exists();start=time.time();life.deadline=start+1500;state=dict(status='RUNNING',started_at=start)
 try:
  life.gate(run);life.save(run/'STATUS.json',state)
  argv=['docker','run','-d','--name','yuchen-fastrsi-'+NAME,'--network','none','--cpus','8','--memory','64g','--shm-size','2g',
   '--device','/dev/kfd','--device','/dev/mkfd','--device','/dev/dri','--group-add','video','--group-add','render',
   '-e','HIP_VISIBLE_DEVICES=0','-e','CUDA_VISIBLE_DEVICES=0','-e','PYTHONDONTWRITEBYTECODE=1','-e','HF_HUB_OFFLINE=1','-e','OMP_NUM_THREADS=4','-e','MKL_NUM_THREADS=4',
   '-v','/opt/hyhal:/opt/hyhal:ro','-v','/etc/hfm:/etc/hfm:ro','-v',str(life.PROJECT)+':/project:ro','-v',str(run)+':/output:rw',
   life.IMAGE,'timeout','1400','bash','-lc','source /opt/dtk/env.sh && source /opt/hyhal/env.sh && exec /project/envs/clef-decision-v1/bin/python /output/code/run_phase_learning.py --model /project/models/clef-flash/'+REV+' --run /output']
  cid=life.launch('learner',argv,run);life.wait_exit(cid,1410);result=json.loads((run/'learning/RESULT.json').read_text());assert result['status']=='COMPLETED';state['status']='COMPLETED'
 except BaseException:state.update(status='FAILED',error=traceback.format_exc())
 finally:
  life.save(run/'STOP_RECEIPT.json',life.stop_all());state.update(seconds=time.time()-start,finished_at=time.time());life.save(run/'STATUS.json',state);print(json.dumps(state),flush=True)
 if state['status']!='COMPLETED':raise SystemExit(1)
if __name__=='__main__':main()

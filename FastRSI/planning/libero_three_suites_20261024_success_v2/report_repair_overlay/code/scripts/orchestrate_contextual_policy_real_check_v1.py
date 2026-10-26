"""Bounded one-card real local-route regression; no robot calls."""
import fcntl
import hashlib
import json
import shutil
import signal
import time
import traceback
import orchestrate_full_os as life

NAME='contextual_policy_real_check_v1_20261006'


def main():
    run=life.PROJECT/'runs'/NAME
    lock=(life.PROJECT/'runs/full_os_v1.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    output=run/'job';output.mkdir(exist_ok=False)
    start=time.time();life.deadline=start+480
    state=dict(status='PREPARING',started_at=start,cards=1,robot_episodes=0)
    def stop(signum,frame):raise RuntimeError('Owned extraction interrupted '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for rel,digest in json.loads((run/'FILE_MANIFEST.json').read_text()).items():
            assert hashlib.sha256((run/rel).read_bytes()).hexdigest()==digest,rel
        reg=json.loads((run/'inputs/REGISTRATION.json').read_text())
        for rel,digest in reg['historical_sources'].items():
            assert hashlib.sha256((life.PROJECT/rel).read_bytes()).hexdigest()==digest,rel
        life.gate(run)
        source=life.PROJECT/'runs/native_motion_effect_skill_value_v1_20261006/N/stream'
        old=json.loads((source/'LAUNCH_pi05.json').read_text())['argv']
        ix=next(i for i,x in enumerate(old) if x.startswith('sha256:'))
        argv=old[:ix];argv[argv.index('--name')+1]='yuchen-fastrsi-contextual-real-check-v1'
        argv[argv.index('--network')+1]='none'
        for i,v in enumerate(argv):
            if v.startswith('PYTHONPATH='):argv[i]=v+':/audit/code/src'
            elif v==str(source)+':/pilot:rw':argv[i]=str(output)+':/output:rw'
            elif v==str(source/'tokenizer_cache')+':/workspace/openpi_data:rw':
                argv[i]=str(run/'tokenizer_cache')+':/workspace/openpi_data:rw'
        shutil.copytree(source/'tokenizer_cache',run/'tokenizer_cache')
        argv+=['-v',str(run/'code')+':/audit/code:ro','-v',str(run/'inputs')+':/audit/inputs:ro',
               '-v',str(life.PROJECT/'runs/native_pi_component_profile_v1_20261006')+':/reference:ro',
               old[ix],'timeout','420','bash','-lc',
               'source /opt/dtk/env.sh && source /opt/hyhal/env.sh && exec python /audit/code/scripts/check_contextual_policy_real_v1.py']
        cid=life.launch('features',argv,run);state.update(status='RUNNING',container=cid);life.save(run/'STATUS.json',state)
        life.wait_exit(cid,430)
        result=json.loads((output/'RESULT.json').read_text())
        assert result['status']=='COMPLETED' and result['actual_full_decoder_calls']==6
        assert result['extra_prefix_calls']==6 and result['unmarked_output_rng_equal']
        state.update(status='COMPLETED',result='job/RESULT.json')
    except BaseException:
        state.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        life.save(run/'STOP_RECEIPT.json',life.stop_all());state.update(finished_at=time.time(),seconds=time.time()-start)
        life.save(run/'STATUS.json',state);print(json.dumps(state),flush=True)


if __name__=='__main__':main()

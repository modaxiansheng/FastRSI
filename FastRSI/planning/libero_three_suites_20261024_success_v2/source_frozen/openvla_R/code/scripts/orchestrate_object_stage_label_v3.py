"""One CPU-only label-alignment refit using already paid visual features."""
import fcntl
import hashlib
import json
import time
import traceback

import orchestrate_full_os as life

NAME = 'object_stage_label_alignment_v3_20261006'


def main():
    run = life.PROJECT / 'runs' / NAME
    for relative, expected in json.loads((run / 'CODE_MANIFEST.json').read_text()).items():
        if hashlib.sha256((run / 'code' / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Refit code binding changed')
    lock = open(life.PROJECT / 'runs/full_os_v1.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    life.deadline = time.time() + 170
    out = run / 'job'; out.mkdir(exist_ok=False)
    state = dict(status='RUNNING', started_at=time.time(), gpu_cards_reserved=0,
                 new_visual_calls=0, new_robot_episodes=0)
    life.save(out / 'STATUS.json', state)
    try:
        life.gate(out)
        argv = ['docker','run','-d','--name','yuchen-fastrsi-object-stage-current-v3',
            '--network','none','--cpus','4','--memory','8g',
            '-e','HIP_VISIBLE_DEVICES=-1','-e','CUDA_VISIBLE_DEVICES=-1',
            '-e','OMP_NUM_THREADS=4','-e','MKL_NUM_THREADS=4',
            '-e','PYTHONDONTWRITEBYTECODE=1',
            '-v','/opt/hyhal:/opt/hyhal:ro','-v','/etc/hfm:/etc/hfm:ro',
            '-v',str(life.PROJECT)+':/project:ro','-v',str(out)+':/output:rw',
            life.IMAGE,'timeout','120','bash','-lc',
            'source /opt/dtk/env.sh && source /opt/hyhal/env.sh && exec '
            '/project/envs/clef-decision-v1/bin/python '
            '/project/runs/'+NAME+'/code/scripts/refit_object_stage_current_v3.py '
            '--project /project --prior /project/runs/object_stage_memory_v2_20261006 '
            '--output /output/learning']
        cid=life.launch('refit',argv,out); life.wait_exit(cid,130)
        result=json.loads((out/'learning/RESULT.json').read_text())
        if result['status']!='COMPLETED':raise ValueError('Corrected supervision fit failed')
        state['status']='COMPLETED'
    except BaseException:
        state.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        life.save(out/'STOP_RECEIPT.json',life.stop_all())
        state.update(seconds=time.time()-state['started_at'],finished_at=time.time())
        life.save(out/'STATUS.json',state)


if __name__=='__main__':main()

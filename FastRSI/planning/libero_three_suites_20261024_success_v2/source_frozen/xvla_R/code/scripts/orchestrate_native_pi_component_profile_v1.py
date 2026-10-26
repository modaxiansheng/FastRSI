"""One-card passive profile, exact existing dependencies, retained inputs only."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time
import traceback

import orchestrate_full_os as life

NAME = 'native_pi_component_profile_v1_20261006'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    run = life.PROJECT/'runs'/NAME
    lock = (life.PROJECT/'runs/full_os_v1.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = run/'job'
    output.mkdir(exist_ok=False)
    start = time.time()
    life.deadline = start+480
    state = dict(status='PREPARING', started_at=start, cards=1, robot_episodes=0)
    def interrupted(signum, frame):
        raise RuntimeError('Owned component profile interrupted '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for rel, digest in json.loads((run/'FILE_MANIFEST.json').read_text()).items():
            assert sha(run/rel) == digest, rel
        reg = json.loads((run/'inputs/REGISTRATION.json').read_text())
        for rel, digest in reg['historical_sources'].items():
            assert sha(life.PROJECT/rel) == digest, rel
        life.gate(run)
        old_stream = life.PROJECT/'runs/native_motion_effect_skill_value_v1_20261006/N/stream'
        old = json.loads((old_stream/'LAUNCH_pi05.json').read_text())['argv']
        image_index = next(i for i,v in enumerate(old) if v.startswith('sha256:'))
        argv = old[:image_index]
        argv[argv.index('--name')+1] = 'yuchen-fastrsi-pi-component-profile-v1'
        argv[argv.index('--network')+1] = 'none'
        for i,v in enumerate(argv):
            if v.startswith('PYTHONPATH='):
                argv[i] = v+':/audit/code/src'
            elif v == str(old_stream)+':/pilot:rw':
                argv[i] = str(output)+':/output:rw'
            elif v == str(old_stream/'tokenizer_cache')+':/workspace/openpi_data:rw':
                argv[i] = str(run/'tokenizer_cache')+':/workspace/openpi_data:rw'
        shutil.copytree(old_stream/'tokenizer_cache', run/'tokenizer_cache')
        argv += ['-v',str(run/'code')+':/audit/code:ro','-v',str(run/'inputs')+':/audit/inputs:ro',
                 old[image_index], 'timeout','420','bash','-lc',
                 'source /opt/dtk/env.sh && source /opt/hyhal/env.sh && exec python /audit/code/scripts/profile_native_pi_components_v1.py']
        cid = life.launch('profile', argv, run)
        state.update(status='RUNNING', container=cid)
        life.save(run/'STATUS.json', state)
        life.wait_exit(cid, 430)
        result = json.loads((output/'RESULT.json').read_text())
        assert result['status']=='COMPLETED' and result['new_policy_inferences']==18
        assert result['original_outputs_preserved'] and result['original_rng_preserved']
        state.update(status='COMPLETED', result='job/RESULT.json')
    except BaseException:
        state.update(status='FAILED', error=traceback.format_exc())
        raise
    finally:
        life.save(run/'STOP_RECEIPT.json', life.stop_all())
        state.update(finished_at=time.time(), seconds=time.time()-start)
        life.save(run/'STATUS.json', state)
        print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()

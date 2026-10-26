"""CPU LIBERO target, frozen code plus timing observer, bounded to one hour."""
import json
import subprocess
from pathlib import Path

root=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
assets=root.parent/'frozen_archive/extra-freeze-20260929T145255Z/frozen/assets-20260929-v1'
run=root/'runs/pilot_20261001'
config=root/'configs/libero'
config.mkdir(parents=True,exist_ok=True)
base='/workspace/openpi/third_party/libero/libero'
paths={'benchmark_root':base, 'bddl_files':base+'/bddl_files',
       'init_states':base+'/init_files','datasets':'/tmp/libero-datasets','assets':base+'/assets'}
# The actual frozen checkout has one nested libero package; check rather than guess.
repo=assets/'dependencies/openpi/third_party/libero'
if (repo/'libero/libero/bddl_files').is_dir():
    base='/workspace/openpi/third_party/libero/libero/libero'
elif not (repo/'libero/bddl_files').is_dir():
    raise RuntimeError('Cannot find frozen LIBERO assets')
paths.update(benchmark_root=base,bddl_files=base+'/bddl_files',init_states=base+'/init_files',assets=base+'/assets')
(config/'config.yaml').write_text(''.join(f'{key}: {value}\n' for key,value in paths.items()))
argv=['docker','run','-d','--name','yuchen-fastrsi-target-20261001-v1','--network','host',
      '--cpus','8','--memory','24g','--pids-limit','1024','--shm-size','2g',
      '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONPATH=/workspace/openpi/third_party/libero:/project/scripts',
      '-e','LIBERO_CONFIG_PATH=/libero-config','-e','MUJOCO_GL=egl','-e','LIBGL_ALWAYS_SOFTWARE=1',
      '-e','FASTRSI_TRACE=/project/runs/pilot_20261001/TARGET_COST_EVENTS.jsonl',
      '-v',str(root)+':/project:rw','-v',str(assets/'dependencies/openpi')+':/workspace/openpi:ro',
      '-v',str(assets/'dependencies/PhyAgentOS-core')+':/core:ro','-v',str(config)+':/libero-config:ro',
      'sha256:3171fda2362d773433482f79c4fd53f6f428413efbf41c3fdfb43d8c799d6c95',
      'timeout','3600','python','/project/scripts/instrument_target.py',
      '--host','127.0.0.1','--port','49502','--seed','7','--control-mode','relative']
result=subprocess.run(argv,capture_output=True,text=True,timeout=60)
(run/'LAUNCH_target.json').write_text(json.dumps({'argv':argv,'returncode':result.returncode,
                                              'stdout':result.stdout,'stderr':result.stderr},indent=2))
print(result.returncode,result.stdout,result.stderr)
if result.returncode:raise SystemExit(result.returncode)

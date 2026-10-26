"""Resource-only adapter: one extra policy and its verifier share an owned GPU."""
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('retained_lane', ROOT / 'run_lane.py')
lane = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lane)
original_adapt = lane.adapt
gpu = int(sys.argv[sys.argv.index('--gpu-base') + 1])
slot = int(sys.argv[sys.argv.index('--slot-index') + 1])
assert (slot, gpu) in [(6, 2), (7, 6)]


def adapt(module, *args, **kwargs):
    original_adapt(module, *args, **kwargs)
    original_launch = module.launch

    def ports_for(gpu_base, slot_index=None):
        assert gpu_base == gpu and slot_index == slot
        base = 21100 + 100 * slot
        return dict(zip(('policy', 'target', 'qwen', 'verifier'), (base, base + 2, base + 10, base + 20)))

    module.ports_for = ports_for

    def launch(argv, owner, role, prefix):
        if role == 'qwen':
            argv = [('HIP_VISIBLE_DEVICES=' + str(gpu)) if v.startswith('HIP_VISIBLE_DEVICES=') else
                    ('CUDA_VISIBLE_DEVICES=' + str(gpu)) if v.startswith('CUDA_VISIBLE_DEVICES=') else v for v in argv]
        if role == 'runner':
            p = Path(owner) / 'RESOURCE03_WHOLE_DEVICE.json'
            assert not p.exists()
            p.write_text(json.dumps(dict(time=time.time(), logical_slot=slot, policy_gpu=gpu,
                       qwen_gpu=gpu, physical_node='107', adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       change='GPU placement and private ports only; retained policy, model generation, task, budget, recovery and success scoring unchanged'), indent=2))
        return original_launch(argv, owner, role, prefix)

    module.launch = launch


lane.adapt = adapt
if __name__ == '__main__':
    lane.main()

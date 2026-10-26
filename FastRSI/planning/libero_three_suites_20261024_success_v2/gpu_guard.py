"""Visible two-card allocation across service reloads; no model or inference."""
import json
import signal
import time
from pathlib import Path

import torch

running = True


def stop(*_):
    global running
    running = False


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
assert torch.cuda.device_count() == 2
buffers = [torch.empty(96 * 1024 * 1024, dtype=torch.float32, device='cuda:' + str(i)) for i in range(2)]
Path('/control/GPU_GUARD_READY.json').write_text(json.dumps(dict(
    ready=True, time=time.time(), logical_devices=2, physical_devices=[2, 3],
    retained_bytes_per_device=96 * 1024 * 1024 * 4,
    model_loads=0, neural_inferences=0,
    purpose='Keep two authorized cards visible to the existing shared allocator during service transitions')))
while running:
    time.sleep(2)

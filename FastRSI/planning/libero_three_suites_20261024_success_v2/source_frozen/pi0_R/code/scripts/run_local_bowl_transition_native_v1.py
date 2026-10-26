"""One local execution candidate inside the original complete Agent loop."""
import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np
import run_native_baseline as native
import run_native_sampler_goal_component_v1 as sampler
import fastrsi.session_flow as flow
from cost_trace import emit
from fastrsi.local_bowl_transition_v1 import BowlTransition, GOAL

original_case = native.run_case


async def run_case(case, accepted):
    config = case["local_bowl_transition"]
    raw = Path(config["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != config["sha256"]:
        raise ValueError("Local recipe binding changed")
    spec = json.loads(raw)
    if (case["mode"],case["suite"],case["task_id"],case["task_description"]) != ("development","libero_90",64,GOAL):
        raise ValueError("Only one registered developmental task family")
    source = Path(spec["source_rgb"]["path"])
    calibration = Path(spec["calibration"]["path"])
    for path, reference in ((source,spec["source_rgb"]),(calibration,spec["calibration"])):
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
            raise ValueError("Public source appearance/calibration changed")
    import cv2
    source_rgb = cv2.cvtColor(cv2.imread(str(source)),cv2.COLOR_BGR2RGB)
    hook = BowlTransition(session_id=case["id"], enabled=spec["enabled"],source_rgb=source_rgb,
        source_rois=spec["source_rois"],calibration=json.loads(calibration.read_text()),
        recipe=spec["recipe"],emit=emit,eligibility_gate=spec.get("eligibility_gate"))
    restore = hook.install()
    complete = False
    try:
        result = await original_case(case,accepted)
        complete = True
        return result
    finally:
        restore()
        path = native.ROOT/case["id"]/"LOCAL_TRANSITION_RECEIPT.json"
        with path.open("x") as stream:
            json.dump(dict(enabled=hook.enabled, native_run_completed=complete,
                trigger_step=hook.trigger_step,phase=hook.phase,retired=hook.retired,
                local_actions=hook.local_actions,local_requests=hook.local_requests,
                root_actual_vla_calls=hook.actual_vla_calls,planner_seconds=hook.seconds,
                eligibility_gate=spec.get("eligibility_gate"),gate_seconds=0. if hook.gate is None else hook.gate.seconds,
                events=hook.events,learned=False,physical_release_confirmed=False),stream,indent=2)


async def main():
    restore = sampler.install_runner(native,flow)
    native.run_case = run_case
    try:
        await native.main()
    finally:
        native.run_case = original_case
        restore()


if __name__ == "__main__":
    asyncio.run(main())

"""Native full-Agent development entry for the CONTINUE-only interface probe.

No extra Agent/model call, success label or memory trigger is added. Native
development feedback still runs normally; the hook itself writes no experience.
This wrapper is not a learned policy or an independent evaluation experiment.
"""
import asyncio
import hashlib
import json
from pathlib import Path
import queue

import run_native_baseline as native
from cost_trace import emit
from fastrsi.native_stage_continuation_v1 import NativeStageContinuation


original_case = native.run_case


async def consume_continue(gate):
    while True:
        try:
            request = gate.requests.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.005)
            continue
        gate.resolve(request.request_id, action="continue",
                     component_versions=gate.component_versions)


async def run_case(case, accepted):
    config = case.get("native_stage_continuation_probe")
    if config is None:
        return await original_case(case, accepted)
    if (case.get("mode") != "development" or
            config.get("authority") != "CONTINUE_ONLY_INTERFACE_PROBE" or
            type(config.get("enabled")) is not bool or
            not isinstance(config.get("pause_steps"), list)):
        raise ValueError("Only an explicit developmental continuity check is accepted")
    model_raw = (native.CAMPAIGN/"MODEL_CONFIG.json").read_bytes()
    if hashlib.sha256(model_raw).hexdigest() != config.get("model_config_sha256"):
        raise ValueError("The registered model configuration changed")
    hook_path = Path(__file__).resolve().parents[1]/"src/fastrsi/native_stage_continuation_v1.py"
    versions = dict(model_config=config["model_config_sha256"],
                    stage_hook=hashlib.sha256(hook_path.read_bytes()).hexdigest())
    gate = NativeStageContinuation(session_id=case["id"], goal=case["task_description"],
                                  pause_steps=tuple(config["pause_steps"]),
                                  component_versions=versions, emit=emit)
    restore = gate.install() if config["enabled"] else None
    consumer = asyncio.create_task(consume_continue(gate)) if config["enabled"] else None
    receipt_path = native.ROOT/"NATIVE_STAGE_CONTINUATION_RECEIPT.json"
    if receipt_path.exists():
        if consumer is not None:
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
        if restore is not None:
            restore()
        raise FileExistsError("Preserve the existing continuity receipt")
    completed = False
    try:
        result = await original_case(case, accepted)
        completed = True
        return result
    finally:
        if consumer is not None:
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
        if restore is not None:
            restore()
        receipt_path.write_text(json.dumps(dict(
            authority="CONTINUE_ONLY_INTERFACE_PROBE", enabled=config["enabled"],
            case_id=case["id"], original_goal=case["task_description"],
            component_versions=versions, registered_steps=config["pause_steps"],
            native_run_case_completed=completed, events=gate.events,
            actual_submitted_chunks=gate.submitted_chunks if config["enabled"] else None,
            added_model_calls=0, added_memory_triggers=0,
            skill_selection_bound=False, learning_benefit_measured=False), indent=2)+"\n")


native.run_case = run_case
if __name__ == "__main__":
    asyncio.run(native.main())

"""Actual original Agent case with one registered operation-value intervention."""
import asyncio
import hashlib
import json
from pathlib import Path
import queue

import run_native_baseline as native
from cost_trace import emit
from fastrsi.native_relative_phaselet_v1 import AUTHORITY, NativeRelativePhaseletChoice, load_phaselet


original_case = native.run_case


async def consume(gate, action):
    while True:
        try:
            request = gate.requests.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.005)
            continue
        gate.resolve(request.request_id, action=action, component_versions=gate.component_versions)


async def run_case(case, accepted):
    cfg = case.get("native_relative_phaselet_probe")
    if cfg is None:
        return await original_case(case, accepted)
    if (case.get("mode") != "development" or cfg.get("authority") != AUTHORITY
            or case.get("feedback_window_mode") != "fixed10"
            or case.get("inference_horizon",10) != 10
            or cfg.get("operation") not in ("continue", "relative_phaselet")
            or len(cfg.get("pause_steps", [])) != 1):
        raise ValueError("Only a registered one-boundary developmental value probe is accepted")
    model_sha = hashlib.sha256((native.CAMPAIGN/"MODEL_CONFIG.json").read_bytes()).hexdigest()
    if model_sha != cfg.get("model_config_sha256"):
        raise ValueError("Original model configuration changed")
    phaselet = load_phaselet(cfg["artifact"], cfg["sha256"],
                            goal=case["task_description"], model_config_sha=model_sha)
    source = Path(__file__).resolve().parents[1]/"src/fastrsi"
    versions = dict(model_config=model_sha, phaselet=cfg["sha256"],
                    stage_hook=hashlib.sha256((source/"native_stage_continuation_v1.py").read_bytes()).hexdigest(),
                    relative_executor=hashlib.sha256((source/"native_relative_phaselet_v1.py").read_bytes()).hexdigest())
    path = native.ROOT/"NATIVE_RELATIVE_PHASELET_RECEIPT.json"
    if path.exists():
        raise FileExistsError("Preserve existing relative operation receipt")
    gate = NativeRelativePhaseletChoice(session_id=case["id"], goal=case["task_description"],
        pause_steps=tuple(cfg["pause_steps"]), component_versions=versions, phaselet=phaselet, emit=emit)
    restore = gate.install()
    consumer = asyncio.create_task(consume(gate, cfg["operation"]))
    completed = False
    try:
        result = await original_case(case, accepted)
        completed = True
        return result
    finally:
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        restore()
        path.write_text(json.dumps(dict(authority=AUTHORITY, operation=cfg["operation"],
            original_case_completed=completed, original_goal=case["task_description"],
            versions=versions, audit=gate.audit_snapshot(), added_model_calls=0,
            added_memory_triggers=0, learned_selection=False, default_changed=False), indent=2)+"\n")


native.run_case = run_case
if __name__ == "__main__":
    asyncio.run(native.main())

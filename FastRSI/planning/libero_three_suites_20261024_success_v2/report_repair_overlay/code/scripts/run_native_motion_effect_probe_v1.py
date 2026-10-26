"""Original Agent case, developmental condition-bound native action candidate."""
import asyncio
import hashlib
import json
from pathlib import Path

import run_native_baseline as native
from cost_trace import emit
from fastrsi.native_motion_effect_skill_v1 import AUTHORITY, NativeMotionEffectSkill, load_skill

original_case = native.run_case


async def run_case(case, accepted):
    cfg = case.get("native_motion_effect_probe")
    if cfg is None:
        return await original_case(case, accepted)
    if (case.get("mode") != "development" or cfg.get("authority") != AUTHORITY
            or case.get("feedback_window_mode") != "fixed10"
            or case.get("inference_horizon", 10) != 10 or type(cfg.get("enabled")) is not bool):
        raise ValueError("Only the registered developmental H10/W10 probe is allowed")
    model_sha = hashlib.sha256((native.CAMPAIGN / "MODEL_CONFIG.json").read_bytes()).hexdigest()
    skill = load_skill(cfg["artifact"], cfg["sha256"], case["task_description"], model_sha)
    source = Path(__file__).resolve().parents[1] / "src/fastrsi"
    versions = dict(model_config=model_sha, skill=cfg["sha256"],
                    condition_runtime=hashlib.sha256((source / "native_motion_effect_skill_v1.py").read_bytes()).hexdigest())
    receipt = native.ROOT / "NATIVE_MOTION_EFFECT_SKILL_RECEIPT.json"
    if receipt.exists(): raise FileExistsError("Preserve existing operation receipt")
    gate = NativeMotionEffectSkill(session_id=case["id"], goal=case["task_description"],
                                   component_versions=versions, skill=skill, enabled=cfg["enabled"], emit=emit)
    restore = gate.install()
    completed = False
    try:
        result = await original_case(case, accepted)
        completed = True
        return result
    finally:
        restore()
        receipt.write_text(json.dumps(dict(authority=AUTHORITY, original_case_completed=completed,
            versions=versions, audit=gate.audit_snapshot(), default_changed=False,
            added_model_calls=0, added_memory_triggers=0, neural_distillation_complete=False,
            operation_quality_claimed=False), indent=2) + "\n")


native.run_case = run_case
if __name__ == "__main__":
    asyncio.run(native.main())

"""Original full Agent with a bounded immutable contextual action component."""
import asyncio
import hashlib
import json
from pathlib import Path
import run_native_baseline as native
from cost_trace import emit
from fastrsi.native_contextual_capability_v1 import NativeContextualCapability
from fastrsi.contextual_policy_probe_v1 import AUTHORITY,validate_evaluation,protect_evaluation_component

original_case=native.run_case

async def run_case(case,accepted):
    cfg=case.get('contextual_capability_probe')
    if cfg is None:return await original_case(case,accepted)
    if (case.get('mode') not in ('development','evaluation') or cfg['authority']!=AUTHORITY
            or case.get('feedback_window_mode')!='fixed10' or case.get('inference_horizon',10)!=10):
        raise ValueError('Registered fixed-component H10/W10 only')
    raw=Path(cfg['artifact']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=cfg['sha256']:raise ValueError('Component changed')
    protection=None
    if case['mode']=='evaluation':
        validate_evaluation(case,cfg)
        protection=protect_evaluation_component(cfg['artifact'],cfg['sha256'],emit)
    receipt=native.ROOT/'CONTEXTUAL_CAPABILITY_RECEIPT.json'
    if receipt.exists():raise FileExistsError(receipt)
    versions=dict(component=cfg['sha256'],model_config=hashlib.sha256((native.CAMPAIGN/'MODEL_CONFIG.json').read_bytes()).hexdigest())
    gate=NativeContextualCapability(session_id=case['id'],goal=case['task_description'],
        component_versions=versions,artifact_sha256=cfg['sha256'],emit=emit)
    restore=gate.install();completed=False
    try:
        result=await original_case(case,accepted);completed=True;return result
    finally:
        restore()
        if hashlib.sha256(Path(cfg['artifact']).read_bytes()).hexdigest()!=cfg['sha256']:
            raise ValueError('Component mutated during full Agent')
        receipt.write_text(json.dumps(dict(authority=AUTHORITY,completed=completed,mode=case['mode'],
            evaluation_component_protection=protection,
            audit=gate.audit_snapshot(),default_changed=False,neural_distillation_complete=False),indent=2)+'\n')

native.run_case=run_case
if __name__=='__main__':asyncio.run(native.main())

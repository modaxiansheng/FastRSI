"""Same native sampler target with a tagged public-position feedback skill."""
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from fastrsi.local_bowl_transition_v1 import PublicFeedbackTarget, KEY
from fastrsi.native_sampler_agent_binding_v1 import install
from fastrsi.live_controller_trace import LiveControllerTrace
from cost_trace import emit, span
from stage_trace import capture

spec = importlib.util.spec_from_file_location("frozen_target","/core/PhyAgentOS/runtime/targets/remote/libero/server.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
restorer = install(module,json.loads(Path(os.environ["FASTRSI_NATIVE_SAMPLER_CONFIG"]).read_text()))
config = json.loads(Path(os.environ["FASTRSI_COMPONENT_TARGET_CONFIG"]).read_text())
consumer = PublicFeedbackTarget(config)
root = Path(os.environ["FASTRSI_STAGE_TRACE"]).parent
reset0, action0 = module.LiberoRealRuntime.reset, module.LiberoRealRuntime.action_chunk


def record(runtime, method, chunk=None):
    emit(dict(event="native_observation",session_id=runtime.session_id,method=method,
              steps=runtime.step_idx,native_success=bool(runtime.env.check_success()),runtime_success=bool(runtime.success)))
    with span("stage_diagnostic_capture",session_id=runtime.session_id,step=runtime.step_idx):
        capture(runtime,method,() if chunk is None else (chunk,),os.environ["FASTRSI_STAGE_TRACE"])


def reset(runtime,*args,**kw):
    prior = getattr(runtime,"_fastrsi_live_controls",None)
    if prior is not None:
        prior.detach()
    previous = restorer.last_receipt
    try:
        with span("environment_reset",session_id=runtime.session_id):
            result = reset0(runtime,*args,**kw)
    finally:
        if restorer.last_receipt is not None and restorer.last_receipt is not previous:
            row = dict(event="native_sampler_restore",**restorer.last_receipt)
            emit(row)
            with (root/"NATIVE_SAMPLER_RESTORE.jsonl").open("a") as stream:
                stream.write(json.dumps(row)+"\n")
    with span("controller_target_logging",session_id=runtime.session_id,phase="setup"):
        runtime._fastrsi_live_controls = LiveControllerTrace(runtime,os.environ["FASTRSI_LIVE_CONTROLS"],runtime_source=spec.origin)
    record(runtime,"reset")
    return result


def action(runtime,chunk):
    logger = runtime._fastrsi_live_controls
    with span("environment_execution",session_id=runtime.session_id,method="action_chunk"):
        with logger.chunk(chunk):
            result, receipt = consumer.execute(runtime,chunk,action0)
            logger.bind_execution(receipt,source_skill_sha256=None if receipt is None else receipt["candidate_sha256"])
    if receipt is not None:
        row = dict(event=KEY,stage="actual_target_execution",session_id=runtime.session_id,**receipt)
        emit(row)
        with (root/"LOCAL_TRANSITION_EXECUTION.jsonl").open("a") as stream:
            stream.write(json.dumps(row)+"\n")
        chunk = {**chunk,"actions":receipt["actual_actions"]}
    record(runtime,"action_chunk",chunk)
    with span("controller_target_logging",session_id=runtime.session_id,phase="finalize"):
        logger.boundary(Path(os.environ["FASTRSI_STAGE_TRACE"])/runtime.session_id/"TRACE.jsonl")
    return result


observe0 = module.LiberoRealRuntime.observe
def observe(runtime,*args,**kw):
    with span("observation",session_id=runtime.session_id,method="observe"):
        return observe0(runtime,*args,**kw)

module.LiberoRealRuntime.reset, module.LiberoRealRuntime.action_chunk, module.LiberoRealRuntime.observe = reset, action, observe
module.main()

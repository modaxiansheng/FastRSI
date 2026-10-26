"""Extract one actually executed native chunk, not a post-hoc imagined branch."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


SOURCE = "runs/native_stage_continuation_probe_v1_20261006"
SESSION = "net_t4_i26"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_candidate(project, out):
    started = time.perf_counter()
    source = project/SOURCE
    stream = source/"N/stream"
    read = lambda path: json.loads(path.read_text())
    row = read(stream/"RESULT.json")["cases"][0]
    episode_path = stream/SESSION/"workspace/artifacts/runtime"/SESSION/"episode.json"
    episode = read(episode_path)
    receipt = read(stream/"NATIVE_STAGE_CONTINUATION_RECEIPT.json")
    admission = read(stream/SESSION/"admission/ADMISSION.json")
    if (not episode["success"] or row["chain_status"] != "PASSED"
            or admission["status"] != "ADMITTED" or receipt["enabled"]
            or row["case"]["feedback_window_mode"] != "fixed10"):
        raise ValueError("Source is not admitted untouched native execution")
    trace_path = stream/"stage_trace"/SESSION/"TRACE.jsonl"
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    cost_path = stream/"OS_COST_EVENTS.jsonl"
    inputs = [json.loads(line) for line in cost_path.read_text().splitlines()
              if '"actual_policy_input"' in line]
    chunks = [r for r in rows if r["method"] == "action_chunk"]
    if (len(inputs) != len(chunks) or sum(r["method"] == "reset" for r in rows) != 1
            or any(r["prompt"] != row["case"]["task_description"] for r in inputs)):
        raise ValueError("Source lacks matching actual native action/input provenance")
    selected = [(i,r) for i,r in enumerate(chunks) if r["step"] == 70]
    if len(selected) != 1:
        raise ValueError("Registered source chunk missing")
    index, chunk = selected[0]
    if index != 6 or chunks[index-1]["step"] != 60 or len(chunk["actions"]) != 10:
        raise ValueError("Source ten physical steps do not bind observation60")
    actions = np.asarray(chunk["actions"], dtype=np.float32)
    if actions.shape != (10,7) or not np.isfinite(actions).all():
        raise ValueError("Malformed source action buffer")
    files = [episode_path, trace_path, cost_path, stream/"RESULT.json",
             stream/SESSION/"admission/ADMISSION.json", source/"MODEL_CONFIG.json",
             source/"PROTOCOL.json", stream/"NATIVE_STAGE_CONTINUATION_RECEIPT.json",
             project/"scripts/compile_native_relative_phaselet_v1.py"]
    artifact = dict(version="source_native_relative_phaselet_v1", status="UNVALIDATED_CANDIDATE",
        goal=row["case"]["task_description"], source_native_success=True,
        source_experience_admission="ADMITTED", model_config_sha256=sha(source/"MODEL_CONFIG.json"),
        source_session=SESSION, source_physical_steps=[61,70],
        source_bindings={str(p.relative_to(project)):sha(p) for p in files},
        action_units="original_relative_OSC_native_commands", actions=actions.tolist(),
        actions_float32_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
        source_public_input=inputs[index], stage_or_grasp_label_certified=False,
        applicability="Unknown in new situations; one registered developmental operation-value probe only",
        source_cost_reference=SOURCE+"/N/STATUS.json", source_cost_free=False,
        original_goal_and_action_budget_unchanged=True, object_pose_estimator_used=False,
        absolute_target_waypoints_used=False, learned_eligibility=False)
    out.mkdir(parents=True,exist_ok=False)
    p=out/"CANDIDATE.json"
    p.write_text(json.dumps(artifact,indent=2)+"\n")
    summary=dict(status="COMPILED_UNVALIDATED_RELATIVE_CANDIDATE", candidate_sha256=sha(p),
                 cpu_seconds=time.perf_counter()-started, source_model_calls_repeated=0,
                 new_robot_actions=0, accepted=False, default_changed=False)
    (out/"COMPILE_RECEIPT.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--project",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True);args=parser.parse_args()
    print(json.dumps(compile_candidate(args.project,args.out)))

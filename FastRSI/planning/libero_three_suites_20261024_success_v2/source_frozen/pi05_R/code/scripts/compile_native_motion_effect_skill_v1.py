"""Build immutable condition versions from actual retained operation feedback."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from fastrsi.native_motion_effect_skill_v1 import public_vector


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_bytes())
def save(path, value): path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def example(project, trace_path, public_rows, boundary, bindings):
    trace = [json.loads(s) for s in trace_path.read_text().splitlines()]
    index = next(i for i, row in enumerate(trace) if row["step"] == boundary)
    previous, current = trace[index - 1], trace[index]
    images = []
    for camera in ("agentview_image", "robot0_eye_in_hand_image"):
        path = trace_path.parent / f'{boundary:04d}_{camera}.png'
        digest = sha(path)
        if digest != current["images"][path.name]: raise ValueError("Recorded source pixels changed")
        bindings[str(path.relative_to(project))] = digest
        with Image.open(path) as im: images.append(np.asarray(im.copy()))
    recent = [a[6] for i, row in enumerate(trace[1:index + 1], start=1)
              for a in row["actions"][:row["step"] - trace[i - 1]["step"]]][-20:]
    current_robot = public_rows[boundary]
    previous_robot = public_rows[previous["step"]]
    return public_vector(*images, current_robot, previous_robot, recent).tolist()


def compile_skill(project, output):
    start = time.perf_counter()
    output.mkdir(exist_ok=False)
    source = project / "runs/native_stage_continuation_probe_v1_20261006"
    transfer = project / "runs/native_relative_phaselet_value_probe_v2_20261006"
    template = read(project / "planning/native_relative_phaselet_value_probe_v1_20261006/CANDIDATE.json")
    bindings = dict(template["source_bindings"])
    for name, digest in bindings.items():
        if sha(project / name) != digest: raise ValueError("Original source changed")
    model_sha = sha(source / "MODEL_CONFIG.json")
    if model_sha != sha(transfer / "MODEL_CONFIG.json"): raise ValueError("Models differ")
    public_path = project / "planning/carry_effect_supervision_v1_20261006/PUBLIC_REPLAY_INPUTS.json"
    public = read(public_path)
    source_states = {row["step"]: row["robot8"] for row in public["26"]["rows"]}
    source_trace = source / "N/stream/stage_trace/net_t4_i26/TRACE.jsonl"
    rows = [json.loads(s) for s in source_trace.read_text().splitlines()]
    chunks = [row["actions"] for row in rows if row["step"] in (70, 80)]
    chunks = np.asarray(chunks, np.float32)
    if chunks.shape != (2, 10, 7) or not np.array_equal(chunks[0], np.asarray(template["actions"], np.float32)):
        raise ValueError("Previously executed first chunk changed")
    stream = transfer / "F/stream"
    result = read(stream / "RESULT.json")["cases"][0]
    if (result["chain_status"] != "PASSED" or result["environment_success"] is not True
            or result["experience_admission"]["status"] != "ADMITTED"):
        raise ValueError("Transfer feedback is not admitted successful actual execution")
    cost = stream / "OS_COST_EVENTS.jsonl"
    events = [json.loads(s) for s in cost.read_text().splitlines()]
    replaced = [e for e in events if e["event"] == "native_relative_phaselet_output"]
    inputs = [e for e in events if e["event"] == "actual_policy_input"]
    if len(replaced) != 1 or replaced[0]["physical_steps"] != 60:
        raise ValueError("Known actual relative operation is missing")
    if replaced[0]["actions_float32_sha256"] != hashlib.sha256(chunks[0].tobytes()).hexdigest():
        raise ValueError("Response is from different actions")
    states = {50: inputs[5]["robot_state"], 60: replaced[0]["current_public_robot8"], 70: inputs[6]["robot_state"]}
    transfer_trace = stream / "stage_trace/net_t4_i28/TRACE.jsonl"
    transfer_rows = [json.loads(s) for s in transfer_trace.read_text().splitlines()]
    after = next(row for row in transfer_rows if row["step"] == 70)
    if not np.array_equal(np.asarray(after["actions"], np.float32), chunks[0]):
        raise ValueError("Actual submitted transfer actions differ")
    negative_run = project / "runs/carry_fact_negative_challenge_v1_20261006"
    negative = read(negative_run / "job/REPLAY.json")
    negative_trace = project / read(negative_run / "REGISTRATION.json")["trace"]
    negative_states = {row["step"]: row["robot8"] for row in negative["rows"]}
    negatives = []
    for boundary in (80, 90, 100):
        row = next(row for row in negative["rows"] if row["step"] == boundary)
        if row["subjects"]["white_mug"]["carry_evidence"]["label"] != "NO_CURRENT_CONTACT":
            raise ValueError("Negative contact fact is uncertain")
        negatives.append(example(project, negative_trace, negative_states, boundary, bindings))
    positives = [example(project, source_trace, source_states, 60, bindings),
                 example(project, transfer_trace, states, 60, bindings)]
    effects = [np.asarray(source_states[70][:3]) - source_states[60][:3],
               np.asarray(states[70][:3]) - states[60][:3]]
    for path in (public_path, stream / "RESULT.json", cost, transfer_trace, transfer / "MODEL_CONFIG.json",
                 negative_run / "job/REPLAY.json", negative_trace, negative_run / "REGISTRATION.json",
                 project / "scripts/compile_native_motion_effect_skill_v1.py",
                 project / "src/fastrsi/native_motion_effect_skill_v1.py"):
        bindings[str(path.relative_to(project))] = sha(path)
    versions = {}
    for name, n in (("v1_seed", 1), ("v2_feedback_updated", 2)):
        entries = np.asarray([source_states[60][:3], states[60][:3]][:n])
        responses = np.asarray(effects[:n])
        artifact = dict(version="native_motion_effect_skill_v1", status="UNVALIDATED_DEVELOPMENT_CANDIDATE",
            goal=template["goal"], model_config_sha256=model_sha, chunks=chunks.tolist(),
            actions_sha256=hashlib.sha256(chunks.tobytes()).hexdigest(),
            action_units="original_relative_OSC_native_commands",
            positive_vectors=positives[:n], negative_vectors=negatives,
            entry_min=entries.min(axis=0).tolist(), entry_max=entries.max(axis=0).tolist(),
            response_min=responses.min(axis=0).tolist(), response_max=responses.max(axis=0).tolist(),
            source_bindings=bindings, grasp_certified=False, version_name=name,
            first_chunk_value_observed=True, second_chunk_value_observed=False,
            scope="First mixed closing then all-closed actual chunk, same complete task; two positive operation contexts and three noncontact proxy negatives. Unknown physical applicability, not safety certification.",
            second_chunk_feedback_window=10, output_cap=2, hidden_state_online=False,
            teacher_parameters_changed=False, student_neural_distillation_complete=False)
        path = output / (name + ".json"); save(path, artifact)
        versions[name] = dict(path=path.name, sha256=sha(path), positive_contexts=n)
    save(output / "VERSIONS.json", versions)
    save(output / "COMPILE_RECEIPT.json", dict(status="COMPILED_UNVALIDATED_MOTION_CONDITIONS",
        seconds=time.perf_counter() - start, new_model_calls=0, new_robot_episodes=0,
        first_chunk_response_updates=effects[1].tolist(), default_changed=False,
        action_continuation_quality_proven=False))
    print(json.dumps(versions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compile_skill(args.project, args.output)

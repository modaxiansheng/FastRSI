"""Load one H10/H20 model and replay paid development observations; no robot.

Two distinct resident Sessions replay the same inputs after begin/reset. Reusing
an ended Session identity or rewinding an active Session remains forbidden.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
import traceback

import numpy as np

from fastrsi.experimental_horizon_policy import ExperimentalHorizonPolicy
from fastrsi.resident_policy import ResidentPolicy


def sha(data):
    return hashlib.sha256(data).hexdigest()


def array_binding(array):
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": sha(array.tobytes())}


def load_inputs(protocol_path):
    """Validate the original saved request bytes and every source array binding."""
    protocol = json.loads(Path(protocol_path).read_text())
    specs = protocol["vla_replay"]["inputs"]
    if len(specs) != 3 or len({s["name"] for s in specs}) != 3:
        raise ValueError("Expected three distinct saved development inputs")
    observations, receipts = [], []
    expected_keys = {"observation/image", "observation/wrist_image", "observation/state", "prompt"}
    for spec in specs:
        path = Path(spec["path"])
        if sha(path.read_bytes()) != spec["sha256"]:
            raise ValueError("Saved observation SHA mismatch: " + str(path))
        with np.load(path, allow_pickle=False) as values:
            if set(values.files) != expected_keys or values["prompt"].shape != ():
                raise ValueError("Incomplete or unexpected observation fields")
            observation = {k: values[k].copy() for k in values.files if k != "prompt"}
            observation["prompt"] = str(values["prompt"].item())
        bindings = {k: array_binding(v) for k, v in observation.items() if k != "prompt"}
        if bindings != spec["source_array_bindings"] or not observation["prompt"].strip():
            raise ValueError("Observation source binding or prompt differs")
        for key, shape, dtype in [("observation/image", (256, 256, 3), "uint8"),
                                  ("observation/wrist_image", (256, 256, 3), "uint8"),
                                  ("observation/state", (8,), "float32")]:
            if observation[key].shape != shape or str(observation[key].dtype) != dtype:
                raise ValueError("Unexpected legal-input representation")
            if not np.isfinite(observation[key]).all():
                raise ValueError("Nonfinite observation")
        observations.append(observation)
        receipts.append({"name": spec["name"], "path": str(path), "npz_sha256": spec["sha256"],
                         "array_bindings": bindings, "prompt": observation["prompt"],
                         "prompt_sha256": sha(observation["prompt"].encode())})
    return observations, receipts


def replay_sessions(resident, observations, on_call, on_admin):
    """No warmup: all six real calls and both begin/end operations are charged."""
    sequences, begins = [], []
    for label in ("before", "after_reset"):
        session = "horizon-check-" + label
        begin = resident.admin(dict(op="begin", session_id=session,
                                    request_id=session + "-begin", max_inferences=len(observations)))
        begins.append(begin)
        on_admin(begin)
        connection = session + "-owner"
        resident.connect(connection)
        outputs = []
        try:
            for index, observation in enumerate(observations):
                request = {k: v.copy() if isinstance(v, np.ndarray) else v
                           for k, v in observation.items()}
                started = time.perf_counter()
                try:
                    response = resident.infer(request, connection_id=connection)
                    actions = np.asarray(response["actions"]).copy()
                except BaseException:
                    on_call({"session": label, "input_index": index, "status": "FAILED",
                             "seconds": time.perf_counter() - started}, None)
                    raise
                on_call({"session": label, "input_index": index, "status": "PASSED",
                         "seconds": time.perf_counter() - started,
                         "actions": array_binding(actions), "policy_meta": response.get("policy_meta"),
                         "policy_timing": response.get("policy_timing")}, actions)
                outputs.append(actions)
        finally:
            resident.disconnect(connection)
            end = resident.admin(dict(op="end", session_id=session, request_id=session + "-end"))
            on_admin(end)
        sequences.append(outputs)
    if begins[0]["rng_hashes"] != begins[1]["rng_hashes"]:
        raise ValueError("Resident begin did not restore the same RNG state")
    comparisons = [{"input_index": i, "replay_exact":
                    a.dtype == b.dtype and np.array_equal(a, b),
                    "shape": list(a.shape), "max_abs_difference": float(np.max(np.abs(a - b)))}
                   for i, (a, b) in enumerate(zip(*sequences))]
    if not all(c["replay_exact"] for c in comparisons):
        raise ValueError("Resident reset replay is not elementwise identical")
    status = resident.admin({"op": "status"})
    if status["active"] is not None or status["owner_connected"] or status["broken"]:
        raise ValueError("Resident replay did not release its Session")
    return {"replay_exact": True, "comparisons": comparisons, "release": status,
            "reset_scope": "Same resident model, two distinct bounded Sessions; no active Session rewind"}


def checkpoint_header(path):
    """Bind the supplied file layout without pretending a header hashes all weights."""
    path = Path(path) / "model.safetensors"
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        if not 0 < size <= 16 * 1024 * 1024:
            raise ValueError("Unexpected checkpoint header size")
        header = handle.read(size)
    if len(header) != size:
        raise ValueError("Truncated checkpoint header")
    json.loads(header)
    return {"path": str(path), "file_bytes": path.stat().st_size, "header_bytes": size,
            "header_sha256": sha(header), "scope": "Header only; full checkpoint provenance belongs to launch binding"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--horizon", type=int, choices=(10, 20), default=10)
    parser.add_argument("--pytorch-device", default="cuda:0")
    parser.add_argument("--torch-seed", type=int, choices=(7,), default=7)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = {"status": "RUNNING", "horizon": args.horizon, "torch_seed": args.torch_seed,
              "pytorch_device": args.pytorch_device, "calls": [], "admin_receipts": [],
              "robot_episodes": 0, "learning_updates": 0, "quality_or_speed_claim": False,
              "scope": "Model-load and reset-replay feasibility only; no robot task or controller comparison"}
    def save():
        result["full_wall_seconds"] = time.perf_counter() - started
        (args.output / "RESULT.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    save()
    try:
        result["protocol_sha256"] = sha(args.protocol.read_bytes())
        observations, result["inputs"] = load_inputs(args.protocol)
        result["checkpoint"] = checkpoint_header(args.checkpoint_dir)
        save()
        load_started = time.perf_counter()
        try:
            native = ExperimentalHorizonPolicy(checkpoint_dir=args.checkpoint_dir,
                action_horizon=args.horizon, pytorch_device=args.pytorch_device, torch_seed=args.torch_seed)
        finally:
            result["load_seconds"] = time.perf_counter() - load_started
            save()
        snapshot_started = time.perf_counter()
        resident = ResidentPolicy(native, max_sessions=2, max_inferences=3)
        result["resident_snapshot_seconds"] = time.perf_counter() - snapshot_started
        result["metadata"] = resident.metadata
        result["rng_hashes"] = resident.rng.hashes
        save()
        def on_call(receipt, actions):
            receipt["input_name"] = result["inputs"][receipt["input_index"]]["name"]
            if actions is not None:
                name = f"{receipt['session']}_{receipt['input_index']}.npy"
                np.save(args.output / name, actions, allow_pickle=False)
                receipt["actions_file"] = name
                receipt["actions_file_sha256"] = sha((args.output / name).read_bytes())
            result["calls"].append(receipt)
            save()
        def on_admin(receipt):
            result["admin_receipts"].append(receipt)
            save()
        result.update(replay_sessions(resident, observations, on_call, on_admin))
        result["status"] = "PASSED"
    except BaseException:
        result.update(status="FAILED", error=traceback.format_exc())
        raise
    finally:
        result["calls_seconds_sum"] = sum(c["seconds"] for c in result["calls"])
        save()
        print(json.dumps({k: result.get(k) for k in ("status", "horizon", "load_seconds",
              "calls_seconds_sum", "full_wall_seconds", "replay_exact", "robot_episodes")}), flush=True)


if __name__ == "__main__":
    main()

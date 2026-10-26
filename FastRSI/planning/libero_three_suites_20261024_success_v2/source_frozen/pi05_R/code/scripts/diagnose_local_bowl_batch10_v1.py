"""Read-only diagnosis of a completed, fixed ten-pair operation trial.

No robot/model calls; subsets are diagnostics, never replacement main scores.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path):
    return [json.loads(v) for v in path.read_text().splitlines() if v.strip()]


def totals(pairs):
    result = {}
    for arm in ("native", "transition"):
        values = [p["arms"][arm] for p in pairs]
        keys = ("actual_actions", "total_actual_vla_calls", "total_control_requests",
                "Qwen_requests", "full_wall_seconds", "startup_seconds", "local_motion_seconds")
        total = {k: sum(v[k] or 0 for v in values) for k in keys}
        total["successes"] = sum(v["environment_success"] is True for v in values)
        for group in ("os_cost", "target_cost"):
            names = sorted(set().union(*(v[group]["inclusive_seconds_do_not_sum_nested"] for v in values)))
            total[group] = {n: sum(v[group]["inclusive_seconds_do_not_sum_nested"].get(n, 0)
                                  for v in values) for n in names}
        result[arm] = total
    result["seeds"] = [p["seed"] for p in pairs]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    batch, out = args.batch.resolve(), args.output.resolve()
    aggregate = json.loads((batch / "AGGREGATE.json").read_text())
    pairs = aggregate["pairs"]
    assert len(pairs) == 10
    common = [p for p in pairs if all(p["arms"][a]["environment_success"]
                                    for a in ("native", "transition"))]
    assert len(common) == 6
    sources = {}
    for f in sorted((out / "code").glob("*.py")):
        sha = digest(f)
        checks = []
        for p in pairs:
            manifest = json.loads((batch / "raw" / f"transition_{p['seed']}" / "CODE_MANIFEST.json").read_text())
            key = next(k for k in manifest if k.endswith("/" + f.name))
            assert manifest[key] == sha
            checks.append(p["seed"])
        sources[f.name] = dict(sha256=sha, batch_seeds_exact=checks)

    handoffs = []
    for pair in pairs:
        b = pair["arms"]["transition"]
        if not b["local_entry"]:
            continue
        p = batch / "raw" / f"transition_{pair['seed']}"
        sid = b["root_session"]
        receipt = json.loads((p / "stream" / sid / "LOCAL_TRANSITION_RECEIPT.json").read_text())
        handoff = b["local_handoff"][0]
        step = handoff["step"]
        entry = b["local_entry"][0]["step"]
        executions = lines(p / "stream/LOCAL_TRANSITION_EXECUTION.jsonl")
        inputs = [e for e in lines(p / "stream/OS_COST_EVENTS.jsonl")
                  if e["event"] == "actual_policy_input"]
        hashes = {}
        for key, name in (("observation/image", "agentview_image"),
                          ("observation/wrist_image", "robot0_eye_in_hand_image")):
            frame = np.asarray(Image.open(p / f"stream/stage_trace/{sid}/{step:04d}_{name}.png"))
            hashes[key] = hashlib.sha256(frame.tobytes()).hexdigest()
        matches = [(i, e) for i, e in enumerate(inputs)
                   if all(e["array_bindings"][k]["sha256"] == v for k, v in hashes.items())]
        assert len(matches) == 1
        index, first = matches[0]
        # Native inference every ten actual steps before entry; local calls bypass it.
        assert index == entry // 10
        assert handoff["reason"] == "LOCAL_ACTION_CAP"
        goal = np.asarray(executions[-1]["request"]["goal_xyz"])
        state = np.asarray(first["robot_state"])
        error = float(np.linalg.norm(state[:3] - goal))
        assert error > 0.01
        controls = lines(p / f"stream/live_controls/{sid}/CONTROLLER_TARGETS.jsonl")
        assert controls[step]["execution_origin"] == "native_action"
        assert receipt["local_actions"] == 40 and receipt["local_requests"] == 8
        handoffs.append(dict(seed=pair["seed"], entry_step=entry, handoff_step=step,
                             reason=handoff["reason"], native_input_index=index,
                             front_and_wrist_pixels_match_current_boundary=True,
                             first_post_handoff_action_origin=controls[step]["execution_origin"],
                             current_robot8=state.tolist(), last_requested_goal_xyz=goal.tolist(),
                             residual_to_last_requested_goal_m=error,
                             residual_is_not_object_pose_or_success_certificate=True))

    nominal_native_requests_b = sum(math.ceil(p["arms"]["transition"]["actual_actions"] / 10) for p in pairs)
    assert nominal_native_requests_b == 311
    assert aggregate["groups"]["transition"]["total_control_requests"] == nominal_native_requests_b + 36
    result = dict(
        scope="Existing trajectories only; fixed manual operation, not learning or held-out evaluation",
        source_aggregate_sha256=digest(batch / "AGGREGATE.json"), source_code=sources,
        all_ten_main=totals(pairs), common_success_diagnostic=totals(common),
        common_success_without_640720_sensitivity_only=totals([p for p in common if p["seed"] != 640720]),
        handoffs=handoffs,
        request_decomposition=dict(native_A=293, B_same_actions_nominal_H10=311,
                                   local_40_steps_replace_four_VLA_per_entry=36,
                                   local_eight_requests_per_entry=72,
                                   B_actual_VLA=275, B_actual_controls=347,
                                   extra_requests_from_5_vs_10_chunking=36,
                                   extra_H10_equivalent_requests_from_longer_execution=18),
        decision="SEAL_CURRENT_OPERATION_NO_TARGETED_REPAIR_OR_NEW_ROLLOUT",
        reason="Observed premature entry is not a proven sufficient economic fix; common-success aggregate also adds work and cost",
        causal_root_of_640776="NOT_IDENTIFIED",
        new_robot_runs=0, new_model_calls=0, new_training=0,
        default_modified=False, learned_gain=False,
        nested_costs_are_not_additive=True,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "DIAGNOSIS.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(common_success=result["common_success_diagnostic"],
                          fresh_handoffs=len(handoffs), decision=result["decision"]), indent=2))


if __name__ == "__main__":
    main()

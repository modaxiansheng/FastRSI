"""Replay previously executed native development actions for carry evidence.

Every boundary must reproduce saved pixels and public policy state before any
privileged annotation is eligible for a later offline learner. No model calls,
counterfactual actions, OS memory writes or live-controller privileges.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

from fastrsi.carry_effect_supervision_v1 import PROTOCOL, carry_evidence
from replay_object_stage_supervision_v2 import CAMERAS, GOAL, RUNTIME, static_roles, validate_source
from stage_trace import png_bytes


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def run(args):
    if args.output.exists():
        raise FileExistsError("Replay output already exists")
    trace_blob, result_blob, cost_blob = (p.read_bytes() for p in (args.trace, args.result, args.cost))
    rows = [json.loads(line) for line in trace_blob.splitlines()]
    case = validate_source(json.loads(result_blob), rows, args.init)
    mixed = getattr(args, "mixed_facts_only", False)
    stop = getattr(args, "stop_step", None)
    if stop is not None:
        if not mixed or stop not in [r["step"] for r in rows] or stop <= 0:
            raise ValueError("A declared mixed-source diagnostic requires a recorded stop boundary")
        rows = [r for r in rows if r["step"] <= stop]
    inputs = [json.loads(line) for line in cost_blob.splitlines()
              if json.loads(line).get("event") == "actual_policy_input"]
    if not mixed and (len(inputs) != len(rows)-1 or any(r.get("prompt") != GOAL for r in inputs)):
        raise ValueError("Source is not one purely native whole-goal action trace")
    report = dict(status="RUNNING", case=case, protocol=PROTOCOL,
                  source_trace_sha256=sha(trace_blob), source_result_sha256=sha(result_blob),
                  source_cost_sha256=sha(cost_blob), rows=[], online_input_used=False,
                  new_policy_calls=0, new_model_calls=0, new_online_episodes=0,
                  training_permission="NONE_UNTIL_EXACT_REPLAY_COMPLETE",
                  action_value_supervision=False, reuse_authorized=False)
    report.update(source_actions_are_native_expert=not mixed,
                  source_kind="mixed_action_facts_only" if mixed else "native_action_facts",
                  original_policy_input_state_verified=not mixed,
                  registered_stop_step=stop)
    started = time.perf_counter()
    runtime = None
    try:
        spec = importlib.util.spec_from_file_location("carry_effect_native_runtime", RUNTIME)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        report["runtime_source_sha256"] = sha(RUNTIME.read_bytes())
        runtime = module.LiberoRealRuntime(dict(benchmark_name="libero_10", task_id=4,
            init_state_id=args.init, seed=7, max_steps=520, control_mode="relative"))
        histories, initial_z = {}, {}
        for index, entry in enumerate(rows):
            if index == 0:
                runtime.reset({"session_id": case["id"]})
            else:
                runtime.action_chunk({"actions": entry["actions"]})
            if runtime.step_idx != entry["step"] or bool(runtime.success) != entry["native_success"]:
                raise ValueError("Source physical boundary/result changed")
            for camera in CAMERAS:
                name = f'{entry["step"]:04d}_{camera}.png'
                if sha(png_bytes(runtime._last_obs[camera])) != entry["images"][name]:
                    raise ValueError("Source pixels differ: " + name)
            robot = module._libero_state(runtime._last_obs)
            if not mixed and index < len(inputs):
                actual = inputs[index]
                if (actual.get("case") != case["id"]
                        or np.asarray(actual.get("robot_state"), dtype=np.float32).tobytes() != robot.tobytes()):
                    raise ValueError("Source public policy state differs")
                for key, camera in (("observation/image", CAMERAS[0]), ("observation/wrist_image", CAMERAS[1])):
                    if actual["array_bindings"][key]["sha256"] != sha(runtime._last_obs[camera].tobytes()):
                        raise ValueError("Policy image does not bind source trace")
            env = runtime.env.env
            if index == 0:
                roles = static_roles(env)
                if runtime.language != GOAL:
                    raise ValueError("Loaded task identity changed")
            grip = env.sim.data.site_xpos[env.robots[0].eef_site_id].copy().tolist()
            closed = bool(index and entry["actions"][entry["step"]-rows[index-1]["step"]-1][6] > 0)
            subjects = {}
            for subject, predicate in roles.items():
                obj = env.get_object(predicate[1])
                objpos = env.sim.data.get_body_xpos(obj.root_body).copy().tolist()
                initial_z.setdefault(subject, objpos[2])
                contact = bool(env.check_contact(env.robots[0].gripper, obj))
                native_grasp = bool(env._check_grasp(env.robots[0].gripper, obj.contact_geoms))
                fact = dict(step=entry["step"], contact=contact, closed_command=closed,
                            object_world_position=objpos, grip_site_world_position=grip,
                            initial_object_z=initial_z[subject])
                histories.setdefault(subject, []).append(fact)
                subjects[subject] = dict(privileged_label_evidence=fact,
                    native_two_finger_grasp_diagnostic=native_grasp,
                    native_subgoal_diagnostic=bool(env._eval_predicate(predicate)),
                    carry_evidence=carry_evidence(histories[subject]))
            report["rows"].append(dict(step=entry["step"], images=entry["images"],
                robot8=robot.tolist(), subjects=subjects, exact_images=True,
                actual_native_success=bool(runtime.success)))
        report.update(status="EXACT_IMAGE_REPLAY_RECONSTRUCTED_PUBLIC_STATE" if mixed else "EXACT_IMAGE_AND_PUBLIC_STATE_REPLAY",
                      training_permission="DEVELOPMENT_CARRY_FACTS_ONLY",
                      source_boundaries=len(rows), action_steps=rows[-1]["step"])
    except BaseException as exc:
        report.update(status="FAILED", error=repr(exc)); raise
    finally:
        report["seconds"] = time.perf_counter()-started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle: json.dump(report, handle, indent=2, allow_nan=False)
        if runtime is not None and runtime.env is not None: runtime.env.close()
    print(json.dumps({k: report[k] for k in ("status", "seconds", "source_boundaries")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("trace", "result", "cost", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--init", type=int, required=True)
    parser.add_argument("--mixed-facts-only", action="store_true")
    parser.add_argument("--stop-step", type=int)
    run(parser.parse_args())

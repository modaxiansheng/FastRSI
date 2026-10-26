"""One isolated P/F service lifetime for a frozen native dev/dev/eval stream."""
import argparse
import hashlib
import json
import math
import re
import shutil
import signal
import socket
import sys
import time
import traceback
from pathlib import Path

import orchestrate_full_os as life
from native_service_config import configure_qwen, model_env


def read(path):
    return json.loads(Path(path).read_text())


def arm_binding(protocol, arm):
    expected = {"P": {"gpu_base": 0, "port_offset": 0},
                "F": {"gpu_base": 2, "port_offset": 100}}
    if arm not in expected:
        raise ValueError("Unknown comparison arm")
    bound = protocol.get("arms", {}).get(arm)
    if not isinstance(bound, dict) or any(type(bound.get(k)) is not int or bound[k] != v
                                          for k, v in expected[arm].items()):
        raise ValueError("Arm GPU/port placement differs from declared P/F allocation")
    deadline = protocol.get("absolute_deadline")
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= 0:
        raise ValueError("Missing absolute campaign deadline")
    return dict(expected[arm])


def relocate_argv(argv, binding):
    """Remap only the service ports and exact visibility environment values."""
    shift = binding["port_offset"]
    base = binding["gpu_base"]
    out = []
    for value in argv:
        value = re.sub(r"(?<!\d)(49500|49502|49510|49520)(?!\d)",
                       lambda m: str(int(m.group()) + shift), value)
        for prefix in ("HIP_VISIBLE_DEVICES=", "CUDA_VISIBLE_DEVICES="):
            if value == prefix + "0":
                value = prefix + str(base)
            elif value == prefix + "1":
                value = prefix + str(base + 1)
        out.append(value)
    return out


def validate_roster(cases, arm, campaign_base, flow_sha):
    if not isinstance(cases, list) or len(cases) != 3 or [c.get("mode") for c in cases] != ["development", "development", "evaluation"]:
        raise ValueError("Expected a frozen development/development/evaluation roster")
    ids = [c.get("id") for c in cases]
    if any(not isinstance(x, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in ids) or len(set(ids)) != 3:
        raise ValueError("Invalid or repeated case identity")
    keys = [c.get("conversation_key") for c in cases]
    if any(not isinstance(k, str) or not k for k in keys) or keys[0] != keys[1] or keys[2] == keys[0]:
        raise ValueError("Development must share a key; evaluation must have its own key")
    for case in cases:
        if case.get("suite") != "libero_10" or case.get("max_steps") != 520:
            raise ValueError("Unbound benchmark or execution limit")
        if any(type(case.get(k)) is not int or case[k] < 0 for k in ("task_id", "init_state_id")):
            raise ValueError("Invalid task/initial-state identity")
        if not isinstance(case.get("task_description"), str) or not case["task_description"].strip():
            raise ValueError("Missing benchmark task text")
        if case.get("session_intent") is not True:
            raise ValueError("Comparison requires the supplied native Session intent")
        if arm == "P":
            if case.get("session_flow_artifact") or case.get("session_flow_sha256"):
                raise ValueError("P must use original Agent planning")
        elif arm == "F":
            if (case.get("session_flow_artifact") != campaign_base + "/FLOW.json"
                    or case.get("session_flow_sha256") != flow_sha):
                raise ValueError("F flow does not match the frozen shared artifact")
        else:
            raise ValueError("Unknown comparison arm")


def validate_stream_report(report, cases, stream_base):
    """Validate controller completion separately from task success or learning."""
    if report.get("status") != "COMPLETED" or report.get("experience_admission_mode") != "guarded_v1":
        raise ValueError("Native stream did not finish under guarded admission")
    rows = report.get("cases", [])
    if len(rows) != 3 or len(cases) != 3:
        raise ValueError("Missing stream cases")
    previous = None
    attempts = 0
    transitions = []
    for index, (row, case) in enumerate(zip(rows, cases)):
        if row.get("case") != case or row.get("chain_status") != "PASSED" or row.get("status") != "PASSED":
            raise ValueError("Case identity or complete chain differs from roster")
        if row.get("accepted_source") != previous:
            raise ValueError("Case inherited an unexpected experience source")
        outcomes = row.get("outcomes")
        if not isinstance(outcomes, list) or not 1 <= len(outcomes) <= 3:
            raise ValueError("Case attempt bound violated")
        attempts += len(outcomes)
        terminal = row.get("terminal_report", {})
        if terminal.get("status") != "PASSED" or terminal.get("read_coverage", {}).get("complete") is not True:
            raise ValueError("Native terminal consumption did not complete")
        checkpoint = row.get("budget_checkpoint")
        if "budget_checkpoint" in case:
            if (not isinstance(checkpoint, dict)
                    or checkpoint.get("status") not in ("NOT_DISPATCHED", "AUTHORIZED_PENDING", "DEFERRED_NO_ACCEPTED_SOURCE")
                    or checkpoint.get("actual_worker_dispatch") is not False):
                raise ValueError("Budget checkpoint failed, is missing, or dispatched an unbound worker")
        inherited = previous
        if index < 2:
            admission = row.get("experience_admission", {})
            status = admission.get("status")
            if status == "ADMITTED":
                previous = stream_base + "/" + case["id"] + "/admission/accepted_experience"
            elif status != "QUARANTINED":
                raise ValueError("Development admission missing or invalid")
            if admission.get("next_source") != previous:
                raise ValueError("Admission pointer does not match its actual transition")
        else:
            status = "EVALUATION_NO_UPDATE"
            before, after = row.get("experience_before"), row.get("experience_after")
            if not isinstance(before, dict) or not before or before != after:
                raise ValueError("Evaluation persistent experience changed or is unbound")
            if row.get("experience_admission", {}).get("status") not in (None, "EVALUATION_NO_UPDATE"):
                raise ValueError("Evaluation promoted experience")
        transitions.append({"case": case["id"], "mode": case["mode"], "source": inherited,
                            "admission": status, "next_source": previous,
                            "budget_checkpoint": checkpoint,
                            "environment_success": row.get("environment_success"),
                            "verdicts": [o.get("verdict") for o in outcomes]})
    if attempts > 9:
        raise ValueError("Stream exceeds nine execution attempts")
    return {"attempts": attempts, "transitions": transitions, "final_accepted_source": previous,
            "scope": "Complete chain and admitted inheritance only; not proof of success or learned benefit"}


def check_artifacts(stream, summary):
    """Cross-check emitted admission/inheritance records, not just final fields."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from fastrsi.experience_admission import validate_snapshot
    for transition in summary["transitions"]:
        folder = stream / transition["case"]
        inheritance = read(folder / "INHERITANCE.json")
        if inheritance.get("source") != transition["source"] or inheritance.get("preserved_before_agent") is not True:
            raise ValueError("Saved workspace inheritance disagrees with controller")
        if transition["mode"] == "evaluation":
            if not (folder / "workspace/.FASTRSI_EVALUATION_CLOSED.json").is_file():
                raise ValueError("Evaluation close receipt missing")
            if (folder / "admission").exists():
                raise ValueError("Evaluation unexpectedly published an admission")
            continue
        receipt = read(folder / "admission/ADMISSION.json")
        if (receipt.get("status") != transition["admission"]
                or receipt.get("next_source") != transition["next_source"]
                or receipt.get("parent") != transition["source"]):
            raise ValueError("Saved admission record disagrees with controller")
        if transition["admission"] == "ADMITTED":
            validate_snapshot(folder / "admission/accepted_experience")
        elif (folder / "admission/accepted_experience").exists():
            raise ValueError("Quarantine contains a published experience source")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("arm", choices=("P", "F"))
    args = parser.parse_args()
    name, arm = args.name, args.arm
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Invalid campaign name")
    if life.active:
        raise RuntimeError("Worker must start with an empty owned-service list")
    campaign = life.PROJECT / "runs" / name
    group = campaign / arm
    protocol = read(campaign / "PROTOCOL.json")
    placement = arm_binding(protocol, arm)
    life.CAMPAIGN = group
    life.BASE = "/project/runs/" + name + "/" + arm
    life.deadline = protocol["absolute_deadline"]
    cases = read(group / "ROSTER.json")
    cfg = read(group / "MODEL_CONFIG.json")
    flow_sha = hashlib.sha256((campaign / "FLOW.json").read_bytes()).hexdigest()
    validate_roster(cases, arm, "/project/runs/" + name, flow_sha)
    read(group / "PROTOCOL.json")  # required frozen group input, never rewritten
    if not (group / "code/scripts/run_native_baseline.py").is_file():
        raise ValueError("Frozen group runner is missing")
    frozen_inputs = {filename: hashlib.sha256((group / filename).read_bytes()).hexdigest()
                     for filename in ("ROSTER.json", "MODEL_CONFIG.json", "PROTOCOL.json")}
    stream = group / "stream"
    stream.mkdir(exist_ok=False)
    base = life.BASE + "/stream"
    started = time.time()
    state = {"arm": arm, "status": "STARTING", "started_at": started,
             "startup_seconds": None, "controller_seconds": None,
             "placement": placement, "frozen_inputs": frozen_inputs}
    life.save(group / "STATUS.json", state)

    def interrupted(signum, frame):
        raise KeyboardInterrupt("Interrupted comparison worker " + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if life.deadline - time.time() < 180:
            raise TimeoutError("Insufficient remaining stream budget")
        shutil.copytree(life.PROJECT / "models/openpi_data", stream / "tokenizer_cache")
        services = {}
        for role, receipt in (("pi05", "LAUNCH_pi05_v2.json"), ("qwen", "LAUNCH_qwen.json"), ("target", "LAUNCH_target.json")):
            argv = list(read(life.OLD / receipt)["argv"])
            argv[argv.index("--name") + 1] = "yuchen-fastrsi-" + name + "-" + arm.lower() + "-" + role
            argv = [v.replace(str(life.OLD), str(stream)).replace("/project/runs/pilot_20261001", base) for v in argv]
            if role == "pi05":
                argv = [v.replace(str(life.PROJECT / "models/openpi_data"), str(stream / "tokenizer_cache")) for v in argv]
                if "--torch-seed 7" not in argv[-1]:
                    raise ValueError("Policy seed differs from the bound receipt")
            elif role == "qwen":
                argv = configure_qwen(argv, cfg, life)
            else:
                argv = [v.replace("/project/scripts/instrument_target.py", life.BASE + "/code/scripts/instrument_target.py")
                        .replace(":/project/scripts", ":" + life.BASE + "/code/scripts") for v in argv]
                if argv[argv.index("--seed") + 1] != "7":
                    raise ValueError("Target seed differs from the bound receipt")
            argv = relocate_argv(argv, placement)
            at = argv.index("timeout") + 1
            lifetime = min(int(argv[at]), int(life.deadline - time.time()))
            if lifetime < 60:
                raise TimeoutError("Insufficient service lifetime")
            argv[at] = str(lifetime)
            services[role] = life.launch(role, argv, stream)
        offset = placement["port_offset"]
        life.health(49510 + offset)
        life.health(49500 + offset)
        until = min(time.time() + 120, life.deadline)
        while True:
            try:
                with socket.create_connection(("127.0.0.1", 49502 + offset), timeout=1):
                    break
            except OSError:
                if time.time() >= until:
                    raise
                time.sleep(1)
        task_ids = sorted({case["task_id"] for case in cases})
        command = ("import json; from libero.libero import benchmark; s=benchmark.get_benchmark_dict()['libero_10'](); "
                   "print('TASKS='+json.dumps({i:s.get_task(i).language for i in " + repr(task_ids) + "}))")
        receipt = life.command(["docker", "exec", services["target"], "python", "-c", command], timeout=45)
        life.save(group / "LIVE_TASK_BINDING.json", receipt)
        if receipt["returncode"] != 0:
            raise RuntimeError("Live benchmark language lookup failed")
        languages = json.loads(next(line[6:] for line in receipt["stdout"].splitlines() if line.startswith("TASKS=")))
        if any(case["task_description"] != languages[str(case["task_id"])] for case in cases):
            raise ValueError("Frozen task text differs from the live benchmark")
        remaining = int(life.deadline - time.time() - 30)
        if remaining < 180:
            raise TimeoutError("Insufficient stream budget after startup")
        state.update(status="RUNNING", startup_seconds=time.time() - started)
        life.save(group / "STATUS.json", state)
        argv = life.cpu_args("yuchen-fastrsi-" + name + "-" + arm.lower() + "-runner", stream) + model_env(cfg) + [
            "-e", "FASTRSI_NATIVE_RUN=" + base, "-e", "FASTRSI_CONSUME_TERMINAL=1",
            "-e", "FASTRSI_EXPERIENCE_ADMISSION=guarded_v1", life.IMAGE, "timeout", str(remaining), "bash", "-lc",
            "source /opt/dtk/env.sh && unset FASTRSI_INITIAL_EXPERIENCE && exec /project/envs/os-v1/bin/python "
            + life.BASE + "/code/scripts/run_native_baseline.py"]
        runner = life.launch("runner", argv, stream)
        life.wait_exit(runner, remaining + 5)
        report = read(stream / "RESULT.json")
        state["controller_seconds"] = report["seconds"]
        summary = validate_stream_report(report, cases, base)
        check_artifacts(stream, summary)
        if frozen_inputs != {filename: hashlib.sha256((group / filename).read_bytes()).hexdigest() for filename in frozen_inputs}:
            raise ValueError("Frozen group inputs changed during execution")
        state.update(status="COMPLETED", **summary)
    except BaseException:
        state.update(status="FAILED", error=traceback.format_exc())
    finally:
        try:
            stopped = life.stop_all()  # this fresh process records only its own launches
            life.save(stream / "STOP_RECEIPT.json", stopped)
            if any(row["running"] for row in stopped):
                raise RuntimeError("An owned container remains running")
        except BaseException:
            state.update(status="FAILED", cleanup_error=traceback.format_exc())
        state.update(finished_at=time.time(), full_wall_seconds=time.time() - started)
        life.save(group / "STATUS.json", state)
        print(json.dumps(state), flush=True)
    if state["status"] != "COMPLETED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

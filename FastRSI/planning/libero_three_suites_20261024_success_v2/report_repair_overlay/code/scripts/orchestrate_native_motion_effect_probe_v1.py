"""One real N/F motion-effect skill operation-value check using the original full-OS worker.

Development task4/init36; no elapsed-stage oracle or extra Agent calls.
Same H10 feedback; at most two original-relative chunks under public conditions.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import traceback

import orchestrate_full_os as life


NAME = "native_motion_effect_skill_value_v1_20261006"
SOURCE = "runs/native_stage_continuation_probe_v1_20261006"
GOAL = "put the white mug on the left plate and put the yellow and white mug on the right plate"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree(path):
    return {str(p.relative_to(path)): sha(p) for p in path.rglob("*") if p.is_file()}


def freeze(source, target):
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def main():
    project = life.PROJECT
    run = project/"runs"/NAME
    lock = (project/"runs/full_os_v1.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run.mkdir(exist_ok=False)
    started = time.time()
    life.deadline = started+1200
    life.CAMPAIGN = run
    life.BASE = "/project/runs/"+NAME
    workers = []
    state = dict(status="PREPARING", started_at=started,
                 authority="MOTION_EFFECT_SKILL_DEVELOPMENT_VALUE")
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Interrupted owned continuity check "+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        life.gate(run)
        for port in [p+shift for p in (49500, 49502, 49510, 49520) for shift in (0, 100)]:
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    raise ValueError("Registered continuity port already occupied")
        processes = life.command(["ps", "-eo", "pid,ppid,args"])
        if processes["returncode"]:
            raise RuntimeError("Cannot inspect current experiment processes")
        foreign = [line for line in processes["stdout"].splitlines()
                   if any(key in line for key in ("run_placement_worker.py", "native_openpi_server.py",
                                                 "qwen_agent_server.py"))]
        life.save(run/"PROCESS_CHECK.json", dict(checked_at=time.time(), matching_processes=foreign))
        if foreign:
            raise ValueError("Existing live worker/service; inspect before dispatch")
        source = project/SOURCE
        for relative in ("MODEL_CONFIG.json", "FLOW.json"):
            shutil.copyfile(source/relative, run/relative)
        freeze(source/"N/stream/net_t4_i26/admission/accepted_experience", run/"initial_experience")
        life.save(run/"EXPERIENCE_BINDING.json", tree(run/"initial_experience"))
        candidate=project/"planning/native_motion_effect_skill_v2_20261006/v2_feedback_updated.json"
        from fastrsi.native_motion_effect_skill_v1 import load_skill
        artifact=load_skill(candidate,sha(candidate),GOAL,sha(run/"MODEL_CONFIG.json"))
        for rel,digest in artifact["source_bindings"].items():
            if sha(project/rel)!=digest:raise ValueError("Phaselet source evidence changed: "+rel)
        shutil.copyfile(candidate,run/"MOTION_SKILL.json")
        for relative in ("scripts", "src/fastrsi"):
            freeze(project/relative, run/"code"/relative)
        case = dict(suite="libero_10", task_id=4, init_state_id=36,
            mode="development", max_steps=520, native_experience_maintenance=True,
            task_description=GOAL)
        life.save(run/"PLAN.json", dict(cases=[case]))
        life.save(run/"PROTOCOL.json", dict(slot=0, absolute_deadline=life.deadline,
            diagnostic_trace=True, control_kind="feedback_window",
            arm_modes=dict(N="fixed10", F="fixed10"), terminal_evidence_view=True,
            planning_entries=dict(N="public_session_flow", F="public_session_flow"),
            skill_sha256=sha(run/"MOTION_SKILL.json"),
            changed_goal=False, stage_controller_learning_disabled=True,
            trigger="First actually committed mixed closing then all-positive ten-step chunk; no elapsed step as stage label",
            entry="Legal dual RGB/proprioception/action history vs source26 and actual successful relative-operation28; noncontact facts only as rejection examples",
            continuation="Second ten steps only if live measured first-chunk movement matches the learned response envelope; otherwise original policy",
            resources="two cards per arm, four cards parallel; same models/seeds/limits",
            operation_cap=2, physical_feedback_window=10,
            native_feedback_original_rules=True,
            scope="Task4/init36 developmental operation value; local manifests contain no task4/init36 trajectory, other tasks at36 are not same task",
            stopping="One registered pair, full original goal and total520. Include all failures, local decisions, physical actions, model calls and costs. No test isolation or net-learning claim."))
        life.save(run/"SOURCE_BINDINGS.json", {
            SOURCE+"/MODEL_CONFIG.json": sha(source/"MODEL_CONFIG.json"),
            SOURCE+"/N/stream/net_t4_i26/admission/ADMISSION.json": sha(source/"N/stream/net_t4_i26/admission/ADMISSION.json"),
            SOURCE+"/N/stream/stage_trace/net_t4_i26/TRACE.jsonl":
                sha(source/"N/stream/stage_trace/net_t4_i26/TRACE.jsonl")})
        worker = run/"code/scripts/run_placement_worker.py"
        text = worker.read_text()
        needle = "        case['terminal_prefetch']=False"
        if text.count(needle) != 1:
            raise ValueError("Original worker binding point changed")
        text = text.replace(needle, needle+"\n        case['native_motion_effect_probe']=dict(authority='MOTION_EFFECT_SKILL_DEVELOPMENT_VALUE',enabled=arm=='F',artifact='/project/'+str((campaign/'MOTION_SKILL.json').relative_to(life.PROJECT)),sha256=protocol['skill_sha256'])")
        needle = "runner_script='run_native_baseline.py'"
        if text.count(needle) != 1:
            raise ValueError("Original runner selection changed")
        worker.write_text(text.replace(needle, "runner_script='run_native_motion_effect_probe_v1.py'"))
        life.save(run/"CODE_MANIFEST.json", tree(run/"code"))
        for arm in ("N", "F"):
            group = run/arm
            freeze(run/"code", group/"code")
            if arm == "F":
                for relative in ("scripts/run_native_baseline.py", "scripts/qwen_agent_server.py"):
                    path = group/"code"/relative
                    text = path.read_text()
                    for port in (49500, 49502, 49510, 49520):
                        text = text.replace(str(port), str(port+100))
                    path.write_text(text)
            life.save(group/"CODE_MANIFEST.json", tree(group/"code"))
        state.update(status="REGISTERED", registration="PROTOCOL.json")
        life.save(run/"STATUS.json", state)
        for arm in ("N", "F"):
            with (run/arm/"controller.log").open("x") as log:
                process = subprocess.Popen(["python3", str(run/"code/scripts/run_placement_worker.py"), NAME, arm],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            workers.append((arm, process))
            life.save(run/arm/"DISPATCH.json", dict(pid=process.pid, started_at=time.time()))
        state["status"] = "RUNNING"
        life.save(run/"STATUS.json", state)
        while any(proc.poll() is None for _, proc in workers):
            if time.time() > life.deadline:
                raise TimeoutError("Bounded continuity check expired")
            if any(proc.poll() not in (None, 0) for _, proc in workers):
                raise RuntimeError("One actual worker failed; preserve both results")
            time.sleep(2)
        arm_results={arm: json.loads((run/arm/"STATUS.json").read_text()) for arm,_ in workers}
        if any(proc.returncode!=0 for _,proc in workers) or any(r["status"]!="COMPLETED" for r in arm_results.values()):
            raise RuntimeError("Both workers ended; at least one failed. Preserve failed receipts.")
        state.update(status="COMPLETED", arms=arm_results)
    except BaseException:
        state.update(status="FAILED", error=traceback.format_exc())
        raise
    finally:
        for _, proc in workers:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for _, proc in workers:
            try:
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                state["cleanup_attention"] = "Inspect these exact owned worker PIDs"
        state.update(finished_at=time.time(), seconds=time.time()-started)
        life.save(run/"STATUS.json", state)
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()

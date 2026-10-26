"""Verify actual continuity and keep its costs separate from learning claims."""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re


NAME = "native_stage_continuation_probe_v1_20261006"
GOAL = "put the white mug on the left plate and put the yellow and white mug on the right plate"
SESSION = "net_t4_i26"


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp(value):
    # Docker writes nanoseconds; the host's older Python accepts at most six
    # fractional digits. Keep the explicit timezone and microsecond accuracy.
    normalized = value.replace("Z", "+00:00")
    normalized = re.sub(r"(\.\d{6})\d+(?=[+-]\d\d:\d\d$)", r"\1", normalized)
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("Allocation timestamps must include a timezone")
    return parsed.timestamp()


def analyze(project):
    run = project/"runs"/NAME
    parent = read(run/"STATUS.json")
    if parent["status"] != "COMPLETED":
        raise ValueError("The real paired execution did not complete")
    for relative, digest in read(run/"SOURCE_BINDINGS.json").items():
        if sha(project/relative) != digest:
            raise ValueError("The registered original evidence changed")
    arms = {}
    roots = {}
    inputs_by_arm = {}
    for arm in ("N", "F"):
        group = run/arm
        stream = group/"stream"
        status = read(group/"STATUS.json")
        report = read(stream/"RESULT.json")
        if status["status"] != "COMPLETED" or report["status"] != "COMPLETED" or len(report["cases"]) != 1:
            raise ValueError("Actual full-OS worker incomplete")
        row = report["cases"][0]
        if row["chain_status"] != "PASSED" or not row["terminal_report"]["read_coverage"]["complete"]:
            raise ValueError("The original full chain or necessary report did not complete")
        if (row["case"]["id"] != SESSION or row["case"]["task_description"] != GOAL
                or row["case"]["feedback_window_mode"] != "fixed10"):
            raise ValueError("Root goal or action configuration changed")
        trace = lines(stream/"stage_trace"/SESSION/"TRACE.jsonl")
        roots[arm] = trace
        episode = read(stream/SESSION/"workspace/artifacts/runtime"/SESSION/"episode.json")
        if episode["num_steps"] != trace[-1]["step"]:
            raise ValueError("Native episode and real trace disagree")
        cost = lines(stream/"OS_COST_EVENTS.jsonl")
        inputs = [event for event in cost if event.get("event") == "actual_policy_input"]
        inputs_by_arm[arm] = inputs
        if any(event["prompt"] != GOAL for event in inputs):
            raise ValueError("Policy did not keep the complete goal")
        target_events = lines(stream/"TARGET_COST_EVENTS.jsonl")
        receipt = read(stream/"NATIVE_STAGE_CONTINUATION_RECEIPT.json")
        expected_types = [] if arm == "N" else ["native_stage_pause", "native_stage_resume"]
        if ([event["event"] for event in receipt["events"]] != expected_types
                or receipt["enabled"] != (arm == "F") or not receipt["native_run_case_completed"]
                or receipt["original_goal"] != GOAL or receipt["added_model_calls"] != 0
                or receipt["added_memory_triggers"] != 0):
            raise ValueError("Actual pause/resume differs from the registration")
        if arm == "F":
            pause, resume = receipt["events"]
            if (pause["request_id"] != resume["request_id"] or pause["physical_steps"] != 120
                    or resume["physical_steps"] != 120 or pause["submitted_chunks"] != 12
                    or pause["observation_id"] != "libero_obs_120"
                    or resume["action"] != "continue"
                    or receipt["actual_submitted_chunks"] != len(inputs)):
                raise ValueError("The boundary did not resume the original episode at the same step")
            root_continuation = episode["metadata"]["native_stage_continuation"]
            if root_continuation["visited_steps"] != [120] or root_continuation["executable_alternatives_bound"]:
                raise ValueError("Root episode metadata does not bind the continuation probe")
        if sum(item.get("method") == "reset" for item in trace) != 1:
            raise ValueError("The actual root trace reset count changed")
        stop = read(stream/"STOP_RECEIPT.json")
        if len(stop) != 4 or any(item["running"] for item in stop):
            raise ValueError("The four exactly owned services were not stopped")
        service_calls = [read(path) for path in sorted((stream/"service_calls").glob("call_*/RECEIPT.json"))]
        if any(item["status"] != "OK" for item in service_calls):
            raise ValueError("Original model service protocol error")
        spans = defaultdict(float)
        for event in cost:
            if event.get("event") == "end":
                spans[event["name"]] += event["duration_seconds"]
        allocation = {item["role"]: stamp(item["finished"])-stamp(item["started"])
                      for item in stop if item["role"] in ("qwen", "pi05")}
        arms[arm] = dict(native_root_success=episode["success"],
            native_report_success=row["environment_success"],
            semantic_verdicts=[item.get("verdict") for item in row["outcomes"]],
            chain_status=row["chain_status"], physical_steps=episode["num_steps"],
            vla_calls=len(inputs), qwen_calls=len(service_calls), decision_model_calls=0,
            root_trace_resets=1, full_environment_attempts=len(row["outcomes"]),
            target_environment_resets=sum(event.get("event") == "end" and event.get("name") == "environment_reset"
                                          for event in target_events),
            action_submissions=sum(event.get("event") == "end" and event.get("name") == "environment_execution"
                                   for event in target_events),
            startup_seconds=status["startup_seconds"], controller_seconds=status["controller_seconds"],
            worker_full_seconds=status["full_wall_seconds"],
            two_card_reservation_seconds=2*status["full_wall_seconds"],
            model_allocation_card_seconds=sum(allocation.values()), model_allocation_seconds=allocation,
            nested_os_spans=dict(spans), experience_admission=row["experience_admission"],
            continuation_receipt=receipt, qwen_generation_seconds=sum(r["generation_seconds"] for r in service_calls),
            qwen_prompt_tokens=sum(r["prompt_tokens"] for r in service_calls),
            qwen_completion_tokens=sum(r["completion_tokens"] for r in service_calls),
            stopped_container_ids=[item["container"] for item in stop])
    if len(roots["N"]) != len(roots["F"]) or len(inputs_by_arm["N"]) != len(inputs_by_arm["F"]):
        raise ValueError("A pause changed the number of root boundaries or policy requests")
    for left, right in zip(roots["N"], roots["F"]):
        for key in ("session_id", "step", "observation_id", "method", "actions", "images", "robot_state"):
            if left.get(key) != right.get(key):
                raise ValueError("Real continuity trace differs: "+key)
    for left, right in zip(inputs_by_arm["N"], inputs_by_arm["F"]):
        for key in ("prompt", "array_bindings", "robot_state"):
            if left.get(key) != right.get(key):
                raise ValueError("Actual policy request differs: "+key)
    return dict(status="PASSED_REAL_CONTINUITY_ONLY", arms=arms,
        original_root_goal=GOAL, actual_all_root_rgb_state_actions_equal=True,
        actual_all_policy_requests_equal=True, root_pause_step=120,
        parallel_parent_seconds=parent["seconds"],
        experiment_reservation_card_seconds=sum(value["two_card_reservation_seconds"] for value in arms.values()),
        experiment_model_allocation_card_seconds=sum(value["model_allocation_card_seconds"] for value in arms.values()),
        learning_benefit_established=False, new_capability_accepted=False, default_changed=False,
        full_agent_evaluation_isolation_newly_verified=False,
        scope="One known development pair. Actual native Agent/Skills/feedback retained with common public Session construction. A CONTINUE-only entry check, not learning or a speedup experiment.",
        accounting="Full worker includes startup, execution, native verification, natural feedback and cleanup. Nested spans are not additional charges. Parallel parent elapsed is experiment throughput.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.project)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/"SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    run = args.project/"runs"/NAME
    files = {str(p.relative_to(args.project)): sha(p) for p in run.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts}
    (args.out/"SOURCE_MANIFEST.json").write_text(json.dumps(files, indent=2)+"\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "arms"}))


if __name__ == "__main__":
    main()

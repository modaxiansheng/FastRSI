"""Report real operation consequences; fewer calls alone is not a benefit."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import struct

from summarize_native_stage_continuation_probe_v1 import read, lines, sha, stamp


NAME="native_relative_phaselet_value_probe_v1_20261006"
SESSION="net_t4_i28"


def analyze(project):
    run=project/"runs"/NAME
    parent=read(run/"STATUS.json")
    if parent["status"]!="COMPLETED":raise ValueError("Actual paired flow is incomplete")
    candidate=read(run/"PHASELET.json")
    for rel,digest in candidate["source_bindings"].items():
        if sha(project/rel)!=digest:raise ValueError("Source evidence changed: "+rel)
    arms={};traces={};policies={};outputs={}
    for arm in ("N","F"):
        group=run/arm;stream=group/"stream"
        status=read(group/"STATUS.json");report=read(stream/"RESULT.json")
        if status["status"]!="COMPLETED" or report["status"]!="COMPLETED":
            raise ValueError("Full native worker did not complete")
        row=report["cases"][0]
        if row["chain_status"]!="PASSED" or not row["terminal_report"]["read_coverage"]["complete"]:
            raise ValueError("Native chain/report missing")
        root=read(stream/SESSION/"workspace/artifacts/runtime"/SESSION/"episode.json")
        if "native_stage_continuation" in root["metadata"]:
            raise ValueError("Debug audit still enters original native result")
        cost=lines(stream/"OS_COST_EVENTS.jsonl")
        target=lines(stream/"TARGET_COST_EVENTS.jsonl")
        inputs=[x for x in cost if x.get("event")=="actual_policy_input"]
        if any(x["prompt"]!=candidate["goal"] for x in inputs):
            raise ValueError("Original full goal changed")
        policies[arm]=inputs
        receipt=read(stream/"NATIVE_RELATIVE_PHASELET_RECEIPT.json")
        if (receipt["original_case_completed"] is not True or receipt["learned_selection"]
                or receipt["added_model_calls"] or receipt["added_memory_triggers"]):
            raise ValueError("Unexpected extra process or learning")
        audit=receipt["audit"];events=audit["events"]
        pauses=[x for x in events if x["event"]=="native_stage_pause"]
        resumes=[x for x in events if x["event"]=="native_stage_resume"]
        outputs[arm]=[x for x in events if x["event"]=="native_relative_phaselet_output"]
        expected=0 if arm=="N" else 1
        if (len(pauses)!=1 or len(resumes)!=1 or pauses[0]["physical_steps"]!=60
                or pauses[0]["request_id"]!=resumes[0]["request_id"]
                or len(outputs[arm])!=expected or audit["skipped_policy_calls"]!=expected):
            raise ValueError("Registered physical intervention not executed as specified")
        trace=lines(stream/"stage_trace"/SESSION/"TRACE.jsonl");traces[arm]=trace
        if sum(r["method"]=="reset" for r in trace)!=1:
            raise ValueError("The root reset count changed")
        stop=read(stream/"STOP_RECEIPT.json")
        if len(stop)!=4 or any(x["running"] for x in stop):
            raise ValueError("Exact owned service cleanup incomplete")
        calls=[read(f) for f in (stream/"service_calls").glob("*/RECEIPT.json")]
        if any(x["status"]!="OK" for x in calls):raise ValueError("Model service protocol failed")
        spans=defaultdict(float)
        for event in cost:
            if event.get("event")=="end":spans[event["name"]]+=event["duration_seconds"]
        allocations={x["role"]:stamp(x["finished"])-stamp(x["started"])
                     for x in stop if x["role"] in ("qwen","pi05")}
        arms[arm]=dict(native_primary_success=row["environment_success"],root_native_success=root["success"],
            semantic_verdicts=[x.get("verdict") for x in row["outcomes"]],chain_status=row["chain_status"],
            root_physical_steps=root["num_steps"],full_environment_attempts=len(row["outcomes"]),
            action_submissions=sum(x.get("event")=="end" and x.get("name")=="environment_execution" for x in target),
            vla_calls=len(inputs),qwen_calls=len(calls),decision_model_calls=0,
            phaselet_outputs=expected,root_reset_count=1,worker_full_seconds=status["full_wall_seconds"],
            startup_seconds=status["startup_seconds"],controller_seconds=status["controller_seconds"],
            model_allocation_card_seconds=sum(allocations.values()),model_allocation_seconds=allocations,
            nested_os_spans=dict(spans),experience_admission=row["experience_admission"],
            qwen_generation_seconds=sum(x["generation_seconds"] for x in calls),
            qwen_prompt_tokens=sum(x["prompt_tokens"] for x in calls),
            qwen_completion_tokens=sum(x["completion_tokens"] for x in calls),
            stop_container_ids=[x["container"] for x in stop],receipt=receipt)
    prefixes={a:[r for r in traces[a] if r["step"]<=60] for a in ("N","F")}
    if len(prefixes["N"])!=7 or len(prefixes["F"])!=7:raise ValueError("Prefix incomplete")
    for l,r in zip(prefixes["N"],prefixes["F"]):
        for k in ("step","observation_id","method","images","actions"):
            if l.get(k)!=r.get(k):raise ValueError("Intervention prefix changed: "+k)
    for l,r in zip(policies["N"][:6],policies["F"][:6]):
        for k in ("prompt","array_bindings","robot_state"):
            if l[k]!=r[k]:raise ValueError("Native initial policy prefix changed")
    output=outputs["F"][0]
    if (output["current_arrays"]!=policies["N"][6]["array_bindings"]
            or output["current_public_robot8"]!=policies["N"][6]["robot_state"]):
        raise ValueError("Current lawful input changed before intervention")
    applied=[r for r in traces["F"] if r["method"]=="action_chunk" and r["step"]==70]
    if len(applied)!=1 or applied[0]["actions"]!=candidate["actions"]:
        raise ValueError("Actual target did not execute original relative candidate")
    action_sha=hashlib.sha256(struct.pack("<70f",*(v for row in applied[0]["actions"] for v in row))).hexdigest()
    if action_sha!=candidate["actions_float32_sha256"]:raise ValueError("Action digest changed")
    n=arms["N"];f=arms["F"]
    deltas={key:f[key]-n[key] for key in ("root_physical_steps","action_submissions","vla_calls",
        "qwen_calls","worker_full_seconds","controller_seconds","model_allocation_card_seconds")}
    return dict(status="REAL_OPERATION_VALUE_PROBE_COMPLETED",arms=arms,
        original_goal=candidate["goal"],same_real_prefix_through60=True,exact_relative_source_actions_executed=True,
        original_native_metadata_preserved=True,incremental_F_minus_N=deltas,
        primary_success_kept_in_this_pair=f["native_primary_success"]==n["native_primary_success"]==True,
        both_raw_native_root_success=f["root_native_success"]==n["root_native_success"]==True,
        actual_fewer_expensive_calls=(f["vla_calls"]+f["qwen_calls"]<n["vla_calls"]+n["qwen_calls"]),
        no_more_root_physical_steps=f["root_physical_steps"]<=n["root_physical_steps"],
        source_cost_reference=candidate["source_cost_reference"],source_cost_free=False,
        experiment_parallel_parent_seconds=parent["seconds"],
        experiment_model_allocation_card_seconds=sum(x["model_allocation_card_seconds"] for x in arms.values()),
        learning_benefit_established=False,eligibility_learned=False,candidate_accepted=False,
        evaluation_isolation_newly_verified=False,default_changed=False,
        scope="One already-used developer context, forced operation at60. Tests causal operation value; no learned eligibility, independent generalization or net learning claim.",
        accounting="Worker includes startup, actual execution, required verification, natural feedback and cleanup. Nested spans overlap; parallel parent elapsed is throughput.")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--project",type=Path,required=True)
    parser.add_argument("--run-name",default=NAME)
    parser.add_argument("--out",type=Path,required=True);args=parser.parse_args()
    if not args.run_name.startswith("native_relative_phaselet_value_probe_") or "/" in args.run_name:
        raise ValueError("Unbound operation-value run")
    NAME=args.run_name
    summary=analyze(args.project);args.out.mkdir(parents=True,exist_ok=False)
    (args.out/"SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
    run=args.project/"runs"/NAME
    manifest={str(p.relative_to(args.project)):sha(p) for p in run.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    (args.out/"SOURCE_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k!="arms"}))

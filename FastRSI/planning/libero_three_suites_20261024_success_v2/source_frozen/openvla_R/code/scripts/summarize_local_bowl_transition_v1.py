"""Recompute the actual complete-Agent operation pair, including failed arms."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path):
    return [] if not path.exists() else [json.loads(v) for v in path.read_text().splitlines() if v.strip()]


def spans(path):
    counts, seconds = Counter(), defaultdict(float)
    for row in lines(path):
        if row.get("event") == "end":
            counts[row["name"]] += 1
            seconds[row["name"]] += row["duration_seconds"]
    return dict(completed_counts=dict(counts),inclusive_seconds_do_not_sum_nested=dict(seconds))


def summarize(campaign):
    status=json.loads((campaign/"STATUS.json").read_text())
    reg=json.loads((campaign/"REGISTRATION.json").read_text())
    stream=campaign/"stream"
    controls={}
    for path in (stream/"live_controls").glob("*/CONTROLLER_TARGETS.jsonl"):
        controls[path.parent.name]=lines(path)
    result_path=stream/"RESULT.json"
    raw=json.loads(result_path.read_text()) if result_path.exists() else {}
    cases=raw.get("cases",[])
    local_path=stream/reg["root_session"]/"LOCAL_TRANSITION_RECEIPT.json"
    local=json.loads(local_path.read_text()) if local_path.exists() else {}
    target_local=lines(stream/"LOCAL_TRANSITION_EXECUTION.jsonl")
    local_count=sum(len(r["actual_actions"]) for r in target_local)
    if local:
        assert local_count==local["local_actions"]
        assert len(target_local)<=local["local_requests"]
        if status["status"]=="COMPLETED":
            assert len(target_local)==local["local_requests"]
    for sid, rows in controls.items():
        assert [r["step_after"] for r in rows]==list(range(1,len(rows)+1))
        trace=lines(stream/"stage_trace"/sid/"TRACE.jsonl")
        actual=[];previous=0
        for r in trace:
            count=r["step"]-previous
            actual.extend(r.get("actions",[])[:count]);previous=r["step"]
            for name,sha in r["images"].items():
                assert digest(stream/"stage_trace"/sid/name)==sha
        assert actual==[r["actual_action"] for r in rows]
    all_controls=[r for rows in controls.values() for r in rows]
    if target_local:
        assert sum(r["execution_origin"]=="goal_skill_action" for r in all_controls)==local_count
    os_cost=spans(stream/"OS_COST_EVENTS.jsonl")
    target_cost=spans(stream/"TARGET_COST_EVENTS.jsonl")
    receipt=dict(status=status["status"],root_session=reg["root_session"],
        environment_success=cases[0].get("environment_success") if cases else None,
        verification_status=cases[0].get("verification_status") if cases else None,
        chain_status=cases[0].get("chain_status") if cases else None,
        semantic_verdicts=[r.get("verdict") for c in cases for r in c.get("outcomes",[])],
        sessions=len(controls),actual_actions=len(all_controls),local_actions=local_count,
        local_requests=len(target_local),root_actual_vla_calls=local.get("root_actual_vla_calls"),
        issued_local_requests=local.get("local_requests",0),
        uncompleted_local_requests=local.get("local_requests",0)-len(target_local),
        total_actual_vla_calls=os_cost["completed_counts"].get("policy_client",0),
        total_control_requests=target_cost["completed_counts"].get("environment_execution",0),
        Qwen_requests=len(list((stream/"service_calls").glob("*/REQUEST.json"))),
        full_wall_seconds=status.get("full_wall_seconds"),startup_seconds=status.get("model_and_target_startup_seconds"),
        os_cost=os_cost,target_cost=target_cost,planner_seconds=local.get("planner_seconds"),
        local_motion_seconds=sum(r["seconds"] for r in target_local),
        local_entry=[r for r in local.get("events",[]) if r["stage"]=="enter"],
        local_handoff=[r for r in local.get("events",[]) if r["stage"]=="handoff"],
        local_unknown=[r for r in local.get("events",[]) if r["stage"]=="no_local_action"],
        eligibility_gate=local.get("eligibility_gate"),gate_seconds=local.get("gate_seconds"),
        gate_decisions=[r for r in local.get("events",[]) if r["stage"]=="eligibility_gate"],
        default_unchanged=status.get("default_entry_unchanged"),
        source_result_sha256=digest(result_path) if result_path.exists() else None,
        feedback_motion_fact="Per-step commands from returned public position; no object true pose read",
        learned=False,heldout=False,physical_release_certified=False)
    return receipt,controls


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native",type=Path,required=True)
    parser.add_argument("--transition",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    a,ac=summarize(args.native);b,bc=summarize(args.transition)
    first=b["local_entry"][0]["step"] if b["local_entry"] else None
    prefix=None
    if first is not None:
        ar=ac[a["root_session"]];br=bc[b["root_session"]]
        prefix=dict(actions=first,actual_actions_exact=[r["actual_action"] for r in ar[:first]]==[r["actual_action"] for r in br[:first]],
            controller_targets_exact=[r["target_pos"] for r in ar[:first]]==[r["target_pos"] for r in br[:first]])
    roots=[args.native,args.transition]
    restore=[lines(r/"stream/NATIVE_SAMPLER_RESTORE.jsonl") for r in roots]
    initial_frames=[r/"stream/stage_trace"/s["root_session"]/"0000_agentview_image.png" for r,s in zip(roots,(a,b))]
    result=dict(version="local_bowl_transition_operation_pair_v1",arms=dict(native=a,transition=b),
        matched_prefix=prefix,initial_front_image_exact=digest(initial_frames[0])==digest(initial_frames[1]),
        restore_receipts=restore,scope="Single exposed developmental scene; manual skill, not learning/net return/generalization",
        all_failures_included=True,costs_are_nested_not_additive=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:{j:v[j] for j in ("environment_success","chain_status","actual_actions","total_actual_vla_calls","local_actions","local_requests","total_control_requests","Qwen_requests","full_wall_seconds")} for k,v in result["arms"].items()},ensure_ascii=False))


if __name__=="__main__":main()

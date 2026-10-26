"""Original Agent proposal -> external validation -> new Agent update, no launch.

propose may register a typed draft, leave unchanged, or actually write a bounded
validation request with original native file tools. It then stops. update accepts
only complete hash-bound actual consumer results, starts a new native Session,
and checks read-only inheritance/selection after its actual component decision.
The provided operation is human-defined; no phase here performs robot execution,
distillation, admission, or a net-benefit measurement.
"""
import argparse
import asyncio
import copy
import json
import os
from pathlib import Path
import time
import traceback

import probe_native_goal_component_update_v1 as base
from fastrsi import native_goal_component_v1 as component
from fastrsi import native_validation_request_v1 as protocol

VERSION = "native_goal_learning_cycle_v1"
STATE_REL = "skills/goal-component-feedback/CURRENT_COMPONENT_STATE.json"
SCHEMA_REL = "skills/goal-component-feedback/VALIDATION_REQUEST_PROTOCOL.json"
ACTUAL_FEEDBACK_REL = "skills/goal-component-feedback/ACTUAL_VALIDATION_FEEDBACK.json"
REQUEST_REL = "reports/VALIDATION_REQUEST.json"


def current_component_state(workspace, spec, backend):
    """Deterministic native schema/scheduler facts, never a model interpretation."""
    resolved = backend.component_selection_anchor(Path(workspace), spec)
    skill = resolved["skill"]
    declared = component.extract_component(skill)
    return dict(version=protocol.STATE_VERSION, origin="NATIVE_SCHEMA_AND_SESSION_SCHEDULER",
        scope="Actual registry snapshot at the beginning of this specific Agent Session",
        goal=spec["task_description"], target_ref=spec["target_ref"], skillruntime_ref=spec["skillruntime_ref"],
        registry_sha256={name: base.sha((Path(workspace) / name).read_bytes()) for name in ("SKILLRUNTIME.md", "TARGETS.md")},
        selected_skill=copy.deepcopy(skill), action_chunk=copy.deepcopy(skill["output_contract"]["action"]["chunk"]),
        native_goal_component=declared, component_is_registered=declared is not None,
        execution=copy.deepcopy(spec["execution"]),
        interpretation="Consumer schema describes a possible declaration; only this parsed registry says whether one is installed")


def stage_cycle_state(workspace, spec, backend, raw, phase, actual_feedback=None):
    manifest, protected = base.stage_inputs(workspace, raw)
    state = current_component_state(workspace, spec, backend)
    base.save(Path(workspace) / STATE_REL, state)
    state_sha = base.sha((Path(workspace) / STATE_REL).read_bytes())
    semantics = json.loads((Path(workspace) / base.SEMANTICS_REL).read_text())
    declaration_schema = semantics.get("declaration_schema")
    if not isinstance(declaration_schema, dict):
        raise ValueError("Existing exact typed declaration schema is required")
    schema = protocol.request_schema(current_component_state_sha256=state_sha, declaration_schema=declaration_schema)
    schema["phase"] = phase
    if phase != "propose":
        schema["legal_decisions"] = ["register", "noop"]
        schema["status"] = "ACTUAL_EXTERNAL_FEEDBACK_AVAILABLE_NO_MORE_REQUESTS_THIS_CYCLE"
    base.save(Path(workspace) / SCHEMA_REL, schema)
    for rel in (STATE_REL, SCHEMA_REL):
        protected[rel] = (Path(workspace) / rel).read_bytes()
    required = (*base.REQUIRED_READS, base.MANIFEST_REL, STATE_REL, SCHEMA_REL)
    if actual_feedback is not None:
        base.save(Path(workspace) / ACTUAL_FEEDBACK_REL, actual_feedback)
        protected[ACTUAL_FEEDBACK_REL] = (Path(workspace) / ACTUAL_FEEDBACK_REL).read_bytes()
        required += (ACTUAL_FEEDBACK_REL,)
    return manifest, protected, state, state_sha, required


def install_cycle_tools(agent, workspace, log, tracker, protected, validate_draft, backend,
                        *, phase, baseline_declaration, model_sha, state_sha):
    original = agent.tools.execute
    root = Path(workspace)

    async def execute(name, params):
        rel = base.relative(root, params.get("path", ""))
        if name == "verify_session":
            return "Error: no robot Session was executed in this proposal/update phase"
        if name in base.WRITE_TOOLS:
            allowed = {"SKILLRUNTIME.md", base.PROPOSAL_REL}
            if phase == "propose":
                allowed.add(REQUEST_REL)
            if rel not in allowed or rel == REQUEST_REL and name != "write_file":
                return "Error: only typed registry/note and propose-phase write_file validation request are allowed"
            try:
                tracker.require()
            except ValueError as exc:
                return "Error: " + str(exc)
        registry_before = (root / "SKILLRUNTIME.md").read_bytes()
        request_before = (root / REQUEST_REL).read_bytes() if (root / REQUEST_REL).exists() else None
        seen_before, eof_before = copy.deepcopy(tracker.seen), set(tracker.eof)
        result = await original(name, params)
        if name == "read_files" and isinstance(result, str) and len(result) > 16000:
            tracker.seen, tracker.eof = seen_before, eof_before
        else:
            tracker.observe(name, params, result)
        if name in base.WRITE_TOOLS and rel in ("SKILLRUNTIME.md", REQUEST_REL) and not result.startswith("Error"):
            try:
                _, declaration = validate_draft()
                request_path = root / REQUEST_REL
                if request_path.exists():
                    if declaration != baseline_declaration:
                        raise ValueError("Choose draft registration OR external validation request, not self-approval plus request")
                    protocol.validate_request(json.loads(request_path.read_text()), workspace=root,
                        expected_model_config_sha256=model_sha, current_component_state_sha256=state_sha)
                for path, blob in protected.items():
                    if (root / path).read_bytes() != blob:
                        raise ValueError("Fixed facts/source/current native state changed: " + path)
            except Exception as exc:
                (root / "SKILLRUNTIME.md").write_bytes(registry_before)
                if request_before is None:
                    if (root / REQUEST_REL).exists():
                        (root / REQUEST_REL).unlink()
                else:
                    (root / REQUEST_REL).write_bytes(request_before)
                backend.record(log, event="invalid_cycle_write_rolled_back", name=name, error=str(exc))
                return "Error: invalid cycle write rolled back: " + str(exc)
            backend.record(log, event="actual_cycle_file_commit", component_path=rel,
                declaration_present=declaration is not None, validation_requested=(root / REQUEST_REL).exists(),
                empirical_admission=False, robot_episodes=0)
        return result

    agent.tools.execute = execute
    return original


def phase_prompt(phase):
    common = (
        "Use the original native file tools. Completely read DEVELOPMENT_FEEDBACK, CONSUMER_SEMANTICS, "
        "CURRENT_COMPONENT_STATE, VALIDATION_REQUEST_PROTOCOL and INPUT_MANIFEST under skills/goal-component-feedback/, "
        "and TARGETS.md/SKILLRUNTIME.md. CURRENT_COMPONENT_STATE is deterministically parsed from the actual native "
        "registry, not a Qwen claim. A declaration_schema is merely a possible interface; it does not prove that "
        "native_goal_component is currently installed. Paginate read_file limit=100 until complete. Each tool "
        "message must be at most16000 characters; read_files is allowed only if its entire aggregate fits. "
        "Keep the exact complete task goal, pi05/runtime/target/observation/config contracts. Do not write Python, "
        "new skills, source geometry, RGB, calibration, facts, Session queues or evaluation outcomes. "
        "The dual-anchor primitive was human-defined; adopting it is not discovering or validating it. "
        "Model-config SHA is an external Qwen service contract, not source pi05 weight provenance. "
        "Only output_contract.action.chunk.native_goal_component in the existing pi05 SkillRuntime may change. "
        "A note may be saved to skills/goal-component-feedback/PROPOSAL.md. No-op is fully legal. "
    )
    if phase == "propose":
        return common + (
            "Choose from registering an unvalidated typed draft, leaving unchanged, or asking for real bounded "
            "development validation. If requesting validation, actually use write_file to save "
            "reports/VALIDATION_REQUEST.json matching the request protocol, with your own reason and one or two "
            "allowed development seeds. Both arms use the same draws/400cap and necessary validation; four "
            "evaluation draws are excluded. Requesting validation does not execute anything or approve the candidate. "
            "Choose registration or request, not both. Do not invent expected branch outcomes or request merely "
            "to satisfy this prompt. Report the actual saved decision; this phase ends awaiting an external consumer."
        )
    return common + (
        "Completely read ACTUAL_VALIDATION_FEEDBACK.json in that same facts directory. It was extracted from "
        "completed hash-bound actual consumer results, including failures and all reported costs. This is a new "
        "native Agent Session after external development execution. Decide whether those facts justify changing "
        "the typed executable component or leaving it unchanged. Unknown results remain unknown; conflicting "
        "semantic text is quarantined and cannot become a success label. Do not fabricate unexecuted branches, "
        "claim generalization/net saving, or request more experiments in this cycle. Save any actual update with "
        "the original file tools. The following read-only Session will independently discover/select the registry."
    )


async def readonly_selection(output, campaign, common, source_workspace, protected, backend, provider_factory):
    folder = output / "new_session"; folder.mkdir()
    case = dict(common, id="goal_cycle_readonly_selection", mode="evaluation", planning_only=True,
                require_discovery_reads=True, native_experience_maintenance=False)
    w, requested, previous = backend.setup_workspace(folder, case, source_workspace)
    if (w / "SKILLRUNTIME.md").read_bytes() != (source_workspace / "SKILLRUNTIME.md").read_bytes():
        raise ValueError("Read-only Session did not inherit exact executable registry")
    # The new Session's state is a new deterministic fact, not the old pre-update snapshot.
    base.save(w / STATE_REL, current_component_state(w, requested, backend))
    payload = json.loads((w / "REQUEST.json").read_text())
    payload["current_component_state_read_path"] = STATE_REL
    payload["current_component_state_sha256"] = base.sha((w / STATE_REL).read_bytes())
    base.save(w / "REQUEST.json", payload)
    log = folder / "AGENT_EVENTS.jsonl"
    fixed_before = backend.hashes(w)
    agent = backend.bind_agent(w, provider_factory(folder, "selection"), log,
        evaluation=True, component_updates=True, max_iterations=12, read_files=True)
    reads = base.CompleteReads(w, ("SKILLRUNTIME.md", "TARGETS.md", "REQUEST.json", STATE_REL))
    original = base.install_read_tracker(agent, reads)
    tracked = agent.tools.execute
    async def checked_readonly_tools(name, params):
        if name in base.WRITE_TOOLS and base.relative(w, params.get("path", "")) == "SESSIONS.md":
            try:
                reads.require()
            except ValueError as exc:
                return "Error: " + str(exc)
        return await tracked(name, params)
    agent.tools.execute = checked_readonly_tools
    try:
        with base.scoped_selection_validation(backend, w, case):
            reply, planned = await backend.plan_session(agent, case, w, requested, previous, log)
        reads.require()
        base.save(folder / "REPLY.json", dict(text=reply))
        declaration = component.extract_component(base.selected_skill(w, planned, backend))
        anchor = backend.component_selection_anchor(w, requested)
        backend.validate_component_selection(w, planned, anchor, goal_component_case=case)
        receipt = dict(planned=planned, selected_declaration=declaration, complete_reads=reads.complete(),
                       registry_bytes_inherited=True, task_executed=False, workspace=str(w))
    finally:
        agent.tools.execute = original
        await backend.finish_agent(agent, log)
    if backend.hashes(w) != fixed_before or not (w / ".FASTRSI_EVALUATION_CLOSED.json").is_file():
        raise ValueError("Read-only new Session/end/background modified fixed components")
    for rel, blob in protected.items():
        if rel != STATE_REL and (w / rel).read_bytes() != blob:
            raise ValueError("Read-only inheritance changed fixed facts: " + rel)
    base.save(folder / "SELECTION.json", receipt)
    return receipt


async def run_cycle(campaign, output, *, phase, initial_workspace=None, proposal_output=None,
                    feedback_path=None, feedback_sha256=None, backend=None, provider_factory=None,
                    max_tokens=3072, max_iterations=12):
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    if phase not in ("propose", "update") or output.exists() and any(output.iterdir()):
        raise ValueError("Use propose/update and a fresh empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = dict(version=VERSION, phase=phase, status="RUNNING", robot_episodes=0,
                  policy_inference_calls=0, empirical_admission=False, sealed=False,
                  distillation_performed=False, learning_benefit=False,
                  provider_mode="INJECTED_CPU_PROTOCOL_ONLY" if provider_factory else "ORIGINAL_MEASURED_QWEN")
    try:
        if os.environ.get("FASTRSI_EXPERIENCE_ADMISSION") != "legacy_raw":
            raise ValueError("Explicit isolated draft inheritance required; no seal or empirical admission")
        raw = base.campaign_inputs(campaign)
        actual_feedback = None
        if phase == "update":
            if proposal_output is None or feedback_path is None or feedback_sha256 is None or initial_workspace is not None:
                raise ValueError("Update requires prior request output and exact external feedback SHA, not an arbitrary workspace")
            previous = Path(proposal_output).resolve()
            prior = json.loads((previous / "RESULT.json").read_text())
            if prior.get("status") != "AWAITING_EXTERNAL_VALIDATION" or prior.get("phase") != "propose":
                raise ValueError("Update cannot bypass the actual model-written request phase")
            request_path = previous / "VALIDATED_VALIDATION_REQUEST.json"
            request_raw = request_path.read_bytes()
            if base.sha(request_raw) != prior["request"]["export_sha256"]:
                raise ValueError("Prior exported request changed")
            initial_workspace = previous / "agent_session/workspace"
            original_request = (initial_workspace / REQUEST_REL).read_bytes()
            if (base.sha(original_request) != prior["request"]["original_sha256"]
                    or json.loads(original_request) != json.loads(request_raw)):
                raise ValueError("Original model-written validation request changed")
            protocol.validate_request(json.loads(request_raw), workspace=initial_workspace,
                expected_model_config_sha256=base.sha(raw["MODEL_CONFIG.json"]),
                current_component_state_sha256=base.sha((initial_workspace / STATE_REL).read_bytes()))
            feedback_raw = Path(feedback_path).read_bytes()
            if base.sha(feedback_raw) != feedback_sha256:
                raise ValueError("External feedback input changed")
            actual_feedback = protocol.validate_feedback(json.loads(feedback_raw), base=Path(feedback_path).parent,
                request=json.loads(request_raw), request_sha256=base.sha(request_raw))
            if base.sha((initial_workspace / "SKILLRUNTIME.md").read_bytes()) != prior["decision"]["registry_sha256"]:
                raise ValueError("Prior model-requested workspace changed")
            result["external_feedback"] = dict(path=str(feedback_path), sha256=feedback_sha256,
                                               request_sha256=base.sha(request_raw), actual_branches=len(actual_feedback["executions"]))
        elif proposal_output is not None or feedback_path is not None or feedback_sha256 is not None:
            raise ValueError("Propose must not receive future consumer/evaluation feedback")
        if initial_workspace is not None and (Path(initial_workspace) / ".FASTRSI_EVALUATION_CLOSED.json").exists():
            raise ValueError("Never inherit a finished evaluation as development experience")
        backend = backend or base.load_backend(output, campaign)
        provider_factory = provider_factory or (lambda folder, role: backend.measured_provider(folder, role, max_tokens))
        common = dict(suite="libero_90", task_id=64, init_state_id=29, max_steps=400,
            development_suite_extension="libero90_stack_flow_v1", task_description=component.SOURCE_GOAL,
            native_registry_lifecycle="incremental_v1", native_component_updates="registered_v1",
            registered_goal_component=True, model_config=str(campaign / "MODEL_CONFIG.json"),
            model_config_sha256=base.sha(raw["MODEL_CONFIG.json"]))
        folder = output / "agent_session"; folder.mkdir()
        case = dict(common, id="goal_cycle_" + phase, mode="development")
        w, spec, _ = backend.setup_workspace(folder, case, Path(initial_workspace) if initial_workspace else None)
        if (w / REQUEST_REL).exists():
            old = (w / REQUEST_REL).read_bytes()
            archive = w / "reports" / ("PRIOR_VALIDATION_REQUEST_" + base.sha(old) + ".json")
            archive.write_bytes(old)
            (w / REQUEST_REL).unlink()  # Only new branch; original model request remains immutable.
        _, protected, state, state_sha, required = stage_cycle_state(w, spec, backend, raw, phase, actual_feedback)
        anchor = backend.component_selection_anchor(w, spec)
        baseline = backend.read_yaml_block(w / "SKILLRUNTIME.md")
        baseline_declaration = component.extract_component(base.selected_skill(w, spec, backend))
        tracker = base.CompleteReads(w, required)
        log = folder / "AGENT_EVENTS.jsonl"
        agent = backend.bind_agent(w, provider_factory(folder, phase), log,
            component_updates=True, max_iterations=max_iterations, read_files=True)
        validate = lambda: base.validate_draft(w, spec, anchor, case, baseline, backend)
        original = install_cycle_tools(agent, w, log, tracker, protected, validate, backend, phase=phase,
            baseline_declaration=baseline_declaration, model_sha=common["model_config_sha256"], state_sha=state_sha)
        try:
            reply = await agent.process_direct(phase_prompt(phase), session_key="cli:goal-cycle-" + phase)
            base.save(folder / "REPLY.json", dict(text=reply))
        finally:
            agent.tools.execute = original
            await backend.finish_agent(agent, log)
        tracker.require()
        compatibility, declaration = validate()
        for rel, blob in protected.items():
            if (w / rel).read_bytes() != blob:
                raise ValueError("Agent/finish changed immutable facts or source: " + rel)
        changed = declaration != baseline_declaration
        result["decision"] = dict(component_changed=changed, declaration=declaration,
            registry_sha256=base.sha((w / "SKILLRUNTIME.md").read_bytes()),
            current_component_state_sha256=state_sha, complete_reads=tracker.complete(),
            compatibility=compatibility, origin="Actual original Agent file-tool decision about human-defined executable interface")
        if (w / REQUEST_REL).exists():
            if phase != "propose" or changed:
                raise ValueError("Only unapproved propose-phase requests may await external execution")
            request = protocol.validate_request(json.loads((w / REQUEST_REL).read_text()), workspace=w,
                expected_model_config_sha256=common["model_config_sha256"], current_component_state_sha256=state_sha)
            base.save(output / "VALIDATED_VALIDATION_REQUEST.json", request)
            result["request"] = dict(original_path=str(w / REQUEST_REL), original_sha256=base.sha((w / REQUEST_REL).read_bytes()),
                export_sha256=base.sha((output / "VALIDATED_VALIDATION_REQUEST.json").read_bytes()),
                status="AWAITING_EXTERNAL_VALIDATION_NOT_ACCEPTED", max_real_episodes=2 * len(request["development_seeds"]))
            result["status"] = "AWAITING_EXTERNAL_VALIDATION"
        else:
            if phase == "update":
                result["new_session"] = await readonly_selection(output, campaign, common, w, protected, backend, provider_factory)
                result["evaluation_updates_blocked"] = True
            result["status"] = "COMPLETED_TYPED_DRAFT_CHANGE" if changed else "COMPLETED_LEGAL_NO_COMPONENT_CHANGE"
        for rel, blob in raw.items():
            if (campaign / rel).read_bytes() != blob:
                raise ValueError("Original campaign input changed")
        result["original_campaign_inputs_unchanged"] = True
    except Exception:
        result.update(status="FAILED", error=traceback.format_exc())
    result.update(elapsed_seconds=time.perf_counter() - started,
                  model_calls=len(list(output.glob("*/*_call_*/REQUEST.json"))))
    base.save(output / "RESULT.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("propose", "update"), required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-workspace", type=Path)
    parser.add_argument("--proposal-output", type=Path)
    parser.add_argument("--feedback", type=Path)
    parser.add_argument("--feedback-sha256")
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--allow-real-model-calls", action="store_true")
    args = parser.parse_args(argv)
    if not args.allow_real_model_calls or not 256 <= args.max_tokens <= 4096 or not 1 <= args.max_iterations <= 16:
        parser.error("Explicit real-call switch, 256..4096 tokens and 1..16 iterations required")
    result = asyncio.run(run_cycle(args.campaign, args.output, phase=args.phase,
        initial_workspace=args.initial_workspace, proposal_output=args.proposal_output,
        feedback_path=args.feedback, feedback_sha256=args.feedback_sha256,
        max_tokens=args.max_tokens, max_iterations=args.max_iterations))
    print(json.dumps(result, ensure_ascii=False))
    return 1 if result["status"] == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Original Agent typed-program proposal and read-only new-Session selection.

No policy, simulator, fixture inference, empirical admission, or seal is run.
The registered dual-anchor primitive is human-defined; model adoption is a
separate observable event and is not discovery, distillation, or net benefit.
"""
import argparse
import asyncio
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import time
import traceback

from fastrsi import native_goal_component_v1 as component

VERSION = "native_goal_component_update_probe_v1"
FEEDBACK_REL = "skills/goal-component-feedback/DEVELOPMENT_FEEDBACK.json"
SEMANTICS_REL = "skills/goal-component-feedback/CONSUMER_SEMANTICS.json"
MANIFEST_REL = "skills/goal-component-feedback/INPUT_MANIFEST.json"
PROPOSAL_REL = "skills/goal-component-feedback/PROPOSAL.md"
STATE_REL = "skills/goal-component-feedback/CURRENT_COMPONENT_STATE.json"
INPUT_REFS = {
    "SOURCE.json": "configs/native_goal_component/SOURCE.json",
    "SOURCE.png": "configs/native_goal_component/SOURCE.png",
    "CALIBRATION.json": "configs/native_goal_component/CALIBRATION.json",
}
REQUIRED_READS = (FEEDBACK_REL, SEMANTICS_REL, "TARGETS.md", "SKILLRUNTIME.md")
WRITE_TOOLS = {"write_file", "edit_file", "append_yaml_item", "patch_yaml_item"}


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False, default=str) + "\n")


def relative(workspace, value):
    path = Path(value)
    root = Path(workspace).resolve()
    path = path if path.is_absolute() else root / path
    try:
        return path.resolve().relative_to(root).as_posix()
    except (ValueError, OSError):
        return None


def finalize_semantics(value, model_sha):
    """Expand one contract placeholder, never an action/decision placeholder."""
    if isinstance(value, dict):
        return {k: finalize_semantics(v, model_sha) for k, v in value.items()}
    if isinstance(value, list):
        return [finalize_semantics(v, model_sha) for v in value]
    return model_sha if value == "$MODEL_CONFIG_SHA256" else value


def campaign_inputs(campaign):
    root = Path(campaign).resolve()
    paths = [root / "DEVELOPMENT_FEEDBACK.json", root / "CONSUMER_SEMANTICS.json",
             root / "MODEL_CONFIG.json"] + [root / "inputs" / n for n in INPUT_REFS]
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError("Missing or symlink campaign input: " + str(path))
    raw = {p.relative_to(root).as_posix(): p.read_bytes() for p in paths}
    feedback = json.loads(raw["DEVELOPMENT_FEEDBACK.json"])
    semantics = json.loads(raw["CONSUMER_SEMANTICS.json"])
    if not isinstance(feedback, dict) or feedback.get("goal") != component.SOURCE_GOAL:
        raise ValueError("Feedback must bind the exact complete task64 goal")
    if not isinstance(semantics, dict):
        raise ValueError("Consumer semantics must be a JSON object")
    expected = {"SOURCE.json": component.SOURCE31_RAW_SHA256,
                "SOURCE.png": component.SOURCE_RGB_RAW_SHA256,
                "CALIBRATION.json": component.CALIBRATION_RAW_SHA256}
    for name, expected_sha in expected.items():
        if sha(raw["inputs/" + name]) != expected_sha:
            raise ValueError("Campaign input differs from the registered source31: " + name)
    return raw


def stage_inputs(workspace, raw):
    root = Path(workspace)
    for name, rel in INPUT_REFS.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw["inputs/" + name])
    fact = root / FEEDBACK_REL
    fact.parent.mkdir(parents=True, exist_ok=True)
    fact.write_bytes(raw["DEVELOPMENT_FEEDBACK.json"])
    model_sha = sha(raw["MODEL_CONFIG.json"])
    save(root / SEMANTICS_REL, finalize_semantics(json.loads(raw["CONSUMER_SEMANTICS.json"]), model_sha))
    manifest = {
        "version": VERSION,
        "source_origin": "Registered human-defined primitive; proposal choice remains open",
        "model_config_sha256": model_sha,
        "model_identity_scope": "EXTERNAL_SERVICE_MODEL_CONTRACT_NOT_PI_WEIGHTS_OR_SOURCE_VLA_PROVENANCE",
        "input_sha256": {name: sha(blob) for name, blob in raw.items()},
        "staged_sha256": {rel: sha((root / rel).read_bytes()) for rel in
                          [*INPUT_REFS.values(), FEEDBACK_REL, SEMANTICS_REL]},
        "semantics_transformation": "Only exact $MODEL_CONFIG_SHA256 string values expanded",
        "robot_episodes": 0, "empirical_admission": False,
    }
    save(root / MANIFEST_REL, manifest)
    protected = {rel: (root / rel).read_bytes() for rel in
                 [*INPUT_REFS.values(), FEEDBACK_REL, SEMANTICS_REL, MANIFEST_REL]}
    return manifest, protected


def stage_successor_catalogue(workspace, package, model_sha, manifest, protected):
    """Explicit frozen source package; never infer new commands from prose."""
    from fastrsi.native_goal_component_v1 import _reference, _read
    package = Path(package).resolve()
    package_manifest = package/'MANIFEST.json'
    if package_manifest.is_symlink() or not package_manifest.is_file():
        raise ValueError('Ordinary successor input manifest required')
    document = json.loads(package_manifest.read_bytes())
    if set(document) != {'version', 'catalogue_path', 'files'} or document['version'] != 'native_goal_component_package_v2':
        raise ValueError('Unknown successor source package')
    files = document['files']; catalogue_path = document['catalogue_path']
    if not isinstance(files, dict) or catalogue_path not in files:
        raise ValueError('Bound successor catalogue missing')
    before = sha(package_manifest.read_bytes())
    for rel, digest in files.items():
        ref = dict(path=rel, sha256=digest); _reference(ref, 'successor package')
        if rel in protected or (Path(workspace)/rel).exists():
            raise ValueError('Successor input may not overwrite existing workspace data')
        blob = _read(package, ref)
        path = Path(workspace)/rel; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(blob)
        protected[rel] = blob
    if sha(package_manifest.read_bytes()) != before:
        raise ValueError('Successor package changed during staging')
    catalogue = finalize_semantics(json.loads(protected[catalogue_path]), model_sha)
    if set(catalogue) != {'version', 'candidates'} or catalogue['version'] != 'native_goal_component_catalogue_v2':
        raise ValueError('Unknown successor catalogue')
    rendered = (json.dumps(catalogue, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    Path(workspace, catalogue_path).write_bytes(rendered); protected[catalogue_path] = rendered
    manifest['successor_source_package'] = dict(manifest_sha256=before, original_files=files,
        catalogue_path=catalogue_path, transformation='Exact service MODEL_CONFIG placeholder only; no source actions changed')
    manifest['staged_sha256'].update({rel: sha(protected[rel]) for rel in files})
    save(Path(workspace)/MANIFEST_REL, manifest); protected[MANIFEST_REL] = (Path(workspace)/MANIFEST_REL).read_bytes()
    return catalogue['candidates'], catalogue_path


class CompleteReads:
    """Certify original read_file pages actually returned all immutable lines.

    This does not perform reads itself or insert facts into model messages.
    read_files child operations pass through the same original tool wrapper.
    """
    FOOTER = re.compile(r"\n\n\((Showing lines (\d+)-(\d+) of (\d+)\. Use offset=(\d+) to continue\.|End of file — (\d+) lines total)\)\Z")

    def __init__(self, workspace, paths):
        self.workspace = Path(workspace)
        self.expected = {p: (self.workspace / p).read_text().splitlines() for p in paths}
        self.seen = {p: set() for p in paths}
        self.eof = set()

    def observe(self, name, params, result):
        if (name != "read_file" or not isinstance(result, str) or result.startswith("Error")
                or len(result) > 16000):
            return
        rel = relative(self.workspace, params.get("path", ""))
        if rel not in self.expected:
            return
        lines = self.expected[rel]
        if not lines and result.startswith("(Empty file:"):
            self.eof.add(rel)
            return
        match = self.FOOTER.search(result)
        if match is None:
            return
        total = int(match[6] or match[4])
        rows = result[:match.start()].splitlines()
        parsed = []
        for row in rows:
            numbered = re.fullmatch(r"(\d+)\| (.*)", row)
            if numbered is None:
                return
            number = int(numbered[1])
            if not 1 <= number <= len(lines) or numbered[2] != lines[number - 1]:
                return
            parsed.append(number)
        offset = params.get("offset", 1)
        if type(offset) is not int or total != len(lines) or not parsed or parsed != list(range(offset, offset + len(parsed))):
            return
        end = parsed[-1]
        if match[6]:
            if end != total:
                return
            self.eof.add(rel)
        elif tuple(map(int, match.group(2, 3, 5))) != (offset, end, end + 1) or end >= total:
            return
        self.seen[rel].update(parsed)

    def complete(self):
        return sorted(p for p in self.expected if p in self.eof and
                      self.seen[p] == set(range(1, len(self.expected[p]) + 1)))

    def require(self):
        missing = sorted(set(self.expected) - set(self.complete()))
        if missing:
            raise ValueError("Complete native tool reads still required: " + ", ".join(missing))


def selected_skill(workspace, spec, backend):
    doc = backend.read_yaml_block(Path(workspace) / "SKILLRUNTIME.md")
    rows = [s for s in doc["skillruntimes"] if "skillruntime://" + s["id"] == spec["skillruntime_ref"]]
    if len(rows) != 1:
        raise ValueError("Exactly one selected registered SkillRuntime required")
    return rows[0]


def stripped_registry(document):
    value = copy.deepcopy(document)
    for skill in value["skillruntimes"]:
        chunk = skill.get("output_contract", {}).get("action", {}).get("chunk", {})
        if "preferred_replan_after_steps" in chunk:
            raise ValueError("This goal-component probe must not use the prefix candidate")
        chunk.pop(component.FIELD, None)
    return value


def validate_draft(workspace, spec, anchor, case, baseline_doc, backend):
    actual = backend.read_yaml_block(Path(workspace) / "SKILLRUNTIME.md")
    if stripped_registry(actual) != stripped_registry(baseline_doc):
        raise ValueError("Only the typed native_goal_component field may change")
    for skill in actual["skillruntimes"]:
        component.extract_component(skill)
    receipt = backend.validate_component_selection(workspace, spec, anchor,
                                                   goal_component_case=case)
    return receipt, component.extract_component(selected_skill(workspace, spec, backend))


def install_proposal_tools(agent, workspace, log, tracker, protected, validate, backend):
    original = agent.tools.execute

    async def execute(name, params):
        rel = relative(workspace, params.get("path", ""))
        if name in WRITE_TOOLS:
            error = None
            if rel not in ("SKILLRUNTIME.md", PROPOSAL_REL):
                error = "Only SKILLRUNTIME.md typed declarations or the proposal note are writable; no dispatch"
            else:
                try:
                    tracker.require()
                except ValueError as exc:
                    error = str(exc)
            if error:
                backend.record(log, event="proposal_write_blocked", name=name, params=params, reason=error)
                return "Error: " + error
        if name == "verify_session":
            return "Error: proposal-only probe has no executed Session to verify"
        before = (Path(workspace) / "SKILLRUNTIME.md").read_bytes()
        seen_before, eof_before = copy.deepcopy(tracker.seen), set(tracker.eof)
        outcome = await original(name, params)
        if name == "read_files" and len(outcome) > 16000:
            # Its child reads are logged, but the original Agent truncates the
            # aggregate tool message. Those children did not all reach the LLM.
            tracker.seen, tracker.eof = seen_before, eof_before
        else:
            tracker.observe(name, params, outcome)
        if name in WRITE_TOOLS and rel == "SKILLRUNTIME.md" and not outcome.startswith("Error"):
            try:
                receipt, declaration = validate()
                for path, blob in protected.items():
                    if (Path(workspace) / path).read_bytes() != blob:
                        raise ValueError("Read-only source/fact changed: " + path)
            except Exception as exc:
                (Path(workspace) / "SKILLRUNTIME.md").write_bytes(before)
                backend.record(log, event="invalid_component_edit_rolled_back", error=str(exc))
                return "Error: invalid component draft rolled back: " + str(exc)
            saved = dict(component_present=declaration is not None,
                         registry_sha256=sha((Path(workspace) / "SKILLRUNTIME.md").read_bytes()),
                         empirical_admission=False, robot_episodes=0)
            backend.record(log, event="native_goal_component_commit", **saved)
            outcome += "\n[Actual draft receipt] " + json.dumps(saved) + "; this is not a physical validation or admitted skill."
        return outcome

    agent.tools.execute = execute
    return original


def install_read_tracker(agent, tracker):
    original = agent.tools.execute

    async def execute(name, params):
        seen_before, eof_before = copy.deepcopy(tracker.seen), set(tracker.eof)
        result = await original(name, params)
        if name == "read_files" and len(result) > 16000:
            tracker.seen, tracker.eof = seen_before, eof_before
        else:
            tracker.observe(name, params, result)
        return result

    agent.tools.execute = execute
    return original


@contextmanager
def scoped_selection_validation(backend, workspace, case):
    """Enable only this new Session; default validation is restored on failure."""
    original = backend.validate_component_selection
    root = Path(workspace).resolve()

    def scoped(w, planned, anchor, **kwargs):
        if Path(w).resolve() == root and planned.get("session_id") == case["id"]:
            if kwargs and kwargs.get("goal_component_case") != case:
                raise ValueError("Different goal-component scope requested")
            return original(w, planned, anchor, goal_component_case=case)
        return original(w, planned, anchor, **kwargs)

    backend.validate_component_selection = scoped
    try:
        yield
    finally:
        backend.validate_component_selection = original


def load_backend(output, campaign):
    os.environ["FASTRSI_NATIVE_RUN"] = str(output)
    import run_native_baseline as backend
    from PhyAgentOS.runtime.state_io.markdown_yaml import read_yaml_block
    backend.ROOT, backend.CAMPAIGN = Path(output), Path(campaign)
    backend.read_yaml_block = read_yaml_block
    return backend


async def run_probe(campaign, output, *, backend=None, provider_factory=None,
                    max_tokens=3072, max_iterations=12, explicit_goal_current_state_v1=False,
                    typed_component_transaction_v1=False, successor_package=None):
    """Injected providers are CPU protocol tests; normal CLI uses measured Qwen."""
    started = time.perf_counter()
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("A fresh empty output directory is required; do not reuse a probe")
    output.mkdir(parents=True, exist_ok=True)
    result = dict(version=VERSION, status="RUNNING", robot_episodes=0,
                  policy_inference_calls=0, empirical_admission=False, sealed=False,
                  distillation_performed=False, learning_benefit=False,
                  scope="Real original Agent typed-program proposal and read-only new-Session planning only")
    try:
        if type(explicit_goal_current_state_v1) is not bool or type(typed_component_transaction_v1) is not bool:
            raise ValueError("Explicit proposal modes must be boolean")
        context_mode = explicit_goal_current_state_v1 or typed_component_transaction_v1
        if successor_package is not None and not typed_component_transaction_v1:
            raise ValueError('Successor source catalogue requires the opt-in typed transaction')
        if os.environ.get("FASTRSI_EXPERIENCE_ADMISSION") != "legacy_raw":
            raise ValueError("Explicit isolated legacy_raw draft inheritance required; no seal/admission")
        raw = campaign_inputs(campaign)
        backend = backend or load_backend(output, campaign)
        provider_factory = provider_factory or (lambda folder, role: backend.measured_provider(folder, role, max_tokens))
        model_path = campaign / "MODEL_CONFIG.json"
        common = dict(suite="libero_90", task_id=64, init_state_id=29, max_steps=400,
                      development_suite_extension="libero90_stack_flow_v1",
                      task_description=component.SOURCE_GOAL,
                      native_registry_lifecycle="incremental_v1",
                      native_component_updates="registered_v1", registered_goal_component=True,
                      model_config=str(model_path), model_config_sha256=sha(raw["MODEL_CONFIG.json"]))
        folder = output / "proposal"
        folder.mkdir()
        case = dict(common, id="goal_component_proposal", mode="development")
        workspace, spec, _ = backend.setup_workspace(folder, case, None)
        manifest, protected = stage_inputs(workspace, raw)
        if typed_component_transaction_v1:
            from fastrsi.typed_component_transaction_v1 import stage_compact_feedback
            result["feedback_representation"] = stage_compact_feedback(workspace, raw["DEVELOPMENT_FEEDBACK.json"],
                manifest, protected, feedback_rel=FEEDBACK_REL, manifest_rel=MANIFEST_REL)
        catalogue, catalogue_path = None, None
        if successor_package is not None:
            catalogue, catalogue_path = stage_successor_catalogue(workspace, successor_package,
                common['model_config_sha256'], manifest, protected)
            result['successor_source_package'] = manifest['successor_source_package']
        if context_mode:
            # Runtime import only: cycle imports this module's established tools.
            # Reuse its pure native-schema helper, not its request/propose loop.
            from probe_native_goal_learning_cycle_v1 import current_component_state
            save(workspace / STATE_REL, current_component_state(workspace, spec, backend))
            protected[STATE_REL] = (workspace / STATE_REL).read_bytes()
        anchor = backend.component_selection_anchor(workspace, spec)
        baseline_doc = backend.read_yaml_block(workspace / "SKILLRUNTIME.md")
        before = backend.hashes(workspace)
        required_reads = (*REQUIRED_READS, STATE_REL) if context_mode else REQUIRED_READS
        if typed_component_transaction_v1:
            required_reads = (*required_reads, MANIFEST_REL)
        if catalogue_path is not None:
            required_reads = (*required_reads, catalogue_path)
        tracker = CompleteReads(workspace, required_reads)
        log = folder / "AGENT_EVENTS.jsonl"
        agent = backend.bind_agent(workspace, provider_factory(folder, "proposal"), log,
                                   component_updates=True, max_iterations=max_iterations, read_files=True)
        check = lambda: validate_draft(workspace, spec, anchor, case, baseline_doc, backend)
        original = install_proposal_tools(agent, workspace, log, tracker, protected, check, backend)
        restore_transaction = None
        transaction_receipts = []
        try:
            if typed_component_transaction_v1:
                from fastrsi.typed_component_transaction_v1 import install as install_transaction
                restore_transaction, transaction_receipts = install_transaction(agent, workspace, spec, tracker,
                    protected, check, backend, log, model_config_sha256=common["model_config_sha256"],
                    candidate_catalogue=catalogue, catalogue_path=catalogue_path)
            if successor_package is None:
                reply = await agent.process_direct(
                (("Task goal: " + component.SOURCE_GOAL + ". Review the actual development evidence for this task "
                  "and decide whether the existing typed component should change or remain unchanged. "
                  "These evidence files are in your active workspace; use the available file tools to inspect them. "
                  "Completely read " + STATE_REL + "; it records the current installed component parsed by native schemas, "
                  "whereas the consumer schema describes only a possible interface. ")
                 if context_mode else "")
                + (("The optional commit_goal_component tool accepts candidate_id=dual_anchor_v1 or unchanged after complete reads. "
                    "Both decisions are legal; it patches only the existing typed declaration using native tools. "
                    "No statement in your final reply counts as a registry transaction. The feedback file uses lossless compact "
                    "JSON and fits a single read_file(limit=100); its unchanged original bytes and SHA are in INPUT_MANIFEST.json. "
                    "This presentation and transaction repair supplies no new experiment evidence. "
                    + PROPOSAL_REL + " is an optional OUTPUT note, not an existing input; reading or creating it is not required. ")
                   if typed_component_transaction_v1 else "")
                + "Use the original file tools to review actual historical developer facts. Read completely "
                + FEEDBACK_REL + ", " + SEMANTICS_REL + ", TARGETS.md and SKILLRUNTIME.md. "
                "Paginate read_file with limit/offset until all evidence lines are visible; each returned tool message "
                "must stay below the native Agent's 16000-character cap. Use limit=100 for large files. "
                "read_files is allowed only when its entire aggregate fits that same cap. "
                "Inspect " + MANIFEST_REL + " for the exact staged input paths and hashes. "
                + ("Decide whether the feedback supports adopting the existing human-defined dual-anchor "
                   if context_mode else
                   "Decide whether the feedback supports adopting the existing registered human-defined dual-anchor ")
                + "operation as an UNVALIDATED DEVELOPMENT DRAFT, or leaving the original component unchanged. "
                "No-op is legal; no action is preselected. The only executable change is the typed "
                "output_contract.action.chunk.native_goal_component declaration described in CONSUMER_SEMANTICS.json, "
                "in the existing pi05_libero_remote SkillRuntime. Keep all surrounding model, runtime, target, "
                "observation and task contracts unchanged. Do not add prefix slicing, Python, a new runtime, "
                "skills, fabricated physical outcomes, or modify any source program/image/calibration/facts. "
                "MODEL_CONFIG SHA identifies the current external service/model contract, not pi0.5 weights. "
                + ("If adopting, call commit_goal_component(candidate_id=dual_anchor_v1), or save the exact refs with original file tools. "
                   "For an explicit no-op transaction, use candidate_id=unchanged. A Markdown note is not an executable update. "
                   if typed_component_transaction_v1 else
                   "If adopting, save the exact declarative refs with existing edit_file/patch_yaml_item/write_file "
                   "tools in SKILLRUNTIME.md; a Markdown note alone is not an executable update. ")
                +
                "You may separately write " + PROPOSAL_REL + " explaining evidence limits. "
                "Do not queue, execute or verify a Session. "
                + ("Limited development evidence does not establish held-out generalization, learning benefit or net savings. "
                   if context_mode else
                   "One exposed developer instance establishes neither generalization nor learning or net savings. ")
                + "Report the actual saved declaration or legal no-op.",
                    session_key="cli:goal-component-proposal")
            else:
                reply = await agent.process_direct(
                    'Task goal: '+component.SOURCE_GOAL+'. Review actual development feedback and decide whether '
                    'to keep the current component or adopt one provided source-bound successor as an unvalidated draft. '
                    'Completely read '+', '.join(required_reads)+'. Use the original read_file/read_files tools and '
                    'paginate with limit=100; tool messages must remain below16000 characters. The catalogue supplies '
                    'the only available candidate identifiers. Its sources are actual successful development records, '
                    'not proof of transfer, learning advantage or economic return. Source annotations are supplied '
                    'command geometry and do not certify physical grasp or release. Do not rewrite any source, goal, '
                    'model, runtime or target. Both no-op and adoption are legal. Use commit_goal_component with one '
                    'catalogue candidate_id or unchanged. A prose answer does not commit an update. Do not queue, '
                    'execute or verify a Session. Report the actual saved draft or legal no-op.',
                    session_key='cli:goal-component-successor-proposal')
            save(folder / "REPLY.json", {"text": reply})
        finally:
            if restore_transaction is not None:
                restore_transaction()
            agent.tools.execute = original
            await backend.finish_agent(agent, log)
        tracker.require()
        if typed_component_transaction_v1:
            result["typed_component_transactions"] = transaction_receipts
        receipt, declaration = check()
        for path, blob in protected.items():
            if (workspace / path).read_bytes() != blob:
                raise ValueError("Read-only supplied input was changed: " + path)
        after = backend.hashes(workspace)
        changed = declaration is not None
        modified = {p: dict(before=before.get(p), after=after.get(p)) for p in sorted(set(before) | set(after))
                    if before.get(p) != after.get(p)}
        save(folder / "DRAFT.json", dict(status="UNVALIDATED_DEVELOPMENT_DRAFT" if changed else "LEGAL_NO_COMPONENT_CHANGE",
            declaration=declaration, compatibility=receipt, complete_reads=tracker.complete(), modified_files=modified,
            empirical_admission=False, source_manifest=manifest))
        result["proposal"] = dict(component_adopted=changed, declaration=declaration,
            registry_sha256=sha((workspace / "SKILLRUNTIME.md").read_bytes()), complete_reads=tracker.complete(),
            modified_files=modified, origin="Model proposal about a human-defined operation interface")
        if context_mode:
            result["proposal"]["current_component_state_sha256"] = sha(protected[STATE_REL])

        if context_mode:
            # This helper re-parses the child's actual registry; a pre-update
            # snapshot must not masquerade as its new installed component.
            from probe_native_goal_learning_cycle_v1 import readonly_selection
            result["new_session"] = await readonly_selection(
                output, campaign, common, workspace, protected, backend, provider_factory)
        else:
            # This remains an isolated unadmitted protocol branch, including no-op.
            second = output / "new_session"
            second.mkdir()
            next_case = dict(common, id="goal_component_new_session", mode="evaluation",
                             planning_only=True, require_discovery_reads=True, native_experience_maintenance=False)
            inherited, requested, previous = backend.setup_workspace(second, next_case, workspace)
            if (inherited / "SKILLRUNTIME.md").read_bytes() != (workspace / "SKILLRUNTIME.md").read_bytes():
                raise ValueError("New Session did not inherit exact registered bytes")
            for path, blob in protected.items():
                if (inherited / path).read_bytes() != blob:
                    raise ValueError("New Session did not preserve fixed refs/facts: " + path)
            log = second / "AGENT_EVENTS.jsonl"
            fixed_before = backend.hashes(inherited)
            agent = backend.bind_agent(inherited, provider_factory(second, "selection"), log,
                                       evaluation=True, component_updates=True, max_iterations=max_iterations, read_files=True)
            inherited_reads = CompleteReads(inherited, ("SKILLRUNTIME.md", "TARGETS.md", "REQUEST.json"))
            original = install_read_tracker(agent, inherited_reads)
            try:
                with scoped_selection_validation(backend, inherited, next_case):
                    reply, planned = await backend.plan_session(agent, next_case, inherited, requested, previous, log)
                save(second / "REPLY.json", {"text": reply})
                # Validate both base and selected directly with this explicit case.
                base = backend.validate_component_selection(inherited, requested, anchor, goal_component_case=next_case)
                chosen = backend.validate_component_selection(inherited, planned, anchor, goal_component_case=next_case)
                inherited_reads.require()
                result["new_session"] = dict(planned=planned, complete_reads=inherited_reads.complete(),
                    selected_declaration=component.extract_component(selected_skill(inherited, planned, backend)),
                    compatibility={"base": base, "selected": chosen},
                    registry_bytes_inherited=True, workspace=str(inherited), task_executed=False)
                save(second / "SELECTION.json", result["new_session"])
            finally:
                agent.tools.execute = original
                await backend.finish_agent(agent, log)
            if fixed_before != backend.hashes(inherited):
                raise ValueError("Evaluation lifecycle changed fixed experience/components")
            if not (inherited / ".FASTRSI_EVALUATION_CLOSED.json").is_file():
                raise ValueError("Evaluation lifecycle did not close")
        for path, blob in raw.items():
            if (campaign / path).read_bytes() != blob:
                raise ValueError("Original campaign input modified: " + path)
        result.update(status="COMPLETED_REAL_AGENT_COMPONENT_SELECTION" if changed else "COMPLETED_LEGAL_NO_COMPONENT_CHANGE",
                      evaluation_updates_blocked=True, original_campaign_inputs_unchanged=True)
    except Exception:
        result.update(status="FAILED", error=traceback.format_exc())
    result["elapsed_seconds"] = time.perf_counter() - started
    result["model_calls"] = len(list(output.glob("*/*_call_*/REQUEST.json")))
    save(output / "RESULT.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model-base-url", default=None)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--allow-real-model-calls", action="store_true")
    parser.add_argument("--explicit-goal-current-state-v1", action="store_true")
    parser.add_argument("--typed-component-transaction-v1", action="store_true")
    parser.add_argument('--successor-package', type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.allow_real_model_calls:
        parser.error("This entry requires explicit --allow-real-model-calls; CPU tests inject a scripted provider")
    if not 1 <= args.max_iterations <= 20 or not 1 <= args.max_tokens <= 4096:
        parser.error("Bounded generation/iteration limits required")
    output = args.output or Path(os.environ.get("FASTRSI_NATIVE_RUN", str(args.campaign / "stream")))
    backend = load_backend(output.resolve(), args.campaign.resolve())
    if args.model_base_url:
        backend.QWEN_API = args.model_base_url
    result = asyncio.run(run_probe(args.campaign, output, backend=backend,
        max_tokens=args.max_tokens, max_iterations=args.max_iterations,
                                  explicit_goal_current_state_v1=args.explicit_goal_current_state_v1,
                                  typed_component_transaction_v1=args.typed_component_transaction_v1,
                                  successor_package=args.successor_package))
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
    return 1 if result["status"] == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())

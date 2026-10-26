"""One semantic order decision inside the existing native Agent, before append.

The common constructor still owns the Session. This component creates no Agent,
does not produce outcome labels, and does not approve its own decision.
"""
import copy
import hashlib
import json
from pathlib import Path

from cost_trace import span
from fastrsi.paged_reads import read_complete
from fastrsi.same_world_native_binding import load_binding, CHOICE_MODE
from fastrsi.tool_origin import TOOL_ORIGIN
from same_world_launch import validate_step_choice


def bind_order_evidence(case, workspace):
    """Expose the same historical facts to every arm without rewriting memory."""
    keys = ('order_evidence_path', 'order_evidence_sha256')
    if not any(key in case for key in keys):
        return None
    if not all(isinstance(case.get(key), str) and case[key] for key in keys):
        raise ValueError('Order evidence requires its exact file and SHA')
    load_binding(case)
    from build_same_world_order_evidence import validate_evidence
    source = Path(case['order_evidence_path'])
    receipt = validate_evidence(source, case['order_evidence_sha256'], verify_sources=False)
    receipt_bytes = source.read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest() != case['order_evidence_sha256']:
        raise ValueError('Order receipt changed during workspace binding')
    facts_bytes = (json.dumps({'version': receipt['version'],
                              'decision_facts': receipt['decision_facts']},
                             ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
    relative = Path('artifacts/order_evidence') / case['order_evidence_sha256']
    destination = Path(workspace) / relative
    destination.mkdir(parents=True, exist_ok=True)
    for name, raw in [('RECEIPT.json', receipt_bytes), ('DECISION_FACTS.json', facts_bytes)]:
        path = destination / name
        if path.exists():
            if path.read_bytes() != raw:
                raise ValueError('Inherited order evidence differs: ' + name)
        else:
            with path.open('xb') as output:
                output.write(raw)
    return dict(path=(relative/'DECISION_FACTS.json').as_posix(),
                sha256=hashlib.sha256(facts_bytes).hexdigest(),
                receipt_sha256=case['order_evidence_sha256'])


def native_order_callback(agent, workspace, case, log_event):
    binding = load_binding(case)
    if binding['selection_mode'] != CHOICE_MODE:
        raise ValueError('Only unresolved registered order choices use this callback')
    w = Path(workspace)
    relative = 'reports/order_decisions/' + case['id'] + '.json'
    output = w / relative

    async def choose_order(request, input_hashes):
        if output.exists():
            raise ValueError('Decision artifact already exists; never reuse without binding')
        source = w / 'SAME_WORLD_BINDING.json'
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        text = await read_complete(
            agent.tools.execute, 'SAME_WORLD_BINDING.json',
            on_page=lambda page: log_event(event='native_order_evidence_page', page=page))
        if hashlib.sha256(source.read_bytes()).hexdigest() != before:
            raise ValueError('Order options changed during native read')
        current = json.loads(text)
        if (current['registration_sha256'] != binding['registration_sha256'] or
                current['allowed_sequences'] != binding['allowed_sequences']):
            raise ValueError('Workspace and registered order choices differ')
        options = [[{'stage_key': key} for key in sequence]
                   for sequence in binding['allowed_sequences']]
        facts_binding = None
        decision_facts = None
        if any(key in case for key in ('order_evidence_path', 'order_evidence_sha256')):
            # Setup exposes a content-addressed view. The actual native read,
            # not a receipt hash, supplies historical facts to this decision.
            from build_same_world_order_evidence import validate_evidence
            relative_root = Path('artifacts/order_evidence') / case['order_evidence_sha256']
            receipt = validate_evidence(w/relative_root/'RECEIPT.json',
                                       case['order_evidence_sha256'], verify_sources=False)
            facts_path = w/relative_root/'DECISION_FACTS.json'
            facts_before = hashlib.sha256(facts_path.read_bytes()).hexdigest()
            facts_text = await read_complete(
                agent.tools.execute, (relative_root/'DECISION_FACTS.json').as_posix(),
                on_page=lambda page: log_event(event='native_order_history_page', page=page))
            facts = json.loads(facts_text)
            if (facts != {'version': receipt['version'], 'decision_facts': receipt['decision_facts']}
                    or hashlib.sha256(facts_path.read_bytes()).hexdigest() != facts_before):
                raise ValueError('Historical decision facts changed or differ from the bound receipt')
            decision_facts = facts['decision_facts']
            facts_binding = dict(path=(relative_root/'DECISION_FACTS.json').as_posix(),
                                 sha256=facts_before, receipt_sha256=case['order_evidence_sha256'])
        evidence = {str(path.relative_to(w)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [w/'LESSONS.md', w/'memory/MEMORY.md'] if path.is_file()}
        inputs = dict(goal=request['goal'], stage_definitions=binding['stage_definitions'],
                      allowed_steps=options, evidence_sha256=evidence,
                      registration_sha256=binding['registration_sha256'],
                      current_fact_sha256=input_hashes,
                      history_read=facts_binding,
                      scope='Task-level prior-experience choice; no current pre-reset visual state')
        log_event(event='native_order_decision_inputs', inputs=inputs,
                  selection_source='native_agent_not_a_trained_fast_policy')
        original_execute = agent.tools.execute

        async def decision_tools(name, params):
            if name in ('read_file', 'read_files', 'list_dir'):
                return await original_execute(name, params)
            if name == 'write_file':
                path = params.get('path')
                resolved = (w/path).resolve() if isinstance(path, str) else None
                if resolved == output.resolve() and not output.exists():
                    return await original_execute(name, params)
            return 'Error: this decision only permits reading facts and writing its one order artifact; no Session is queued yet.'

        agent.tools.execute = decision_tools
        try:
            prompt = (
                'Choose the order for this joint physical task using the registered stage meanings '
                'and relevant admitted prior experience in your native context. '
                'You have no current initial robot image before reset; do not infer it from instance '
                'numbers, Session names, or task indices. Do not invent branch outcomes. '
                'Only write a JSON object with exactly the key steps to ' + relative + '. '
                'The steps must exactly equal one of the following complete options: ' + json.dumps(options) + '. '
                'Goal: ' + request['goal'] + '. Stage meanings: ' + json.dumps(binding['stage_definitions']) + '. '
                + ('Previously executed historical facts, with their distinct cost and evidence scopes: '
                   + json.dumps(decision_facts, ensure_ascii=False) + '. '
                   'These facts describe past executions, not current or future outcomes. '
                   if decision_facts is not None else '') +
                'Use the existing write_file tool. Do not build or append a Session, change other files, '
                'or claim execution succeeded; the common constructor queues after validation.')
            with span('native_order_decision', case=case['id']):
                response = await agent.process_direct(prompt, session_key='order:'+case['id'])
        finally:
            agent.tools.execute = original_execute
        if hashlib.sha256(source.read_bytes()).hexdigest() != before:
            raise ValueError('Order options changed while the native Agent was deciding')
        if facts_binding and hashlib.sha256(facts_path.read_bytes()).hexdigest() != facts_binding['sha256']:
            raise ValueError('Historical decision facts changed while the native Agent was deciding')
        if not output.is_file():
            raise ValueError('Native Agent did not submit an order decision')
        raw = output.read_bytes()
        value = json.loads(raw)
        if not isinstance(value, dict) or set(value) != {'steps'}:
            raise ValueError('Order artifact must contain only steps')
        steps = validate_step_choice(value['steps'], binding['allowed_sequences'])
        log_event(event='native_order_decision', steps=steps,
                  artifact=relative, artifact_sha256=hashlib.sha256(raw).hexdigest(),
                  response=str(response), inputs=inputs,
                  status='SELECTED_NOT_YET_EXECUTED_OR_VERIFIED')
        return copy.deepcopy(steps)

    async def choose(request, input_hashes):
        token = TOOL_ORIGIN.set('native_order_decision')
        try:
            return await choose_order(request, input_hashes)
        finally:
            TOOL_ORIGIN.reset(token)

    return choose

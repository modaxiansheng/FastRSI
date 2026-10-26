"""Public deterministic Session construction; not a learned routing policy.

The bounded interpreter reads current inputs through existing tools, constructs
one schema-valid pending Session, and leaves dispatch to the existing guard.
No prior scene facts, action results or complete Session answer are cached.
"""
import copy
import hashlib
import json
import re
from pathlib import Path
from .tool_origin import TOOL_ORIGIN
from .paged_reads import read_complete

VERSION = 'public_session_construction_v1'
READS = ['skills/benchmarking/SKILL.md', 'RUNTIME.md', 'TARGETS.md',
         'SKILLRUNTIME.md', 'SESSIONS.md', 'LOG.md', 'REQUEST.json']


class FlowFallback(ValueError):
    pass


def intent_from_spec(spec, goal):
    return {'goal': goal, 'session_id': spec['session_id'], 'goal_id': spec.get('goal_id'),
            'benchmark': copy.deepcopy(spec['benchmark']),
            'execution_constraints': {k: copy.deepcopy(spec[k]) for k in
                ['execution', 'timeouts', 'safety_profile', 'verification_profile']},
            'routing_constraints': copy.deepcopy(spec['routing']),
            'construction': 'Construct a Session from current registries and immutable intent. No complete Session answer supplied.'}


def numbered_text(result):
    """Parse complete native read_file output; never treat truncated text as complete."""
    if result.startswith('Error'):
        raise FlowFallback(result)
    if result.startswith('(Empty file:'):
        return ''
    if '(Showing lines ' in result or '(End of file — ' not in result:
        raise FlowFallback('Incomplete native read; hand back for pagination')
    lines = []
    for line in result.splitlines():
        match = re.fullmatch(r'\d+\| (.*)', line)
        if match:
            lines.append(match.group(1))
    return '\n'.join(lines)


def yaml_document(text):
    import yaml
    blocks = re.findall(r'```yaml\s*\n(.*?)\n```', text, flags=re.S)
    if len(blocks) != 1:
        raise FlowFallback('Expected one current YAML registry')
    try:
        value = yaml.safe_load(blocks[0])
    except yaml.YAMLError as exc:
        raise FlowFallback('Invalid registry YAML: '+str(exc)) from exc
    if not isinstance(value, dict):
        raise FlowFallback('Invalid current registry')
    return value


def construct(request, targets, skills, sessions):
    """Interpret the original benchmarking capability-selection contract."""
    from PhyAgentOS.runtime.schemas import SessionSpec
    if 'session' in request or request.get('instruction_decision'):
        raise FlowFallback('Unsupported request: full answer or semantic instruction decision')
    allowed = {'goal','session_id','goal_id','benchmark','execution_constraints','routing_constraints','construction'}
    if set(request) - allowed:
        raise FlowFallback('Unknown intent fields')
    goal = request['goal'];sid = request['session_id'];b = request['benchmark'];c = request['execution_constraints']
    if not isinstance(goal, str) or not goal.strip() or not isinstance(sid, str) or not sid.strip():
        raise FlowFallback('Missing current goal or Session identity')
    from .development_suite import STANDARD_STEP_LIMITS
    if b['benchmark_id'] != 'libero' or b['suite_id'] not in (*STANDARD_STEP_LIMITS, 'libero_90') or b['execution_mode'] != 'policy_loop':
        raise FlowFallback('Outside public v1 benchmark interface')
    if b['suite_id']=='libero_90':
        from .development_suite import valid_goal
        if not valid_goal(b['task_index'], goal, c['execution']['max_steps']):
            raise FlowFallback('Outside declared development scope')
    if set(c) != {'execution','timeouts','safety_profile','verification_profile'}:
        raise FlowFallback('Incomplete execution constraints')
    if c['execution']['reset_policy'] != 'session_runner':
        raise FlowFallback('Reset owner mismatch')
    if any(s.get('session_id') == sid for s in sessions['sessions']):
        raise FlowFallback('Existing Session: inspect and resume, never append again')
    matches = []
    for target in targets['targets']:
        if not target.get('enabled') or target.get('target_kind') != 'simulation':
            continue
        config = target.get('config', {})
        if (config.get('benchmark_name'), config.get('task_id'), config.get('init_state_id'), config.get('max_steps')) != (
            b['suite_id'], b['task_index'], b['instance_id'], c['execution']['max_steps']):
            continue
        for cap in target.get('benchmark_capabilities', []):
            if cap.get('benchmark_id') != b['benchmark_id'] or b['suite_id'] not in cap.get('suites', []):
                continue
            for mode in cap.get('execution_modes', []):
                if mode.get('mode') != b['execution_mode'] or mode.get('reset_owner') != 'session_runner':
                    continue
                for skill in skills['skillruntimes']:
                    sb = skill.get('benchmark', {})
                    if skill.get('enabled', True) is False or skill['id'] not in target.get('supported_skillruntimes', []):
                        continue
                    if target['target_kind'] not in skill.get('supported_target_kinds', []):
                        continue
                    if (sb.get('benchmark_id'), sb.get('execution_mode'), sb.get('target_interface'), sb.get('result_schema'), sb.get('reset_owner')) != (
                        b['benchmark_id'], b['execution_mode'], mode.get('interface'), 'benchmark_execution_result_v1', 'session_runner'):
                        continue
                    matches.append((target, skill))
    if len(matches) != 1:
        raise FlowFallback('No unique exact Target/SkillRuntime capability tuple')
    target, skill = matches[0];routing = request['routing_constraints']
    if routing['target_endpoint'] != target['runtime']['target_endpoint'] or not routing['policy_endpoint'].startswith('openpi://'):
        raise FlowFallback('Current endpoint mismatch')
    value = {'session_id': sid, 'goal_id': request.get('goal_id'), 'target_ref': 'target://'+target['id'],
             'skillruntime_ref': 'skillruntime://'+skill['id'], 'task_description': goal,
             'status':'pending', 'priority':'normal', 'benchmark': copy.deepcopy(b),
             'routing':copy.deepcopy(routing), **copy.deepcopy(c)}
    SessionSpec.model_validate(value)
    return value


def artifact(skill_bytes, source_trace_sha256):
    return {'version':VERSION,'status':'PUBLIC_ENGINEERING_PROTOTYPE_NOT_LEARNED',
            'origin':'Observed admitted development trace; deterministic original benchmarking contract',
            'source_trace_sha256':source_trace_sha256,
            'benchmark_skill_sha256':hashlib.sha256(skill_bytes).hexdigest(),
            'steps':['fresh_native_reads','unique_capability_tuple','bind_current_parameters','append_once'],
            'replaces':'High-level Session construction only; native verification, reporting and feedback retained',
            'reads':READS,'tools':['read_file','append_yaml_item'], 'cached_scene_facts':False}


async def execute_flow(execute, workspace, flow, log_event, choose_steps=None):
    """Return handoff evidence on every partial failure; never redispatch a write."""
    w = Path(workspace);completed=[];read_hashes={};write_attempted=False
    token = TOOL_ORIGIN.set('public_session_flow')
    try:
        if flow.get('version') != VERSION or flow.get('reads') != READS or flow.get('tools') != ['read_file','append_yaml_item'] or flow.get('steps') != ['fresh_native_reads','unique_capability_tuple','bind_current_parameters','append_once']:
            raise FlowFallback('Unknown flow structure')
        contents = {}
        for rel in READS:
            before_read = hashlib.sha256((w/rel).read_bytes()).hexdigest() if (w/rel).exists() else None
            content = await read_complete(execute,rel,on_page=completed.append,
                                          missing_ok=rel=='LOG.md' and not (w/rel).exists())
            if content is None and rel == 'LOG.md' and not (w/rel).exists():
                contents[rel] = '';read_hashes[rel]=None
            else:
                contents[rel] = content
                read_hashes[rel] = hashlib.sha256((w/rel).read_bytes()).hexdigest()
            if read_hashes[rel] != before_read:
                raise FlowFallback('Current input changed during read: '+rel)
        if read_hashes['skills/benchmarking/SKILL.md'] != flow['benchmark_skill_sha256']:
            raise FlowFallback('Benchmark skill changed: revalidate the public procedure')
        request = json.loads(contents['REQUEST.json'])
        planned = construct(request, yaml_document(contents['TARGETS.md']),
                            yaml_document(contents['SKILLRUNTIME.md']), yaml_document(contents['SESSIONS.md']))
        if choose_steps is not None:
            # The semantic provider sees immutable intent and read provenance,
            # never a complete Session to reconstruct. Only steps can change.
            try:
                steps = await choose_steps(copy.deepcopy(request), dict(read_hashes))
            except Exception as exc:
                raise FlowFallback('Semantic step choice failed: ' + str(exc)) from exc
            if not isinstance(steps, list) or not steps or not all(isinstance(row, dict) for row in steps):
                raise FlowFallback('Semantic step choice requires a nonempty list of step objects')
            planned['execution']['steps'] = copy.deepcopy(steps)
            from PhyAgentOS.runtime.schemas import SessionSpec
            SessionSpec.model_validate(planned)
        # Guard against stale snapshot reads immediately before the only write.
        for rel,sha in read_hashes.items():
            actual = hashlib.sha256((w/rel).read_bytes()).hexdigest() if (w/rel).exists() else None
            if actual != sha:
                raise FlowFallback('Current input changed during construction: '+rel)
        write_attempted = True
        result = await execute('append_yaml_item', {'path':'SESSIONS.md','list_key':'sessions','item':planned})
        completed.append({'tool':'append_yaml_item','session_id':planned['session_id'],'result':result})
        if result.startswith('Error'):
            raise FlowFallback(result)
        result = {'status':'QUEUED','planned':planned,'completed':completed,'input_sha256':read_hashes,
                  'origin':'public_session_flow'}
        if choose_steps is None:
            result['planning_model_calls'] = 0
        else:
            result.update(semantic_decision_callback=True, deterministic_constructor_model_calls=0)
        return result
    except (FlowFallback, KeyError, ValueError, TypeError, OSError) as exc:
        handoff={'status':'NATIVE_HANDOFF','reason':str(exc),'completed':completed,
                 'input_sha256':read_hashes,'write_attempted':write_attempted,
                 'instruction':'Read current queue before editing. Resume existing requested draft if present; never duplicate it.'}
        if choose_steps is not None:
            handoff.update(semantic_decision_callback=True, deterministic_constructor_model_calls=0)
        log_event(event='session_flow_handoff',**handoff)
        return handoff
    finally:
        TOOL_ORIGIN.reset(token)

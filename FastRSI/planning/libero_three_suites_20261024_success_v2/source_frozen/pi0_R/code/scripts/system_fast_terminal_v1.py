"""Compile benchmarking Results from complete current canonical records.

Execution, semantic verification, retries and native end hooks are unchanged.
This replaces only the language-model formatting of an already verified result.
"""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compile_report(workspace, case, sessions):
    workspace = Path(workspace).resolve()
    selected = [s for s in sessions if s['session_id'] == case['id']
                or s['session_id'].startswith(case['id'] + '_replan')]
    if not selected or any(s['status'] in ('pending', 'claimed', 'running', 'awaiting_verification')
                           for s in selected):
        raise ValueError('Current terminal records are incomplete')
    sources = {'SESSIONS.md': sha(workspace / 'SESSIONS.md')}
    rows = []
    for session in selected:
        rel = session['result']['artifact_dir'] + '/episode.json'
        path = (workspace / rel).resolve()
        if not path.is_relative_to(workspace) or not path.is_file():
            raise ValueError('Episode reference is missing or outside workspace')
        raw = path.read_bytes()
        episode = json.loads(raw)
        if episode['session_id'] != session['session_id']:
            raise ValueError('Session and episode identities differ')
        benchmark = episode['benchmark']
        if (benchmark['benchmark_name'], benchmark['task_id'], benchmark['init_state_id']) != (
                case['suite'], case['task_id'], case['init_state_id']):
            raise ValueError('Episode benchmark identity differs from task')
        if type(episode['success']) is not bool or type(episode['num_steps']) is not int:
            raise ValueError('Ambiguous recorded outcome')
        if episode['success'] != benchmark['success']:
            raise ValueError('Conflicting canonical environment outcomes')
        sources[rel] = hashlib.sha256(raw).hexdigest()
        semantic_rel = 'reports/evidence/verification-' + session['session_id'] + '.json'
        semantic_path = workspace / semantic_rel
        semantic = json.loads(semantic_path.read_text()) if semantic_path.exists() else None
        if case.get('require_compiled_semantic_receipt'):
            if not semantic or semantic.get('code') != 'verified' or semantic.get('session_id') != session['session_id']:
                raise ValueError('Actual semantic outcome receipt is missing')
        if semantic is not None:
            sources[semantic_rel] = sha(semantic_path)
        rows.append({'session_id': session['session_id'], 'terminal_state': session['status'],
                     'environment_success': episode['success'], 'steps': episode['num_steps'],
                     'return_value': episode['return_value'], 'artifact': rel,
                     'semantic_verification': semantic, 'semantic_artifact': semantic_rel if semantic else None})
    value = {'version': 'system_fast_terminal_v1', 'case_id': case['id'],
             'suite': case['suite'], 'task_id': case['task_id'], 'init_state_id': case['init_state_id'],
             'task_description': case['task_description'], 'attempts': len(rows),
             'physical_steps_all_attempts': sum(r['steps'] for r in rows),
             'environment_final_success': rows[-1]['environment_success'],
             'attempt_results': rows, 'source_sha256': sources,
             'scope': 'Recorded environment outcomes and existing semantic verification; no new visual inference.'}
    text = '# LIBERO task results\n\n' + json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    return value, text


async def consume_terminal(agent, case, workspace, log):
    from PhyAgentOS.runtime.state_io.markdown_yaml import read_yaml_block
    from fastrsi.native_binding import record
    workspace = Path(workspace)
    sessions = read_yaml_block(workspace / 'SESSIONS.md')['sessions']
    value, text = compile_report(workspace, case, sessions)
    selected = value['attempt_results']
    run_id = next(s['benchmark']['run_id'] for s in sessions if s['session_id'] == selected[0]['session_id'])
    rel = 'reports/libero-' + case['id'] + '-' + run_id + '.md'
    (workspace / 'reports').mkdir(exist_ok=True)
    if (workspace / rel).exists():
        raise FileExistsError('Refusing to reuse an old result report')
    # Preserve the actual native write-file tool, tool trace and destination checks.
    result = await agent.tools.execute('write_file', {'path': str(workspace / rel), 'content': text})
    if result.startswith('Error') or (workspace / rel).read_text() != text:
        raise ValueError('Native report write failed')
    if any(not (workspace / r['artifact']).is_file() for r in selected):
        raise ValueError('Structured report artifact reference is missing')
    if any(sha(workspace / p) != h for p, h in value['source_sha256'].items()):
        raise ValueError('Canonical result changed while compiling report')
    receipt = {'status': 'PASSED', 'version': value['version'], 'report': rel,
               'source_sha256': value['source_sha256'], 'report_sha256': sha(workspace / rel),
               'actual_reads': list(value['source_sha256']),
               'read_coverage': {'complete': True, 'fresh_report_write': True,
                                 'version': 'complete_canonical_value_compilation_v1'},
               'terminal_states': {r['session_id']: r['terminal_state'] for r in selected},
               'report_model_calls': 0, 'native_experience_maintenance_enabled': False,
               'scope': 'System optimization: compile all canonical task results; original verification and end hooks retained.'}
    record(log, event='terminal_results_consumed', **receipt)
    return receipt

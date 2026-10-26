"""Source-bound manual completeness audit of the six archived continuous reports.

This is not a general semantic grader. The manual observations below apply only
to the exact report bytes reviewed. Requests, tool writes and canonical outcomes
are checked mechanically; changed reports require a new manual review. No source,
runtime PASSED status, admission result or model output is changed.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = 'continuous_comparison_v1_20261003'
SKILL_SHA = '77b8034fe52faeac1fdfc06d15926acca975a7eb1622d6291466a21a5f63ea24'
REVIEWED = {
    ('P', 'continuous_0_t0_i3'): 'c6c29618f43c5cf1e4487cb60adb56c5d3e0c47ed853aa98442e64f09077c00d',
    ('P', 'continuous_1_t0_i1'): '64f0b5f1f41970c97b9e8c87716a4bcc5b8317f0b3f0742724b5cc84b677ffb7',
    ('P', 'continuous_2_t0_i0'): '7f7510379ee25a4b147a48e646fbd4afa84f54ea46f1c5667a0287afe15cda08',
    ('F', 'continuous_0_t0_i3'): '1ceaaf7a05fedf8e5eb8575b8099b46c21a7c9c1e335b34e1a5324a13262544d',
    ('F', 'continuous_1_t0_i1'): '01254d9b09b8e8a93d91266f561ffcac00116a27275b64f38b743c62a714657b',
    ('F', 'continuous_2_t0_i0'): 'aee0e1d16f626e1fce41631a6974633f864ec743f3edec30ff612a32df6e3858',
}
VISUAL_QUOTE = ('The final observation shows the alphabet soup and tomato sauce cartons '
                'placed inside the basket, fulfilling the task requirement.')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path, root):
    return {'path': str(path.relative_to(root)), 'sha256': sha(path)}


def messages_key(messages):
    return json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def content_facts(messages):
    counts = Counter()
    for message in messages:
        content = message.get('content')
        if isinstance(content, str):
            counts['text_string_messages'] += 1
        elif isinstance(content, list):
            for block in content:
                counts[block.get('type', 'unknown_block') if isinstance(block, dict)
                       else 'unknown_block'] += 1
        elif content is not None:
            counts['unknown_content'] += 1
    images = sum(counts[k] for k in ('image', 'image_url', 'input_image'))
    only_text = not any(v for k, v in counts.items() if k not in ('text_string_messages', 'text'))
    return {'content_types': dict(counts), 'image_blocks': images,
            'text_only': only_text,
            'scope': 'Actual request content blocks, not text references to RGB filenames.'}


def requirement_binding(path, root):
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.strip() == b'## Results')
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith(b'## ')), len(lines))
    raw = b''.join(lines[start:end])
    return {**binding(path, root), 'line_start': start + 1, 'line_end': end,
            'exact_section_text': raw.decode(),
            'section_sha256': hashlib.sha256(raw).hexdigest(),
            'manual_requirement_binding_matches': sha(path) == SKILL_SHA}


def manual_findings(arm, case_id):
    # These are whole-document manual observations, not keyword absence rules.
    fields = {name: {'status': 'MISSING_EXPLICITLY_REPORTED_VALUE', 'basis': 'Whole report manually reviewed.'}
              for name in ('configured_verifier_budget', 'effective_verifier_budget',
                           'consumed_verifier_budget', 'episode_count', 'attempt_count', 'recovery_count')}
    fields.update({
        'first_attempt_score_vs_assisted_final_score': {
            'status': 'ABSENT_APPLICABILITY_REQUIRES_MANUAL_REVIEW',
            'basis': 'Success is reported, but the two score definitions are not separated. '
                     'This three-case pilot is not an official benchmark score; do not fabricate one.'},
        'latency': {'status': 'PRESENT_POLICY_MEAN_ONLY',
                    'basis': 'Mean policy latency is reported; this is not end-to-end or full OS latency. '
                             'The native Results paragraph does not specify which latency scope.'},
        'failures': {'status': 'MANUAL_REVIEW_NO_SEPARATE_FAILURE_SUMMARY',
                     'basis': 'Final success is explicit and no runtime failure occurred in these cases. '
                              'Whether a separate zero-failure statement is required is a reporting choice; '
                              'this is not evidence of a hidden failed rollout.'},
        'artifact_references': {'status': 'PRESENT',
                                'basis': 'Current episode.json and verification_bundle.json are referenced.'},
    })
    notes_claim = arm == 'F' and case_id != 'continuous_2_t0_i0'
    return {'method': 'Manual whole-report review bound to exact report and skill SHA256; no keyword grading.',
            'fields': fields,
            'visual_narrative': {
                'direct_visual_reinspection_by_report_agent_established': False,
                'first_hand_visual_claim_classification': 'MANUAL_REVIEW_SOURCE_ATTRIBUTION_AMBIGUOUS',
                'basis': 'The visual sentence occurs under Verification/Evidence and is copied from the '
                         'verifier. It can be a quotation-like summary rather than a claim of personal '
                         'inspection, but the source is not explicitly attributed. Text-only reporting '
                         'does not independently corroborate the visual explanation.',
                'unqualified_visual_confirmation_repeated_in_notes': notes_claim,
                'notes': ('The Notes section additionally states confirmed by visual evidence and runtime metrics.'
                          if notes_claim else 'No separate Notes endorsement of visual evidence was found.'),
                'admission_implication': 'Do not promote this prose to independently verified visual knowledge '
                                         'merely because the report guard PASSED or environment succeeded.'}}


def audit(root=ROOT):
    root = Path(root).resolve()
    run = root / 'runs' / RUN
    rows = []
    for arm in ('P', 'F'):
        stream = run / arm / 'stream'
        result = read(stream / 'RESULT.json')
        events_path = stream / 'OS_COST_EVENTS.jsonl'
        events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
        services = {}
        verifiers = {}
        for path in sorted((stream / 'service_calls').glob('*/REQUEST.json')):
            request = read(path)
            services.setdefault(messages_key(request['messages']), []).append(path)
            for message in request['messages']:
                if not isinstance(message.get('content'), list):
                    continue
                for block in message['content']:
                    text = block.get('text', '') if isinstance(block, dict) else ''
                    if 'VERIFICATION_BUNDLE:\n' not in text:
                        continue
                    bundle = json.loads(text.split('VERIFICATION_BUNDLE:\n', 1)[1])
                    verifiers.setdefault(bundle['session']['session_id'], []).append(path)
        for case in result['cases']:
            case_id = case['case']['id']
            case_dir = stream / case_id
            workspace = case_dir / 'workspace'
            terminal = case['terminal_report']
            report_path = workspace / terminal['report']
            text = report_path.read_text()
            actual_sha = sha(report_path)
            assert actual_sha == terminal['report_sha256'], 'Archived report differs from its terminal receipt'
            requirement = requirement_binding(workspace / 'skills/benchmarking/SKILL.md', root)
            reviewed = actual_sha == REVIEWED.get((arm, case_id)) and requirement['manual_requirement_binding_matches']
            requests = []
            writes = []
            marker = 'Complete the Results section of the benchmarking skill for Session ' + case_id + '.'
            for path in sorted(case_dir.glob('agent_call_*/REQUEST.json'), key=lambda p: int(p.parent.name.split('_')[-1])):
                request = read(path)
                last_user = next((m.get('content') for m in reversed(request['messages']) if m['role'] == 'user'), '')
                if not isinstance(last_user, str) or marker not in last_user:
                    continue  # Do not mistake inherited prior-case report history for this report phase.
                matches = services.get(messages_key(request['messages']), [])
                assert len(matches) == 1, ('Ambiguous or missing actual service request', path)
                service_path = matches[0]
                response_path = path.parent / 'RESPONSE.json'
                response = read(response_path)
                receipt_path = service_path.parent / 'RECEIPT.json'
                receipt = read(receipt_path)
                requests.append({'request': binding(path, root), 'response': binding(response_path, root),
                    'service_request': binding(service_path, root),
                    'service_response': binding(service_path.parent / 'RESPONSE.json', root),
                    'service_receipt': binding(receipt_path, root),
                    'all_messages_exactly_matched_to_service': True,
                    **content_facts(request['messages']),
                    'tools_returned': [t['name'] for t in response.get('tool_calls', [])],
                    'completion_tokens': receipt.get('completion_tokens'),
                    'generation_seconds': receipt.get('generation_seconds')})
                for tool in response.get('tool_calls', []):
                    if tool['name'] == 'write_file' and Path(tool['arguments']['path']).name == report_path.name:
                        writes.append(tool['arguments']['content'])
            assert requests and writes and writes[-1].encode() == report_path.read_bytes(), 'Report write not bound'
            verifier_paths = verifiers.get(case_id, [])
            assert len(verifier_paths) == 1, 'Expected one current-case verifier request'
            verifier_path = verifier_paths[0]
            verifier_response_path = verifier_path.parent / 'RESPONSE.json'
            verifier_raw = read(verifier_response_path)['choices'][0]['message']['content']
            verifier_output = json.loads(verifier_raw)
            visual_lines = [{'line': i, 'text': line} for i, line in enumerate(text.splitlines(), 1)
                            if VISUAL_QUOTE in line]
            episodes = []
            for session in case['sessions']:
                episode_path = workspace / session['result']['artifact_dir'] / 'episode.json'
                episode = read(episode_path)
                episodes.append({'source': binding(episode_path, root), 'session_id': session['session_id'],
                    'success': episode['success'], 'num_steps': episode['num_steps'],
                    'error_code': episode.get('error_code'), 'error_message': episode.get('error_message')})
            native = case['native_terminal_by_session']
            windows = [e for e in events if e.get('event') == 'feedback_window_choice' and e.get('case') == case_id]
            vla_calls = sum(e.get('event') == 'actual_policy_input' and e.get('case') == case_id for e in events)
            facts = {'case_result': binding(case_dir / 'RESULT.json', root), 'cost_events': binding(events_path, root),
                'original_case_status': case['status'], 'original_chain_status': case['chain_status'],
                'original_terminal_report_status': terminal['status'],
                'environment_success': case['environment_success'],
                'semantic_verdicts': case['evidence_layers']['semantic_verdicts'],
                'native_terminal_by_session': native, 'episodes': episodes,
                'native_executed_action_steps': sum(v['steps'] for v in native.values()),
                'episode_steps_match_native_terminal': all(e['num_steps'] == native[e['session_id']]['steps'] for e in episodes),
                'actual_vla_calls': vla_calls, 'feedback_window_choice_events': len(windows),
                'terminal_agent_calls': len(requests),
                'scope': 'Observed runtime/action/call evidence remains valid independently of report prose completeness. '
                         'Native steps exclude reset settling actions; verifier agreement is not independent validation.'}
            rows.append({'arm': arm, 'case_id': case_id, 'phase': case['case']['mode'],
                'report': {**binding(report_path, root), 'full_text': text},
                'source_results_requirement': requirement,
                'review_status': 'MANUALLY_REVIEWED_EXACT_BYTES' if reviewed else 'MANUAL_REVIEW_REQUIRED_CHANGED_SOURCE',
                'completeness': manual_findings(arm, case_id) if reviewed else None,
                'actual_report_requests': requests,
                'report_requests_all_text_only': all(r['text_only'] for r in requests),
                'visual_sentence_source': {'report_lines': visual_lines,
                    'verifier_request': binding(verifier_path, root),
                    'verifier_response': binding(verifier_response_path, root),
                    'verifier_request_content': content_facts(read(verifier_path)['messages']),
                    'exact_sentence': VISUAL_QUOTE,
                    'sentence_present_in_report': bool(visual_lines),
                    'sentence_exactly_present_in_verifier_evidence': VISUAL_QUOTE in verifier_output.get('evidence', [])},
                'runtime_evidence': facts,
                'additional_configuration_scope_review': {
                    'status': 'MANUAL_REVIEW_CONFIGURED_VS_EFFECTIVE_WINDOW' if arm == 'P' and case_id != 'continuous_2_t0_i0' else 'NOT_FLAGGED',
                    'basis': 'P development reports list Replan Every Steps: 5 from native Session configuration; '
                             'the experiment case feedback_window_mode is fixed10. Reported configuration '
                             'must not be read as the actual action commitment window.',
                    'case_feedback_window_mode': case['case']['feedback_window_mode'],
                    'native_session_replan_every_steps': [s['execution']['replan_every_steps'] for s in case['sessions']]}})
    assert {(row['arm'], row['case_id']) for row in rows} == set(REVIEWED), 'Audit scope must be exactly all six reports'
    reviewed_rows = [r for r in rows if r['completeness'] is not None]
    return {'version': 'continuous_report_completeness_audit_v1', 'campaign': RUN,
        'audit_script': binding(Path(__file__).resolve(), root),
        'status': 'COMPLETED_WITH_REPORTING_GAPS' if len(reviewed_rows) == 6 else 'MANUAL_REVIEW_REQUIRED',
        'source_skill': requirement_binding(root / 'third_party/PhyAgentOS-core/PhyAgentOS/skills/benchmarking/SKILL.md', root),
        'scope': 'Narrow archival text-completeness and attribution audit; no image rejudging, model calls, '
                 'runtime status changes, report edits or retroactive admission decisions.',
        'summary': {'reports': len(rows), 'manually_reviewed_exact_report_bytes': len(reviewed_rows),
            'reports_missing_explicit_budget_episode_attempt_recovery_values': len(reviewed_rows),
            'reports_with_text_only_report_requests': sum(r['report_requests_all_text_only'] for r in rows),
            'reports_repeating_verifier_visual_sentence': sum(r['visual_sentence_source']['sentence_exactly_present_in_verifier_evidence']
                and r['visual_sentence_source']['sentence_present_in_report'] for r in rows),
            'first_hand_visual_claim': 'MANUAL_REVIEW; Verification headings make attribution ambiguous, not automatically false.',
            'environment_successes_preserved': sum(r['runtime_evidence']['environment_success'] is True for r in rows),
            'original_chain_statuses_preserved': [r['runtime_evidence']['original_chain_status'] for r in rows]},
        'conclusions': ['All six archived reports omit explicit verifier budget and episode/attempt/recovery counts.',
            'All report calls are text-only; visual sentences originate from the separately image-bearing verifier.',
            'Report PASSED checks do not certify complete Results fields or independently observed visual prose.',
            'Environment/action/call evidence and original PASSED records are preserved; these text gaps do not '
            'turn successful execution into failure or automatically invalidate accepted workspace snapshots.',
            'Do not directly promote a successful report-writing trace to independently verified visual knowledge.'],
        'reports': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.project)
    with args.output.open('x') as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps({'status': result['status'], 'summary': result['summary']}, ensure_ascii=False))

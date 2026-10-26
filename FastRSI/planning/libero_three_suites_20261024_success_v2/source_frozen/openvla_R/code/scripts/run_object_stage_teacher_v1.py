"""Four teacher fact calls; no hidden facts, outcome labels or IDs in requests."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

FACTS = ('correct_object_visible', 'on_correct_task_target', 'supported', 'released')
VALUES = ('true', 'false', 'unknown')
ORDER = ('previous_front', 'previous_wrist', 'current_front', 'current_wrist')


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def save(p, x): p.write_text(json.dumps(x, ensure_ascii=False, indent=2))


def horizontal_view(blob):
    """Policy raw[::-1,::-1] -> native human-video raw[::-1], no new evidence."""
    from fastrsi.decision_corpus import png_pixels
    from stage_trace import png_bytes
    pixels, (height, width, channels) = png_pixels(blob)
    if channels != 3: raise ValueError('Existing RGB required')
    rows = [[pixels[(y * width + x) * 3:(y * width + x + 1) * 3]
        for x in reversed(range(width))] for y in range(height)]
    return png_bytes(rows)


def make_request(record, project, model, *, current_only=False, human_upright=False):
    if human_upright and not current_only: raise ValueError('Bound current-view transform required')
    from fastrsi.object_stage_memory_v1 import request as stage_request
    state = record['request']['state']
    expected = stage_request(state['subject'], state['robot_state'], state['previous_robot_state'])
    if record['request'] != expected or len(record['images']) != 4:
        raise ValueError('Changed complete legal evidence schema')
    parts = [dict(type='text', text=json.dumps(dict(state=state,
        question='Report the CURRENT physical state of the named mug. Previous views are context only.'), allow_nan=False))]
    for view, ref in zip(ORDER, record['images']):
        path = project / ref['path']
        if not path.resolve().is_relative_to(project.resolve()) or sha(path) != ref['sha256']:
            raise ValueError('Saved image outside project or changed')
        parts.append(dict(type='text', text=view))
        blob = path.read_bytes()
        if human_upright: blob = horizontal_view(blob)
        parts.append(dict(type='image_url', image_url=dict(
            url='data:image/png;base64,' + base64.b64encode(blob).decode())))
    properties = {k: dict(type='string', enum=list(VALUES)) for k in FACTS}
    if current_only:
        current_state = {k: v for k, v in state.items() if k != 'previous_robot_state'}
        parts = [dict(type='text', text=json.dumps(dict(state=current_state,
            question='Report only the CURRENT named mug, using current front and current wrist views.'), allow_nan=False))] + parts[5:]
    properties['evidence'] = dict(type='array', items=dict(type='string', maxLength=512), maxItems=4)
    return dict(model=model, temperature=0, max_tokens=384, tool_choice='required',
        messages=[dict(role='system', content=
            'Use report_object_facts exactly once. Judge only the CURRENT named mug from the legal camera '
            'views and robot state. The task goal is not evidence of completion. Correct target means the '
            'task-designated plate, not simply the plate on the left/right side of a camera image. '
            'Supported means resting on that plate, not floating above it while held. Released means the '
            'mug is no longer held by the fingers; gripper commands or robot numbers alone do not prove this. '
            'A finger visible near a mug is not by itself evidence of holding it. '
            'Use unknown if identity, target, support or release cannot be resolved. Distinguish visible '
            'failure (false) from insufficient evidence (unknown). Briefly state the visual evidence. '
            'Your report is a fallible diagnosis, not a benchmark label, execution authorization or memory update.'),
            dict(role='user', content=parts)],
        tools=[dict(type='function', function=dict(name='report_object_facts',
            description='Report four current physical facts, retaining uncertainty.',
            parameters=dict(type='object', properties=properties,
                required=list(FACTS) + ['evidence'], additionalProperties=False)))])


def parse_response(reply):
    if len(reply.get('choices', [])) != 1: raise ValueError('Expected one completion')
    choice = reply['choices'][0]; calls = choice['message'].get('tool_calls', [])
    if (choice['finish_reason'] not in ('stop', 'tool_calls') or len(calls) != 1
            or calls[0]['function']['name'] != 'report_object_facts'):
        raise ValueError('No complete typed fact call')
    answer = json.loads(calls[0]['function']['arguments'])
    if (set(answer) != set(FACTS) | {'evidence'} or any(answer[k] not in VALUES for k in FACTS)
            or not isinstance(answer['evidence'], list) or len(answer['evidence']) > 4
            or any(not isinstance(v, str) or len(v) > 512 for v in answer['evidence'])):
        raise ValueError('Illegal or unbound candidate facts')
    return answer


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--project', type=Path, required=True); args = p.parse_args()
    reg = read(args.run / 'REGISTRATION.json'); source = args.run / 'inputs/INPUTS.json'
    if sha(source) != reg['input_sha256']: raise ValueError('Changed registered input')
    records = read(source)
    if [r['id'] for r in records] != reg['identities'] or len(records) != 4:
        raise ValueError('Changed bounded request list')
    out = args.run / 'diagnosis'; out.mkdir(exist_ok=False)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    start = time.perf_counter(); rows = []
    for record in records:
        folder = out / record['id']; folder.mkdir()
        request = make_request(record, args.project, read(args.run / 'MODEL_CONFIG.json')['model_id'],
            current_only=reg.get('view_mode') in ('current_only', 'human_upright_current'),
            human_upright=reg.get('view_mode') == 'human_upright_current')
        blob = json.dumps(request, allow_nan=False).encode(); (folder / 'REQUEST.json').write_bytes(blob)
        tick = time.perf_counter(); row = dict(id=record['id'], request_sha256=hashlib.sha256(blob).hexdigest(),
            authority=reg['authority'], controller_changed=False, label=None)
        try:
            req = urllib.request.Request('http://127.0.0.1:49510/v1/chat/completions', data=blob,
                headers={'Content-Type': 'application/json'})
            with opener.open(req, timeout=150) as response: reply = json.load(response)
            save(folder / 'RESPONSE.json', reply)
            answer = parse_response(reply)
            stage = 'complete' if all(answer[k] == 'true' for k in FACTS) else (
                'incomplete' if any(answer[k] == 'false' for k in FACTS) else 'unknown')
            row.update(status='TYPED_DIAGNOSIS', facts=answer, proposed_stage=stage,
                usage=reply.get('usage'), service_call_id=reply['id'])
        except urllib.error.HTTPError as exc:
            (folder / 'ERROR_RESPONSE.json').write_bytes(exc.read())
            row.update(status='REJECTED', error=str(exc))
        except Exception as exc: row.update(status='FAILED', error=repr(exc))
        row['seconds'] = time.perf_counter() - tick; rows.append(row)
        save(out / 'RESULT.json', dict(status='RUNNING', rows=rows))
    # Private simulator predicates are read only after all calls, for mismatch audit.
    offline = {r['id']: r for r in read(args.run / 'inputs/OFFLINE_AUDIT_ONLY.json')}
    for row in rows:
        row['offline_audit'] = offline[row['id']]
        row['audit_complete_matches'] = None if row['status'] != 'TYPED_DIAGNOSIS' else (
            (row.get('proposed_stage') == 'complete') == offline[row['id']]['current_complete_proxy'])
        row['reliable_training_label_created'] = False
    save(out / 'RESULT.json', dict(status='COMPLETED', rows=rows,
        seconds=time.perf_counter() - start, teacher_calls=len(rows), new_robot_episodes=0,
        persistent_updates=0, live_controller_changed=False, verified_training_labels=0))


if __name__ == '__main__': main()

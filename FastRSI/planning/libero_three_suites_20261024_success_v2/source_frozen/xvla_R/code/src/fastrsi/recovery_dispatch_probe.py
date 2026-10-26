"""Offline prompt contrast only; never a runtime verdict override."""
import copy
import json


def bundle(body):
    messages = body.get('messages', [])
    if len(messages) != 2 or [m.get('role') for m in messages] != ['system', 'user']:
        raise ValueError('Expected the complete two-message native verification request')
    parts = messages[1]['content']
    text = parts[0].get('text', '')
    if 'VERIFICATION_BUNDLE:\n' not in text:
        raise ValueError('Missing native evidence bundle')
    value = json.loads(text.split('VERIFICATION_BUNDLE:\n', 1)[1])
    if value.get('version') != 'agent_session_verification_v2':
        raise ValueError('Unexpected source endpoint')
    if len([p for p in parts if p.get('type') == 'image_url']) != 4:
        raise ValueError('Expected two initial and two final images')
    return value


def prepare(body, variant, session_prompt, episode_prompt):
    bundle(body)
    original = body['messages'][0]['content']
    if not original.startswith(session_prompt):
        raise ValueError('Source system prompt differs from bound native source')
    if variant not in ('session', 'episode_prompt'):
        raise ValueError('Unknown contrast')
    result = copy.deepcopy(body)
    if variant == 'episode_prompt':
        # Preserve common output-schema correction, images, full history,
        # generation settings and original goal. This is NOT an endpoint swap.
        result['messages'][0]['content'] = episode_prompt + original[len(session_prompt):]
    return result


def summarize_verdict(response):
    choices = response.get('choices', [])
    if len(choices) != 1 or choices[0].get('finish_reason') != 'stop':
        raise ValueError('Truncated or invalid completion')
    value = json.loads(choices[0]['message']['content'])
    verdict = value.get('verdict')
    if verdict not in ('success', 'replan', 'failure'):
        raise ValueError('Unknown verdict')
    evidence = value.get('evidence')
    if not isinstance(evidence, list) or not evidence or any(not isinstance(x, str) or not x.strip() for x in evidence):
        raise ValueError('Missing evidence')
    for field in ['lesson'] + (['replan_task_description'] if verdict == 'replan' else ['failure_reason'] if verdict == 'failure' else []):
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise ValueError('Missing ' + field)
    return value

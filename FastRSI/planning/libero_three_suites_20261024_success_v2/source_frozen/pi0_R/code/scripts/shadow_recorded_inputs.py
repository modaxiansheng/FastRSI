"""Offline routing probe on metadata from real requests; no visual or outcome labels."""
import hashlib
import json
import statistics
import time
from pathlib import Path
import torch
from laya import Router
from fastrsi.contracts import reuse_admissible

ROOT = Path('/project/runs/pilot_20261001')
OUT = ROOT / 'LAYA_RECORDED_SHADOW_V1.json'
assert not OUT.exists(), 'Never overwrite a recorded run'
torch.set_num_threads(4)
model = '/project/models/huggingface/hub/models--convaiinnovations--laya-typed-decisions/snapshots/1a793eb568e6718f15941d08f85432581df534e3'
started = time.perf_counter()
router = Router(device='cpu', models={'typed-decisions': model}, standalone_repos=True)
questions = {'route': {'type': 'choice',
    'instructions': 'Can an existing verified recovery be reused with this evidence?',
    'criteria': {'reuse': 'An applicable verified recovery and sufficient current evidence exist',
                 'escalate': 'Recovery is unverified or required current evidence is missing'}}}
rows = []
for path in sorted((ROOT / 'model_calls').glob('*.request.json')):
    request = json.loads(path.read_text())
    texts = [part['text'] for message in request['messages']
             if isinstance(message.get('content'), list) for part in message['content']
             if part.get('type') == 'text' and 'VERIFICATION_BUNDLE:\n' in part.get('text', '')]
    bundle = json.loads(texts[0].split('VERIFICATION_BUNDLE:\n', 1)[1])
    state = {'task': bundle['session']['task_description'],
             'prior_lesson_text': bundle.get('lessons_md', ''),
             'candidate_recovery': None, 'validated_recovery_records': [],
             'fresh_visual_summary_available': False}
    tick = time.perf_counter()
    answer = router.predict(state, questions, model='typed-decisions')
    rows.append({'request_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                 'source_request': path.name, 'state': state, 'answer': answer,
                 'seconds': time.perf_counter() - tick,
                 'reuse_admissibility': reuse_admissible(None, {}, None),
                 'structural_label': 'escalate', 'robot_outcome_label': None})
    print(json.dumps(rows[-1], default=str), flush=True)
report = {'scope': 'Offline metadata-only shadow replay of two real OS verification requests. '
          'RGB, runtime outcomes and current Qwen replies excluded. No robot control. '
          'No verified recovery candidates exist, so labels test admissibility only; '
          'this is not recovery effectiveness or general routing accuracy.',
          'status': 'RECORDED_SHADOW_EXECUTED_QUALITY_UNESTABLISHED', 'rows': rows,
          'total_seconds': time.perf_counter() - started}
with OUT.open('x') as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2, default=str)

"""Load/latency smoke only; synthetic input is not a robotics quality result."""
import json
import os
import statistics
import time
import traceback
from pathlib import Path

OUT = Path('/project/runs/pilot_20261001')
started = time.perf_counter()
report = {'scope': 'synthetic load smoke; no robot control or quality claim',
          'code_commit': '6d942c92081fbc139e736bbd9ac0023223c29b7f',
          'model': 'convaiinnovations/laya-typed-decisions',
          'revision': '1a793eb568e6718f15941d08f85432581df534e3', 'device': 'cpu'}
try:
    import torch
    torch.set_num_threads(4)
    from laya import Router
    local_model=os.environ.get('FASTRSI_LAYA_MODEL_PATH')
    router = Router(device='cpu', standalone_repos=True,
                    models={'typed-decisions':local_model} if local_model else None,
                    revisions={'typed-decisions': report['revision']})
    state = {'task': 'pick up a bowl and place it on a plate',
             'candidate_recovery': None, 'validated_recovery_records': [],
             'fresh_visual_summary_available': False}
    questions = {'route': {'type': 'choice',
        'instructions': 'Can an existing verified recovery be reused with this evidence?',
        'criteria': {'reuse': 'An applicable verified recovery and sufficient current evidence exist',
                     'escalate': 'Recovery is unverified or required current evidence is missing'}}}
    timings = []
    answers = []
    for i in range(6):
        tick = time.perf_counter()
        answer = router.predict(state, questions, model='typed-decisions')
        timings.append(time.perf_counter()-tick)
        answers.append(answer)
        print(json.dumps({'call':i,'seconds':timings[-1],'answer':answer},default=str),flush=True)
    report.update(status='LOAD_SMOKE_PASSED', first_call_seconds=timings[0],
                  warm_median_seconds=statistics.median(timings[1:]),
                  call_seconds=timings, answers=answers)
except Exception:
    report.update(status='FAILED', error=traceback.format_exc())
    raise
finally:
    report['total_seconds'] = time.perf_counter()-started
    OUT.mkdir(parents=True,exist_ok=True)
    destination=OUT/('LAYA_LOAD_SMOKE-'+os.environ.get('HOSTNAME','local')+'.json')
    with destination.open('x') as handle:
        json.dump(report,handle,ensure_ascii=False,indent=2,default=str)

"""Run the existing budget rule on a sealed chronological development catalog."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fastrsi.budget_opportunity import replay
from build_budget_history_refresh import Sources, inspect_arm


def main():
    started = time.perf_counter()
    out = ROOT / 'planning/budget_opportunity_v3_20261004'
    out.mkdir(exist_ok=False)
    config = dict(remaining_uses=1, max_quality_followups=1,
                  allow_quality_confirmation=True, remaining_budget_seconds=1700,
                  estimated_pair_seconds=750, maximum_pair_seconds=1700)
    protocol = {'config': config, 'purpose': 'Audit the existing allocation rule, including missed opportunities.',
                'cost_scope': 'Reference arm lifecycle sums only; no current deployment speedup.',
                'data_scope': 'Already seen development data. No final evaluation feedback.',
                'novelty': 'No new stopping thresholds, model training or live promotion.'}
    (out / 'PROTOCOL.json').write_text(json.dumps(protocol, indent=2))
    sources = Sources(ROOT)
    old = sources.json('planning/budget_history_refresh_v1_20261003/CATALOG.json')
    for path, expected in old['source_sha256'].items():
        if sources.binding(path) != expected: raise ValueError('Historical source changed: '+path)
    records = list(old['records'])
    for run, index in [('learning_transfer_v1_20261004', 0),
                       ('learning_transfer_v2_20261004', 1),
                       ('learning_transfer_v2_20261004', 2),
                       ('learning_transfer_v2_20261004', 3)]:
        path = 'runs/'+run+'/development/pair'+str(index)
        arms = [inspect_arm(sources, path+'/campaign_'+arm+'/'+arm, mode, False)
                for arm, mode in [('N', 'fixed10'), ('F', 'fixed5')]]
        a,b = arms
        if a['initial_image_sha256'] != b['initial_image_sha256'] or a['model_config_sha256'] != b['model_config_sha256']:
            raise ValueError('Unmatched new pair')
        for key in ('suite','task_id','init_state_id','task_description','max_steps'):
            if a['case'][key] != b['case'][key]: raise ValueError('Changed task')
        records.append({'evidence_id': path, 'arms': arms,
                        'observed_at': max(a['finished_at'],b['finished_at']),
                        'spec': {'protocol_group': 'learning_transfer_resident_v2_20261004'}})
    rows = []
    for r in records:
        a,b = r['arms']; case = a['case']
        paths = [arm['path']+suffix for arm in (a,b)
                 for suffix in ('/STATUS.json','/stream/RESULT.json','/MODEL_CONFIG.json')]
        rows.append({'id':r['evidence_id'], 'phase':case['mode'], 'goal':case['task_description'],
                     'task':case['task_id'], 'init':case['init_state_id'],
                     'observed_at':r['observed_at'], 'model_sha256':a['model_config_sha256'],
                     'protocol_group':r['spec']['protocol_group'],
                     'native_success':a['native_success'], 'candidate_success':b['native_success'],
                     'native_seconds':a['full_wall_seconds'], 'candidate_seconds':b['full_wall_seconds'],
                     'semantic_aligned':all(x['semantic_alignment']=='aligned' for x in (a,b)),
                     'admission':[x['admission'] for x in (a,b)],
                     'sources':{p:sources.binding(p) for p in paths}})
    rows.sort(key=lambda r:r['observed_at'])
    sources.verify()
    result = replay(rows, config)
    result['wall_seconds'] = time.perf_counter()-started
    result['source_rows'] = len(rows)
    result['independent_task_groups'] = len({(r['goal'],r['model_sha256']) for r in rows})
    result['code_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (Path(__file__),ROOT/'src/fastrsi/budget_opportunity.py',ROOT/'src/fastrsi/window_budget_v2.py')}
    for name, obj in [('INPUTS.json', rows), ('SOURCE_MANIFEST.json',sources.manifest), ('RESULT.json',result)]:
        (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='policies'},ensure_ascii=False,indent=2))
    for name,r in result['policies'].items():
        print(name,json.dumps({k:v for k,v in r.items() if k!='trace'},ensure_ascii=False))


if __name__=='__main__': main()

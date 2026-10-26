"""Cheap task-level development learner. No init IDs or test outcomes in selection."""
import hashlib, json, math, statistics
from pathlib import Path
VERSION = 'task_window_v1'
def key(goal):
    return hashlib.sha256(' '.join(goal.lower().split()).encode()).hexdigest()
def fit(pairs, *, phase):
    if phase != 'development': raise ValueError('Only development feedback may update')
    grouped = {}; seen = set()
    for p in pairs:
        if p['phase'] != 'development' or p['evidence_id'] in seen: raise ValueError('Invalid or duplicate evidence')
        seen.add(p['evidence_id'])
        if not p['sources'] or any(len(v)!=64 for v in p['sources'].values()): raise ValueError('Unbound feedback')
        for mode in ('fixed5','fixed10'):
            a=p[mode]
            if type(a['native_success']) is not bool or not math.isfinite(a['seconds']) or a['seconds']<=0: raise ValueError('Invalid numeric fact')
        grouped.setdefault(key(p['goal']), []).append(p)
    table={}
    for k, rows in grouped.items():
        # A task-level empirical safety preference, not a noninferiority claim.
        wins=sum(p['fixed5']['native_success'] and not p['fixed10']['native_success'] for p in rows)
        harms=sum(p['fixed10']['native_success'] and not p['fixed5']['native_success'] for p in rows)
        matched=[p for p in rows if p['cost_compatible'] and p['fixed5']['native_success'] and p['fixed10']['native_success']]
        gains=[p['fixed10']['seconds']-p['fixed5']['seconds'] for p in matched]
        mean=statistics.mean(gains) if gains else None
        se=statistics.stdev(gains)/math.sqrt(len(gains)) if len(gains)>1 else None
        mode='fixed10';reason='retain_default_insufficient_consistent_benefit'
        if wins and not harms: mode='fixed5';reason='historical_native_quality_advantage_requires_fresh_check'
        elif not harms and len(gains)>=2 and mean-2*se>0: mode='fixed5';reason='positive_matched_task_cost_heuristic'
        table[k]={'goal':rows[0]['goal'],'selected_mode':mode,'reason':reason,'native_quality_wins':wins,'native_quality_harms':harms,'cost_pairs':len(gains),'mean_saving_seconds':mean,'two_se_heuristic':2*se if se is not None else None,'evidence_ids':[p['evidence_id'] for p in rows]}
    return {'version':VERSION,'default':'fixed10','table':table,'pairs':pairs,'promotion_authority':False,'limitations':'Development diagnostic; two-SE heuristic is not a calibrated interval; native facts do not promote conflicting narrative experience.'}
def select(state, goal):
    if state['version']!=VERSION: raise ValueError('Wrong schema')
    k=key(goal);row=state['table'].get(k)
    return {'selected_mode':row['selected_mode'] if row else state['default'],'goal_key':k,'reason':row['reason'] if row else 'unseen_task_default','evidence_ids':row['evidence_ids'] if row else []}
def load(path, sha):
    data=Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=sha: raise ValueError('Task table changed')
    state=json.loads(data)
    if state['version']!=VERSION or state['default']!='fixed10': raise ValueError('Invalid table')
    if fit(state['pairs'],phase='development')!=state: raise ValueError('Table does not match evidence')
    return state

"""Predeclared task-table learning on real paired actions; no invented labels."""
import copy,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fastrsi import task_window


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')

def group_record(receipt, *, phase, index, arm, path, reused=False):
    if receipt.get('arm')!=arm or receipt.get('status')!='COMPLETED':
        raise ValueError('Worker receipt is incomplete or belongs to another arm')
    return {**receipt,'phase':phase,'index':index,'arm':arm,'path':path,'reused':reused}


def history(root):
    catalog=read(root/'planning/budget_history_refresh_v1_20261003/CATALOG.json');pairs=[];excluded=[];seen=set()
    for record in catalog['records']:
        key=record['independent_instance_key'];arms={a['mode']:a for a in record['arms']}
        for a in arms.values():
            for rel,h in a['sources'].items():
                if sha(root/rel)!=h:raise ValueError('History bytes changed: '+rel)
        if key in seen or any(a['semantic_alignment']!='aligned' or a['admission']['status']!='ADMITTED' for a in arms.values()):
            excluded.append(dict(source=record['evidence_id'],reason='Repeated initial condition or unresolved semantic/admission status'));continue
        seen.add(key);case=arms['fixed10']['case'];sources={k:v for a in arms.values() for k,v in a['sources'].items()}
        p=dict(goal=case['task_description'],task=case['task_id'],init=case['init_state_id'],phase='development',evidence_id=record['evidence_id'],cost_compatible=False,sources=sources)
        for mode,a in arms.items():p[mode]=dict(native_success=a['native_success'],seconds=a['full_wall_seconds'],vla_calls=a['vla_calls'],steps=a['steps'])
        pairs.append(p)
    return pairs,excluded


def prepare(root,out):
    out.mkdir(parents=True,exist_ok=False);pairs,excluded=history(root);save(out/'FROZEN.json',task_window.fit(pairs,phase='development'));save(out/'EXCLUDED_HISTORY.json',excluded)
    goals={a['case']['task_id']:a['case']['task_description'] for r in read(root/'planning/budget_history_refresh_v1_20261003/CATALOG.json')['records'] for a in r['arms']};reserved=read(root/'planning/representative_coverage_v1/RESERVED_SPLIT.json')
    finals={r['task_id']:r['reserved_final_method_test_init_ids'] for r in reserved['tasks']}
    exposed=set()
    for p in (root/'runs').glob('**/stream/RESULT.json'):
        for r in read(p).get('cases',[]):
            c=r.get('case',{});exposed.add((c.get('suite'),c.get('task_id'),c.get('init_state_id')))
    def case(t,i,phase):
        if i in finals[t] or ('libero_10',t,i) in exposed:raise ValueError('Instance is reserved or already used: '+str((t,i)))
        return dict(suite='libero_10',task_id=t,init_state_id=i,task_description=goals[t],mode=phase,max_steps=520)
    protocol=dict(name='learning_transfer_v1_20261004',status='PREDECLARED',
        development=[case(t,i,'development') for t,i in [(3,1),(8,47),(3,2),(8,0)]],
        evaluation=[case(t,i,'evaluation') for t,i in [(3,3),(8,1)]],maximum_campaign_seconds=3000,maximum_episodes=14,
        controls={'N':'Global fixed window selected on common development evidence, then frozen','F':'Existing task-table architecture, updated with new reliable paired development evidence','R':'Same task-table architecture, frozen before new evidence'},
        objective='New experience changes task-level execution allocation on fresh development holdout instances; not a new window search, three-way router, or final benchmark',
        data='All methods have access to identical historical and newly acquired development records. No evaluation outcomes used to fit.',
        learning='Unmodified task_window.fit; no threshold tuning. Old costs excluded; current paired full worker lifecycle costs used. Native/semantic conflicts not admitted as update labels.',
        fixed_rule='Minimize observed failures over independent reliable history plus new development; tie-break by summed new-development lifecycle cost; final tie fixed10.',
        evaluation_isolation='Native evaluation mode, readable fixed OS experience, no persistent learning, frozen SHA for task tables.',
        task_selection='Drawer supplies cost heterogeneity; two-moka task retains prior short-window harm as retention check. Selection is developer informed, not random benchmark coverage.',
        starting_skill_snapshot='Existing same admitted OS snapshot copied per episode; skill evolution held fixed for this mechanism comparison.',
        finite_transfer_cases=2,noninferiority_claim=False,production_promotion=False,clef_training=False)
    save(out/'PROTOCOL.json',protocol);save(out/'HISTORY_COST.json',dict(known_pair_lifecycle_seconds=sum(p[m]['seconds'] for p in pairs for m in ('fixed5','fixed10')),scope='Accepted independent source pairs only; complete project expenses remain in historical ledgers; not free cold-start learning.'))
    return protocol


def learn(frozen,new_pairs):
    tick=time.perf_counter();eligible=[];rejected=[]
    for p in new_pairs:
        good=all(p[m]['semantic_verdicts']==(['success'] if p[m]['native_success'] else ['failure']) and p[m]['admission']=='ADMITTED' for m in ('fixed10','fixed5'))
        (eligible if good else rejected).append(p)
    all_pairs=frozen['pairs']+eligible
    updated=task_window.fit(all_pairs,phase='development')
    score={m:(sum(not p[m]['native_success'] for p in all_pairs),sum(p[m]['seconds'] for p in eligible),m!='fixed10') for m in ('fixed10','fixed5')}
    fixed=min(score,key=score.get)
    return updated,dict(fixed_mode=fixed,score=score,accepted_new=[p['evidence_id'] for p in eligible],rejected_new=[p['evidence_id'] for p in rejected],update_seconds=time.perf_counter()-tick,learning_gain_established=False)

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1];print(json.dumps(prepare(root,root/'planning/learning_transfer_v1_20261004'),indent=2))

"""Bind visual inputs and separate native supervision from historical development."""
import hashlib,json,sys,time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fastrsi.completion_decision import request,partition,signature
from fastrsi.decision_corpus import png_pixels


def main():
    started=time.perf_counter();out=ROOT/'planning/completion_learning_v1_20261004';out.mkdir(exist_ok=False)
    sources={}
    def read(path,expected=None):
        path=Path(path);data=(ROOT/path).read_bytes();h=hashlib.sha256(data).hexdigest()
        if expected and h!=expected:raise ValueError('Changed evidence: '+str(path))
        sources[str(path)]=h;return data
    paid=json.loads(read('planning/budget_opportunity_v3_20261004/INPUTS.json'))
    groups=[];first=[];seen=set()
    for r in paid:
        group=r['goal']+'/init'+str(r['init'])
        if group in seen:continue
        seen.add(group);first.append(r)
        groups.append(dict(group=group,goal=r['goal'],observed_at=r['observed_at']))
    roles=partition(groups);inputs=[];labels=[];duplicates=[];by_signature={}
    # Consume seed/update first for collision detection without looking at labels.
    first.sort(key=lambda r:({'seed':0,'update':1,'test':2}[roles[r['goal']+'/init'+str(r['init'])]],r['observed_at']))
    for r in first:
        group=r['goal']+'/init'+str(r['init']);role=roles[group]
        for path,sha in r['sources'].items():read(path,sha)
        results=[p for p in r['sources'] if p.endswith('/stream/RESULT.json')]
        if len(results)!=2:raise ValueError('Expected paired original arms')
        for result_path in results:
            result=json.loads(read(result_path));row=result['cases'][0];case=row['case']
            if result['status']!='COMPLETED' or row['chain_status']!='PASSED' or case['mode']!='development':
                raise ValueError('Only completed development evidence')
            stream=Path(result_path).parent;trace_path=stream/'stage_trace'/case['id']/'TRACE.jsonl'
            trace=[json.loads(x) for x in read(trace_path).splitlines() if x]
            if trace[-1]['native_success']!=row['environment_success']:raise ValueError('Terminal label mismatch')
            indices=sorted({0,len(trace)//2,max(0,len(trace)-2),len(trace)-1})
            for index in indices:
                event=trace[index]
                if type(event.get('native_success')) is not bool:raise ValueError('No instantaneous native fact')
                images=[]
                for camera in ('agentview_image','robot0_eye_in_hand_image'):
                    name=f"{event['step']:04d}_{camera}.png";image_path=trace_path.parent/name
                    raw=read(image_path,event['images'][name]);pixels,shape=png_pixels(raw)
                    images.append(dict(path=str(image_path),sha256=sources[str(image_path)],
                                       pixel_sha256=hashlib.sha256(pixels).hexdigest(),shape=shape))
                sig=signature(case['task_description'],[i['pixel_sha256'] for i in images])
                sample_id=hashlib.sha256((str(trace_path)+':'+str(index)).encode()).hexdigest()
                if sig in by_signature:
                    previous=by_signature[sig]
                    if previous['label']!=event['native_success']:raise ValueError('Identical input conflicting native labels')
                    duplicates.append(dict(id=sample_id,duplicate_of=previous['id'],role=role,
                                           cross_role=previous['role']!=role));continue
                by_signature[sig]=dict(id=sample_id,role=role,label=event['native_success'])
                inputs.append(dict(id=sample_id,request=request(case['task_description']),images=images,input_sha256=sig))
                labels.append(dict(id=sample_id,group=group,goal=case['task_description'],role=role,
                    native_complete=event['native_success'],source_trace=str(trace_path),trace_index=index,
                    source_result=result_path,is_terminal=index==len(trace)-1,source_step=event['step'],
                    label_scope='Instantaneous native development goal predicate; not visual certainty or a narrative label'))
    protocol=dict(status='PREPARED',model_revision='17f0b0ad64efb65d273590632833508766b2aae6',
        inputs='Current two RGB views and goal only; no step, terminal flag, actions, init ID or native label.',
        split='Chronological first initial per task seed; last held out; intermediate update. Retrospective development, not final benchmark.',
        feature='Frozen Clef joint head field vector (1024); original probabilities also retained.',
        fit=dict(method='CPU class-balanced logistic linear head',iterations=300,learning_rate=.03,weight_decay=.01,seed=17),
        threshold=.99,threshold_scope='Fixed screening threshold, not calibrated probability or certification.',
        comparators=['upstream Clef','seed-trained frozen head','seed+new-experience head','same-feature nearest memory','always complete diagnostic'],
        runtime='Shadow only. Qwen and final independent evaluation unchanged. No skill/narrative approval.',
        controls='Qwen/VLA/Clef backbone weights frozen; same feature cost and inputs in frozen/updated arms.',
        groups=groups,roles=roles)
    summary=dict(samples=len(inputs),unique_initial_groups=len(groups),role_samples=dict(Counter(x['role'] for x in labels)),
        role_groups={k:sum(v==k for v in roles.values()) for k in ('seed','update','test')},
        positives={k:sum(x['native_complete'] for x in labels if x['role']==k) for k in ('seed','update','test')},
        duplicate_inputs_removed=len(duplicates),cross_role_duplicate_inputs_removed=sum(d['cross_role'] for d in duplicates),
        seconds=time.perf_counter()-started,new_robot_episodes=0)
    for name,value in [('PROTOCOL.json',protocol),('INPUTS.json',inputs),('SUPERVISION.json',labels),
                       ('SOURCE_MANIFEST.json',sources),('DUPLICATES.json',duplicates),('SUMMARY.json',summary)]:
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

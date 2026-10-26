"""Bounded, immutable experience-bank decisions from public initial RGB.

Nearest validated preference is a small nonparametric decision model. It does
not claim calibrated success probabilities. Fixed/source-nearest controls use
the same candidate bank. Training examples carry no runtime initial-state IDs.
"""
import hashlib,time
from pathlib import Path
import numpy as np
from .experience_compiler import validate_compiled,canonical_sha

def validate_bank(bank):
    if bank.get('version')!=1 or bank.get('kind')!='experience_bank':raise ValueError('Unknown bank format')
    candidates=bank.get('candidates',{})
    if not isinstance(candidates,dict) or len(candidates)>4:raise ValueError('Bank capacity exceeded')
    if not candidates and bank.get('model',{}).get('kind')!='fixed':raise ValueError('Empty bank requires fixed native fallback')
    if not candidates and bank.get('model',{}).get('choice')!='native':raise ValueError('Empty bank requires fixed native fallback')
    if 'native' in candidates:raise ValueError('Reserved fallback name')
    kind=bank['model'].get('kind')
    for key,value in candidates.items():
        if not isinstance(key,str) or not key:raise ValueError('Invalid candidate ID')
        validate_compiled(value['skill'])
        if value['skill']['source_goal']!=bank['goal']:raise ValueError('Mixed goals in bank')
        # A fixed choice never retrieves by appearance. An absent, unused
        # feature vector is explicit; fabricated zeros are not observations.
        if not (kind=='fixed' and value.get('source_features') is None):
            f=np.asarray(value['source_features'],dtype=float)
            if f.shape!=(4,) or not np.isfinite(f).all():raise ValueError('Invalid public source features')
        if 'validity' in value:
            from .skill_validity import validate_validity
            validate_validity(value['validity'],value['skill'])
    model=bank['model'];kind=model.get('kind');allowed=set(candidates)|{'native'}
    if kind=='fixed':
        if model.get('choice') not in allowed:raise ValueError('Illegal fixed choice')
    elif kind=='source_nearest':pass
    elif kind=='preference_nearest':
        examples=model.get('examples',[])
        if not examples or len(examples)>32:raise ValueError('Invalid prototype count')
        for e in examples:
            if set(e)!={'features','choice','evidence_sha256'} or e['choice'] not in allowed:raise ValueError('Illegal learned preference')
            f=np.asarray(e['features'],dtype=float)
            if f.shape!=(4,) or not np.isfinite(f).all():raise ValueError('Invalid prototype features')
            if len(e['evidence_sha256'])!=64:raise ValueError('Unbound training evidence')
    else:raise ValueError('Unknown decision model')
    ref=bank['image_reference']
    if hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest()!=ref['sha256']:raise ValueError('Reference image changed')
    return bank

def rgb_features(bank,image):
    import cv2
    from .visual_binding import match_template
    ref=bank['image_reference'];source=cv2.cvtColor(cv2.imread(ref['path']),cv2.COLOR_BGR2RGB)
    matches=match_template(source,image,ref['roi'],count=2)
    if len(matches['peaks'])!=2:raise ValueError('Two observable correspondences are required')
    peaks=sorted(matches['peaks'],key=lambda p:p['box_xyxy'][0]);h,w=image.shape[:2]
    features=[]
    for p in peaks:
        x0,y0,x1,y1=p['box_xyxy'];features.extend([(x0+x1-1)/(2*w),(y0+y1-1)/(2*h)])
    return features,matches

def select(bank,observation):
    tick=time.perf_counter();model=bank['model'];kind=model['kind'];features=None;matches=None;distances=None
    if kind=='fixed':choice=model['choice']
    else:
        features,matches=rgb_features(bank,observation['observation/image'])
        if kind=='source_nearest':
            keys=sorted(bank['candidates']);xx=[bank['candidates'][k]['source_features'] for k in keys]
        else:
            keys=[e['choice'] for e in model['examples']];xx=[e['features'] for e in model['examples']]
        distances=np.sum((np.asarray(xx)-np.asarray(features))**2,axis=1).tolist()
        choice=keys[int(np.argmin(distances))]
    if choice not in set(bank['candidates'])|{'native'}:raise ValueError('Unknown selected action')
    return dict(choice=choice,features=features,matches=matches,distances=distances,
                seconds=time.perf_counter()-tick,model_kind=kind,model_version=model['version'],
                model_sha256=canonical_sha(model),input_image_sha256=hashlib.sha256(observation['observation/image'].tobytes()).hexdigest())

def execute_bank(runtime,chunk,bank,sha,root_session,enabled,original):
    from .goal_skill import execute
    provenance=chunk.get('provenance',{})
    tag=provenance.get('goal_skill')
    check=provenance.get('goal_skill_validity');exit_tag=provenance.get('goal_skill_exit')
    if not enabled or runtime.session_id!=root_session:
        if any(x is not None for x in (tag,check,exit_tag)):raise ValueError('Bank request outside enabled root Session')
        return original(runtime,chunk),None
    cached=getattr(runtime,'_fastrsi_bank_choice',None)
    if cached is None:
        if runtime.step_idx!=0:raise ValueError('Bank choice requires initial observation')
        choice='native' if tag is None else tag.get('choice_id')
        if choice not in set(bank['candidates'])|{'native'}:raise ValueError('Unregistered bank choice')
        cached=dict(choice=choice,session=runtime.session_id,sha256=sha)
    if cached['session']!=runtime.session_id or cached['sha256']!=sha:raise ValueError('Bank version changed within Session')
    choice=cached['choice']
    if choice=='native':
        if any(x is not None for x in (tag,check,exit_tag)):raise ValueError('Cannot switch after native bank choice')
        output=original(runtime,chunk);runtime._fastrsi_bank_choice=cached
        return output,None
    candidate=bank['candidates'][choice];skill=candidate['skill'];config=candidate.get('validity')
    if cached.get('retired'):
        if any(x is not None for x in (tag,check,exit_tag)):raise ValueError('Retired skill cannot be resumed or checked again')
        return original(runtime,chunk),None
    guard_receipt=None;guard_seconds=0.
    if config is None:
        if check is not None or exit_tag is not None:raise ValueError('Validity check not configured for this candidate')
    else:
        from .skill_validity import verify_proof
        tick=time.perf_counter();boundary=config['boundary_step']
        if runtime.step_idx==boundary and not cached.get('validity_checked'):
            if (check is None)==(exit_tag is None):raise ValueError('Exactly one validity proof is required at its boundary')
            proof=check if check is not None else exit_tag
            guard_receipt=verify_proof(proof,config,bank_sha=sha,choice=choice,session=runtime.session_id,
                                      step=runtime.step_idx,target_qpos=(getattr(runtime,'_last_obs',None) or {}).get('robot0_gripper_qpos'))
            should_exit=proof['action']=='exit_to_native'
            if should_exit != (exit_tag is not None) or (should_exit and tag is not None):
                raise ValueError('Validity proof does not match the requested action path')
            if should_exit:
                actions=np.asarray(chunk.get('actions'),dtype=np.float32)
                if actions.shape!=(10,7) or not np.isfinite(actions).all():raise ValueError('Exit requires a real finite native ten-step action chunk')
                guard_seconds=time.perf_counter()-tick
                # Commit retirement before physics: even a partial native error
                # must not make the candidate executable again in this Session.
                cached={**cached,'validity_checked':True,'retired':True,'exit_step':runtime.step_idx}
                runtime._fastrsi_bank_choice=cached;begin=runtime.step_idx
                output=original(runtime,chunk)
                return output,dict(origin='native_after_validity_exit',skill_execution=False,
                    start_step=begin,end_step=runtime.step_idx,actual_actions=actions[:runtime.step_idx-begin].tolist(),
                    requested_actions=actions.tolist(),
                    requested_steps=len(actions),executed_steps=runtime.step_idx-begin,
                    bank_choice=choice,bank_sha256=sha,candidate_sha256=canonical_sha(skill),
                    validity=guard_receipt,target_guard_seconds=guard_seconds,permanently_retired=True)
        else:
            if check is not None or exit_tag is not None:raise ValueError('Validity proof outside its one permitted boundary')
            if runtime.step_idx>boundary and not cached.get('validity_checked'):raise ValueError('Required validity boundary was skipped')
        guard_seconds=time.perf_counter()-tick
    if tag is not None:
        if tag.get('choice_id')!=choice:raise ValueError('Bank candidate changed within Session')
        clean=dict(tag);clean.pop('choice_id')
        chunk={**chunk,'provenance':{**chunk.get('provenance',{}),'goal_skill':clean}}
    output,receipt=execute(runtime,chunk,skill,sha,root_session,True,original)
    if guard_receipt is not None:cached={**cached,'validity_checked':True}
    runtime._fastrsi_bank_choice=cached
    if receipt is not None:receipt.update(bank_choice=choice,bank_sha256=sha,candidate_sha256=canonical_sha(skill))
    if receipt is not None and config is not None:
        receipt.update(validity=guard_receipt,target_guard_seconds=guard_seconds,permanently_retired=False)
    return output,receipt

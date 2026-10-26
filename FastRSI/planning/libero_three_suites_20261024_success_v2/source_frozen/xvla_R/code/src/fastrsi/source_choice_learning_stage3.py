"""Version-bound, CPU source choice model; no execution or persistent updates.

Source choice is separate from the older continue/handoff experiment. Training
uses one actual matched-prefix policy continuation per episode, not labels for
all frames. Equal-success timings are noisy observations, not certified wins.
"""
import hashlib
import json
import math
from pathlib import Path
import time
import inspect
import textwrap

import numpy as np

from .decision_learning_stage3 import public_features
from .phase_transfer import GOAL, compatible, rotation, state8
from .phase_transfer_v3 import validate_pool_v3, validate_v3_config, base_pool
from .retarget_geometry import rotation_vector

ALLOWED_DEVELOPMENT_INITS = {10, 11, 12, 13, 44}
FEATURE_DIM = 26


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_physical(source):
    """Exclude acquisition IDs, retain the complete actually paid controls."""
    return dict(segments=source['carrier']['segments'], profiles=source['profiles'])


def inventory_binding(bank):
    validate_pool_v3(bank)
    if len(bank['sources']) != 2:
        raise ValueError('This prototype binds exactly two existing sources')
    physical = [source_physical(source) for source in bank['sources']]
    return dict(physical_inventory_sha256=digest(physical),
                physical_source_sha256=[digest(x) for x in physical],
                scope=dict(suite=bank['suite'], task_id=bank['task_id'], goal=bank['goal']),
                controller_config_sha256=bank['source_evidence']['controller_config_sha256'])


def legal_candidates(bank, guard, robot8, phase):
    """Same compatibility and within-source ordering as the frozen actors."""
    validate_pool_v3(bank)
    if type(phase) is not int or phase not in (0, 1):
        raise ValueError('Unknown native command phase')
    state = state8(robot8)
    candidates = []
    for index, source in enumerate(bank['sources']):
        matches = []
        for profile in source['profiles']:
            if profile['phase_ordinal'] != phase or not compatible(state, {'entry_robot8':profile['robot8']}, guard):
                continue
            reference = state8(profile['robot8'])
            position = float(np.linalg.norm(state[:3]-reference[:3]))
            angle = float(np.linalg.norm(rotation_vector(rotation(state[3:6]) @ rotation(reference[3:6]).T)))
            aperture = float(abs(np.abs(state[6:]).sum()-np.abs(reference[6:]).sum()))
            segment = source['carrier']['segments'][phase]
            matches.append(dict(source_index=index, source_id=source['id'], phase_ordinal=phase,
                source_profile_step=profile['source_step'], profile_state_sha256=profile['state_sha256'],
                physical_source_sha256=digest(source_physical(source)), position_m=position,
                rotation_rad=angle, aperture_m=aperture, remaining_source_steps=segment['source_stop']-profile['source_step']))
        if matches:
            candidates.append(min(matches, key=lambda c:(c['position_m'],c['rotation_rad'],c['source_id'],c['source_profile_step'])))
    return candidates


def fixed_select(candidates, policy):
    if not candidates:
        return None
    if policy == 'oldest':
        return min(candidates, key=lambda c:(c['source_index'],c['position_m'],c['rotation_rad'],c['source_id'],c['source_profile_step']))
    if policy == 'nearest':
        return min(candidates, key=lambda c:(c['position_m'],c['rotation_rad'],c['source_id'],c['source_profile_step']))
    raise ValueError('Unknown fixed source policy')


def make_context(bank, pool_raw_sha256, guard, *, prompt, robot8, history,
                 phase, closed_native_steps, lift_from_closed_start_m, array_bindings):
    if prompt != GOAL or not isinstance(pool_raw_sha256, str) or len(pool_raw_sha256) != 64:
        raise ValueError('Exact goal and raw pool version required')
    state = state8(robot8)
    if (array_bindings.get('observation/state') != dict(shape=[8],dtype='float32',sha256=hashlib.sha256(state.tobytes()).hexdigest())
            or type(closed_native_steps) is not int or closed_native_steps < guard['min_closed_steps']
            or type(lift_from_closed_start_m) not in (float,int) or not math.isfinite(lift_from_closed_start_m)
            or lift_from_closed_start_m < guard['min_lift_m']):
        raise ValueError('Missing current public entry evidence')
    if type(history[-1]['step']) is not int:
        raise ValueError('Current public boundary required')
    candidates = legal_candidates(bank, guard, state, phase)
    features = None
    if len(candidates) == 2:
        features = public_features(state, history) + [float(phase==0),float(phase==1),closed_native_steps/60.,lift_from_closed_start_m/.05]
        for c in candidates:
            features += [c['position_m']/guard['position_m'],c['rotation_rad']/guard['rotation_rad'],
                         c['aperture_m']/guard['aperture_m'],c['remaining_source_steps']/60.,c['source_profile_step']/400.]
        if len(features) != FEATURE_DIM or not np.isfinite(features).all():
            raise ValueError('Malformed public candidate features')
    value = dict(version=1,kind='observed_source_choice_context_v1',pool_raw_sha256=pool_raw_sha256,
        inventory=inventory_binding(bank),guard_sha256=digest(guard),prompt=prompt,
        phase=phase,boundary_step=history[-1]['step'],robot8=state.tolist(),history=history,
        closed_native_steps=closed_native_steps,lift_from_closed_start_m=lift_from_closed_start_m,
        array_bindings=array_bindings,candidates=candidates,features=features,
        held_object_verified=False,features_use_images=False,
        feature_scope='Current issued proprio, earlier actually observed fingers, original native command progress and paid source profiles; no task/init IDs, future outcomes, hidden objects or new encoder.')
    return {**value,'sha256':digest(value)}


def validate_context(context):
    body = {k:v for k,v in context.items() if k != 'sha256'}
    if digest(body) != context.get('sha256'):
        raise ValueError('Public source-choice context changed')
    if context.get('kind') != 'observed_source_choice_context_v1' or context.get('prompt') != GOAL:
        raise ValueError('Wrong source-choice contract')
    if context.get('features') is not None and (len(context['features']) != FEATURE_DIM or not np.isfinite(context['features']).all()):
        raise ValueError('Invalid source feature vector')
    if len({c['source_index'] for c in context['candidates']}) != len(context['candidates']):
        raise ValueError('Duplicate source candidates')
    return context


def validate_record(record):
    if record.get('mode') != 'development' or record.get('initial') not in ALLOWED_DEVELOPMENT_INITS:
        raise ValueError('Only registered development outcomes may train')
    if record.get('matched_observed_prefix') is not True or record.get('matched_first_divergence') is not True:
        raise ValueError('Unmatched branch cannot supervise source choice')
    context = validate_context(record['context'])
    if context['features'] is None or len(record.get('branches', [])) != 2:
        raise ValueError('Both real alternatives required')
    if {b['source_index'] for b in record['branches']} != {0,1}:
        raise ValueError('No distinct executed source alternatives')
    if not record.get('group') or not record.get('source_hashes'):
        raise ValueError('Original episode/source SHA missing')
    for digest_value in record['source_hashes'].values():
        if not isinstance(digest_value,str) or len(digest_value) != 64:
            raise ValueError('Original source binding invalid')
    for branch in record['branches']:
        if (branch.get('executed') is not True or branch.get('chain_status') != 'PASSED'
                or type(branch.get('native_final_success')) is not bool or type(branch.get('native_root_success')) is not bool):
            raise ValueError('Branch outcome incomplete')
        c = next(c for c in context['candidates'] if c['source_index']==branch['source_index'])
        if branch.get('physical_source_sha256') != c['physical_source_sha256']:
            raise ValueError('Observed branch belongs to another source')
        if branch.get('source_profile_step') != c['source_profile_step']:
            raise ValueError('Branch did not use the legal source profile')
        cost = branch['cost']['postdecision_execution_seconds']
        if type(cost) not in (float,int) or not math.isfinite(cost) or cost <= 0:
            raise ValueError('Actual execution-cost observation missing')
    return record


def _ridge(x, y, alpha):
    design = np.column_stack((np.ones(len(x)), x))
    penalty = np.eye(design.shape[1])*alpha
    penalty[0,0] = 0
    return np.linalg.solve(design.T@design+penalty, design.T@np.asarray(y)).tolist()


def fit(records, bank, pool_raw_sha256, *, alpha=10.0):
    """Fixed-regularization paired regression, no tuning on held-out episodes."""
    started = time.perf_counter()
    if alpha != 10.0:
        raise ValueError('First probe has one preregistered regularization, not a search')
    binding = inventory_binding(bank)
    groups = set()
    for row in records:
        validate_record(row)
        if row['group'] in groups:
            raise ValueError('Repeated frames/episodes cannot receive multiple training votes')
        groups.add(row['group'])
        if row['context']['inventory'] != binding:
            raise ValueError('Physical source/frame inventory differs from deployment pool')
        if row['context']['guard_sha256'] != records[0]['context']['guard_sha256']:
            raise ValueError('Training records used different guard conditions')
    if len(groups) < 2:
        raise ValueError('TWO_DISTINCT_MATCHED_DEVELOPMENT_EPISODES_REQUIRED')
    x = np.asarray([r['context']['features'] for r in records])
    center = x.mean(axis=0);scale=x.std(axis=0);scale[scale<1e-6]=1.
    z=(x-center)/scale
    quality=[]; root_quality=[]; cost_indices=[]; cost_delta=[]; vla_delta=[]
    for index,row in enumerate(records):
        branches=sorted(row['branches'],key=lambda b:b['source_index'])
        quality.append([float(b['native_final_success']) for b in branches])
        root_quality.append([float(b['native_root_success']) for b in branches])
        if all(b['native_final_success'] for b in branches):
            cost_indices.append(index)
            cost_delta.append(math.log(branches[1]['cost']['postdecision_execution_seconds']/branches[0]['cost']['postdecision_execution_seconds']))
            vla_delta.append(math.log((1+branches[1]['cost']['actual_vla_calls'])/(1+branches[0]['cost']['actual_vla_calls'])))
    cost_weights = _ridge(z[cost_indices],cost_delta,alpha) if cost_indices else None
    vla_weights = _ridge(z[cost_indices],vla_delta,alpha) if cost_indices else None
    body=dict(version=1,kind='paired_source_choice_ridge_v1',deployment='SHADOW_ONLY',
        pool_raw_sha256=pool_raw_sha256,inventory=binding,guard_sha256=records[0]['context']['guard_sha256'],
        center=center.tolist(),scale=scale.tolist(),alpha=alpha,
        quality_weights=np.array(_ridge(z,quality,alpha)).tolist(),
        root_quality_weights=np.array(_ridge(z,root_quality,alpha)).tolist(),
        execution_cost_logratio_weights=cost_weights,vla_logratio_weights=vla_weights,
        training_groups=sorted(groups),training_record_sha256=[digest(r) for r in records],
        distinct_episode_count=len(groups),cost_observation_count=len(cost_indices),
        class_probability_calibrated=False,success_constraint_validated=False,
        selection_rule='Shadow: native quality difference >=0.25 first; otherwise lower predicted remaining execution cost, oldest fallback if cost unavailable.',
        target_scope='Outcomes of the two actually executed fixed-policy continuations from one matched first divergence. Not every-frame success labels, not isolated single-action causal outcomes, not realized learned-controller outcomes.',
        cost_scope='Paired noisy remaining OS execution; terminal report, verification, learning and complete worker bills retained separately and never erased.')
    return {**body,'sha256':digest(body),'fit_seconds':time.perf_counter()-started}


def select(model, context):
    started=time.perf_counter();validate_context(context)
    body={k:v for k,v in model.items() if k not in ('sha256','fit_seconds')}
    if digest(body)!=model.get('sha256') or model.get('kind')!='paired_source_choice_ridge_v1':
        raise ValueError('Decision model version changed')
    if (model['pool_raw_sha256']!=context['pool_raw_sha256'] or model['inventory']!=context['inventory']
            or model['guard_sha256']!=context['guard_sha256']):
        raise ValueError('Current source pool/guard/frame differs from trained deployment binding')
    candidates=context['candidates']
    if len(candidates)!=2:
        chosen=fixed_select(candidates,'oldest')
        return dict(choice=chosen,reason='NO_TWO_WAY_MODEL_DECISION',model_sha256=model['sha256'],execution_authorized=False)
    x=np.r_[1.,(np.asarray(context['features'])-model['center'])/model['scale']]
    quality=np.asarray(x@np.asarray(model['quality_weights']))
    root_quality=np.asarray(x@np.asarray(model['root_quality_weights']))
    delta=None if model['execution_cost_logratio_weights'] is None else float(x@model['execution_cost_logratio_weights'])
    vla_delta=None if model['vla_logratio_weights'] is None else float(x@model['vla_logratio_weights'])
    if abs(quality[1]-quality[0])>=.25:
        source_index=int(quality[1]>quality[0]);reason='UNCALIBRATED_NATIVE_QUALITY_ESTIMATE'
    elif delta is not None:
        source_index=int(delta<0);reason='NOISY_EXECUTION_COST_ESTIMATE'
    else:
        source_index=0;reason='OLDEST_FIXED_FALLBACK_NO_COST_SUPPORT'
    chosen=next(c for c in candidates if c['source_index']==source_index)
    return dict(choice=chosen,reason=reason,model_sha256=model['sha256'],context_sha256=context['sha256'],
        native_quality_estimate=quality.tolist(),native_root_quality_estimate=root_quality.tolist(),
        estimated_execution_logratio_source1_vs_source0=delta,estimated_vla_logratio_source1_vs_source0=vla_delta,
        probability=None,success_constraint_validated=False,execution_authorized=False,
        decision_seconds=time.perf_counter()-started)


def make_learned_source_carrier(config, bank):
    """Opt-in factory; an explicit frozen model is required before any rollout.

    The original carrier choose state machine is reused byte-for-byte except its
    one minimum-over-legal-matches expression. Execution, guards, phase progress,
    active continuation, release and native fallback remain original.
    """
    from . import phase_transfer_pool as original
    validate_v3_config(config,bank)
    if config.get('source_selection')!='learned_source_choice_stage3_v1' or config.get('carrier_mode')!='dense':
        raise ValueError('Explicit learned dense source-selection contract required')
    ref=config.get('decision_model',{})
    if set(ref)!={'artifact','sha256'} or not Path(ref['artifact']).is_absolute():
        raise ValueError('Frozen decision-model artifact/SHA required')
    raw=Path(ref['artifact']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ref['sha256']:
        raise ValueError('Decision-model raw file changed')
    model=json.loads(raw)
    # Validate model and current pool/guard before execution; no inference data.
    body={k:v for k,v in model.items() if k not in ('sha256','fit_seconds')}
    if (digest(body)!=model.get('sha256') or model.get('kind')!='paired_source_choice_ridge_v1'
            or model['pool_raw_sha256']!=config['sha256'] or model['inventory']!=inventory_binding(bank)
            or model['guard_sha256']!=digest(config['guard'])):
        raise ValueError('Decision-model pool/frame/guard/version differs')
    text=textwrap.dedent(inspect.getsource(original.PoolCarrier.choose))
    before="chosen=min(matches,key=lambda x:x[:4]);source,profile=chosen[-2:]"
    if text.count(before)!=1:
        raise ValueError('Original carrier selection hook changed')
    text=text.replace(before,"chosen=self.learned_match(matches,s,step);source,profile=chosen[-2:]")
    namespace=dict(vars(original))
    exec(compile(text,'source_choice_learning_stage3:original_carrier_hook','exec'),namespace)
    bound_choose=namespace['choose']

    class LearnedSourceCarrier(original.PoolCarrier):
        def __init__(self):
            legacy=dict(config);legacy.pop('public_state_binding')
            super().__init__(legacy,base_pool(bank))
            self.c=config;self.observed_history=[];self.model_decisions=[]

        def learned_match(self,matches,state,step):
            context=make_context(bank,config['sha256'],config['guard'],prompt=config['goal'],robot8=state,
                history=self.observed_history,phase=self.progress.phase,closed_native_steps=self.progress.closed,
                lift_from_closed_start_m=float(state[2])-self.progress.base,
                array_bindings={'observation/state':dict(shape=[8],dtype='float32',sha256=hashlib.sha256(state.tobytes()).hexdigest())})
            result=select(model,context)
            chosen=result['choice']
            candidates=[x for x in matches if x[2]==chosen['source_id'] and x[3]==chosen['source_profile_step']]
            if len(candidates)!=1:
                raise ValueError('Model output not an actual legal source/profile')
            self.model_decisions.append(dict(step=step,model_artifact_sha256=ref['sha256'],**result))
            return candidates[0]

        def choose(self,session,step,robot8):
            if session==self.c['root_session']:
                state=state8(robot8)
                if self.observed_history and step<=self.observed_history[-1]['step']:
                    raise ValueError('Repeated or stale learned source observation')
                self.observed_history.append(dict(step=step,positions=state[6:].tolist()))
            n=len(self.model_decisions)
            result=bound_choose(self,session,step,robot8)
            if result is not None:
                result=dict(result,source_selection='learned_source_choice_stage3_v1',
                            decision_model_sha256=model['sha256'],decision_model_artifact_sha256=ref['sha256'])
                if len(self.model_decisions)>n:
                    result['decision_seconds']=self.model_decisions[-1].get('decision_seconds',0.)
                    result['decision_reason']=self.model_decisions[-1]['reason']
            return result

    return LearnedSourceCarrier()

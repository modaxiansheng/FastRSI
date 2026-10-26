"""Offline source identity binding; never an online policy feature or success test.

The five optional JSON arguments are original bytes, not reserialized objects.
No simulator state array, XML, image pixels, model or environment is loaded here.
Caller still verifies execution success and source-trace/controller provenance.
"""
import hashlib
import json
import math


VERSION = 'source_initialization_identity_v1'
CAMERAS = ('agentview_image', 'robot0_eye_in_hand_image')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw, name):
    if not isinstance(raw, bytes):
        raise ValueError(name+' must be original JSON bytes')
    value=json.loads(raw)
    if not isinstance(value,dict):raise ValueError(name+' must be a JSON object')
    return value,_sha(raw)


def _digest(value):
    if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Invalid identity SHA256')
    return value


def _equal(actual, expected, name):
    if actual != expected or (type(expected) in (int,bool) and type(actual) is not type(expected)):
        raise ValueError('Source initialization differs: '+name)


def validate_source_initialization(case, runtime_config, *, restore_event=None,
        sampler_binding=None, registration=None, collection_manifest=None,
        sample_receipt=None, public_observation=None):
    """Return evidence only; fixed sources retain their legacy init_state_id.

    For fixed sources this returns exactly {'init_state_id': old_id}. Consumers
    must keep the legacy evidence layout, rather than add a new metadata object.
    For sampler sources the returned None identity replaces the administrative
    official slot. It must never be passed to a matcher/selector as a feature.
    """
    if case.get('mode')!='development':raise ValueError('Only development sources may become skills')
    if case.get('parent_session_id') is not None or case.get('replan_attempt',0)!=0:
        raise ValueError('A child/replan cannot certify the source root')
    for field,runtime_field in (('suite','benchmark_name'),('task_id','task_id'),('init_state_id','init_state_id')):
        if field not in case or runtime_field not in runtime_config:raise ValueError('Missing source runtime identity')
        _equal(runtime_config[runtime_field],case[field],field)
    _equal(runtime_config.get('control_mode'),'relative','control_mode')
    _equal(runtime_config.get('max_steps'),case.get('max_steps',400),'max_steps')
    if 'native_sampler_binding' not in case:
        if any(x is not None for x in (restore_event,sampler_binding,registration,collection_manifest,sample_receipt,public_observation)):
            raise ValueError('Sampler evidence cannot be attached to a fixed-state source')
        return {'init_state_id':case['init_state_id']}
    if not isinstance(case['native_sampler_binding'],str) or not case['native_sampler_binding']:
        raise ValueError('Missing sampler configuration reference')
    cfg,cfg_sha=_json(sampler_binding,'sampler_binding')
    reg,reg_sha=_json(registration,'registration')
    manifest,manifest_sha=_json(collection_manifest,'collection_manifest')
    sample,sample_sha=_json(sample_receipt,'sample_receipt')
    public,public_sha=_json(public_observation,'public_observation')
    _equal(cfg_sha,case.get('native_sampler_binding_sha256'),'case configuration raw SHA')
    _equal(cfg.get('version'),'native_sampler_agent_binding_v1','binding version')
    binding=cfg.get('restore') or {}
    _equal(binding.get('version'),'native_sampler_draw_restore_v1','restore configuration version')
    sid=case.get('id')
    if not isinstance(sid,str) or not sid:raise ValueError('Missing source Session')
    _equal(binding.get('root_session'),sid,'configured root Session')
    _equal(binding.get('registration_sha256'),reg_sha,'registered raw SHA')
    _equal(binding.get('collection_manifest_sha256'),manifest_sha,'collection raw SHA')
    from .native_sampler_restore_v1 import registration_role
    _equal(registration_role(reg),'DEVELOPMENT_INPUT','registered development partition')
    for key,expected in [('suite',case['suite']),('task_id',case['task_id']),
            ('task_description',case.get('task_description')),('num_steps_wait',10),
            ('evaluation_max_steps',400),('control_mode','relative'),('camera_height',256),('camera_width',256)]:
        _equal(reg.get(key),expected,'registration '+key)
    _equal(runtime_config.get('max_steps'),400,'sampler runtime max_steps')
    runtime_fields_omitted=[]
    for key,expected in [('num_steps_wait',10),('camera_height',256),('camera_width',256)]:
        if key in runtime_config:
            _equal(runtime_config[key],expected,'sampler runtime '+key)
        else:
            runtime_fields_omitted.append(key)
    seed=sample.get('seed');sample_id=binding.get('sample_id')
    if type(seed) is not int or seed not in reg['seeds'] or sample_id!='sample_'+str(seed):
        raise ValueError('Sample is not a registered development draw')
    attempts=sample.get('sampling_attempts');max_attempts=reg.get('max_native_reset_attempts')
    if type(attempts) is not int or type(max_attempts) is not int or not 1 <= attempts <= max_attempts:
        raise ValueError('Native sampling attempts must match the registered positive bound')
    for key,expected in [('status','COLLECTED_RETAINED'),('role','DEVELOPMENT_INPUT'),
            ('settling_actions',10),('policy_actions',0),('construction_completed',True)]:
        _equal(sample.get(key),expected,'sample '+key)
    _equal(manifest.get(sample_id+'/COLLECTION_RECEIPT.json'),sample_sha,'sample receipt raw SHA')
    _equal(manifest.get(sample_id+'/public_observation/OBSERVATION.json'),public_sha,'public metadata raw SHA')
    _equal(sample.get('public_observation_sha256'),public_sha,'sample public metadata SHA')
    _equal(public.get('task_description'),case.get('task_description'),'public goal')
    _equal(sample.get('bddl_sha256'),_digest(reg.get('source_bindings',{}).get('bddl_sha256')),'BDDL identity')
    for suffix in ('PRE_SETTLE_STATE.npy','POST_SETTLE_STATE.npy'):
        _digest(manifest.get(sample_id+'/evaluation_only/'+suffix))
    _equal(manifest.get(sample_id+'/evaluation_only/MODEL.xml'),_digest(sample.get('xml_sha256')),'XML identity')
    event=restore_event
    if not isinstance(event,dict):raise ValueError('Real root native_sampler_restore event required')
    for key,expected in [('event','native_sampler_restore'),('version','native_sampler_agent_binding_v1'),
            ('session_id',sid),('parent_session_id',None),('replan_attempt',0),
            ('status','MATCHED_FROZEN_INPUT'),('instance_id',None),('per_session_action_limit',400)]:
        if key not in event:raise ValueError('Missing root restore '+key)
        _equal(event[key],expected,'root restore '+key)
    if type(case['init_state_id']) is not int:raise ValueError('Ignored official slot must be an integer')
    _equal(event.get('internal_ignored_official_slot'),case['init_state_id'],'ignored official slot')
    identity=event.get('native_sampler') or {}
    for key,expected in [('role','DEVELOPMENT_INPUT'),('sample_id',sample_id),('draw_seed',seed),('collection_manifest_sha256',manifest_sha)]:
        _equal(identity.get(key),expected,'root sampler '+key)
    restore=event.get('restore') or {}
    for key,expected in [('version','native_sampler_draw_restore_v1'),('session_id',sid),
            ('status','MATCHED_FROZEN_INPUT'),('sample_id',sample_id),('collection_manifest_sha256',manifest_sha),
            ('native_draw_calls',1),('native_draw_attempts',attempts),('state_assignment_bypasses',1),
            ('settling_steps',10),('settling_attempts',10),('native_reset_bookkeeping_reused',True),
            ('policy_actions',0),('model_calls',0),('public_rgb_exact',True),('public_robot8_exact',True),
            ('full_state_exact',True),('task_success','NOT_ASSESSED_BY_RESTORER')]:
        _equal(restore.get(key),expected,'nested draw '+key)
    for key in ('pre_settle_state','post_settle_state'):
        ref=sample.get(key)
        if not isinstance(ref,dict):raise ValueError('Missing sample state binding')
        _digest(ref.get('sha256'))
        _equal(restore.get(key),ref,'exact '+key)
    _equal(restore.get('xml_sha256'),sample['xml_sha256'],'exact XML')
    if (not isinstance(public.get('rgb'),dict) or not isinstance(restore.get('public_rgb'),dict)
            or set(public['rgb'])!=set(CAMERAS) or set(restore['public_rgb'])!=set(CAMERAS)):
        raise ValueError('Both native public cameras required')
    for camera in CAMERAS:
        source=public['rgb'][camera]
        if not isinstance(source,dict):raise ValueError('Missing public camera metadata')
        _equal(source.get('shape'),[256,256,3],'public camera shape')
        _equal(source.get('dtype'),'uint8','public camera dtype')
        _digest(source.get('raw_sha256'))
        _equal(manifest.get(sample_id+'/public_observation/'+camera+'.png'),_digest(source.get('png_sha256')),'PNG identity')
        _equal(restore['public_rgb'][camera],{k:source[k] for k in ('shape','dtype','raw_sha256')},'exact '+camera)
    robot=public.get('robot8')
    if not isinstance(robot,list) or len(robot)!=8 or any(type(v) not in (int,float) or not math.isfinite(v) for v in robot):
        raise ValueError('Finite public robot8 required')
    _equal(restore.get('public_robot8'),robot,'exact robot8')
    return dict(version=VERSION,init_state_id=None,initialization_source='native_sampler_extension',
        internal_ignored_official_slot=case['init_state_id'],root_session=sid,role='DEVELOPMENT_INPUT',
        sample_id=sample_id,draw_seed=seed,collection_manifest_sha256=manifest_sha,
        sampler_binding_sha256=cfg_sha,registration_sha256=reg_sha,
        sample_receipt_sha256=sample_sha,public_observation_sha256=public_sha,
        restore_event_sha256=_sha(json.dumps(event,sort_keys=True,separators=(',',':'),allow_nan=False).encode()),
        public_input_exact=True,full_state_exact=True,identity_scope='ADMINISTRATIVE_PROVENANCE_NOT_MODEL_FEATURES',
        runtime_config_omitted_fields=runtime_fields_omitted,
        camera_and_settling_evidence_scope='SHA_BOUND_REGISTRATION_AND_ACTUAL_RESTORE_RECEIPT_PUBLIC_SHAPES',
        source_success_assessed=False)

"""Opt-in successor programs compiled from bound, real development records.

The annotations describe a candidate's command geometry, not physical facts.
This opens source replacement without letting a language model invent commands
or certify transfer. The v1 source31 declaration and default remain unchanged.
"""
import copy
import hashlib
import json
import math
import time

from .experience_compiler import canonical_sha, compile_live_success

VERSION = 'native_goal_component_v2'
TRANSFORM_VERSION = 'dual_anchor_v2'
SAMPLER_RECORDS = ('sampler_binding', 'sampler_registration', 'sampler_manifest',
                   'sampler_receipt', 'sampler_observation')


def validate_declaration(value):
    from .native_goal_component_v1 import _mapping, _reference, _digest
    _mapping(value, ('version', 'source_artifact', 'source_records', 'transform', 'bindings'), 'successor component')
    if value['version'] != VERSION:
        raise ValueError('Unknown successor component version')
    _reference(value['source_artifact'], 'source_artifact')
    records = value['source_records']
    base_records = ('result', 'trace', 'live_controls', 'controller_jsonl', 'target_events')
    record_fields = (base_records + SAMPLER_RECORDS
                     if isinstance(records, dict) and set(records) == set(base_records + SAMPLER_RECORDS)
                     else base_records)
    _mapping(records, record_fields, 'actual source records')
    for name, ref in records.items():
        _reference(ref, name)
    transform = value['transform']
    _mapping(transform, ('version', 'source_rgb', 'calibration', 'source_rois_xyxy', 'anchor_plan', 'max_offset_m'), 'successor transform')
    if transform['version'] != TRANSFORM_VERSION:
        raise ValueError('Only the successor two-anchor geometry is supported')
    for name in ('source_rgb', 'calibration'):
        _reference(transform[name], name)
    refs = [value['source_artifact'], *records.values(), transform['source_rgb'], transform['calibration']]
    if len({ref['path'] for ref in refs}) != len(refs):
        raise ValueError('Source records and artifacts must have distinct paths')
    rois = transform['source_rois_xyxy']
    if (not isinstance(rois, list) or len(rois) != 2 or any(
            not isinstance(roi, list) or len(roi) != 4 or any(type(x) is not int for x in roi)
            or not 0 <= roi[0] < roi[2] or not 0 <= roi[1] < roi[3] for roi in rois)):
        raise ValueError('Two explicit source image regions are required')
    if not (rois[0][2] <= rois[1][0] or rois[1][2] <= rois[0][0]
            or rois[0][3] <= rois[1][1] or rois[1][3] <= rois[0][1]):
        raise ValueError('Source regions must be disjoint')
    limit = transform['max_offset_m']
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or not 0 <= limit <= .06:
        raise ValueError('Keep the existing maximum planar-offset envelope')
    bindings = value['bindings']
    _mapping(bindings, ('source_goal', 'model_config_sha256', 'target_ref', 'action_dim', 'robot_state_dim'), 'bindings')
    if not isinstance(bindings['source_goal'], str) or not bindings['source_goal'].strip():
        raise ValueError('Complete source goal required')
    _digest(bindings['model_config_sha256'], 'model_config')
    if not isinstance(bindings['target_ref'], str) or not bindings['target_ref'].startswith('target://') or not bindings['target_ref'][9:]:
        raise ValueError('Original native target required')
    if type(bindings['action_dim']) is not int or bindings['action_dim'] != 7 or type(bindings['robot_state_dim']) is not int or bindings['robot_state_dim'] != 8:
        raise ValueError('Original robot action and observation dimensions required')
    if not isinstance(transform['anchor_plan'], list) or len(transform['anchor_plan']) != 2:
        raise ValueError('Two source command annotations required')
    return copy.deepcopy(value)


def qualified_source(workspace, declaration):
    """Recompile original records; a candidate's claimed success is insufficient."""
    from .native_goal_component_v1 import _read
    start = time.perf_counter()
    source = json.loads(_read(workspace, declaration['source_artifact']))
    refs = declaration['source_records']
    report = json.loads(_read(workspace, refs['result']))
    trace = [json.loads(line) for line in _read(workspace, refs['trace']).splitlines()]
    live = json.loads(_read(workspace, refs['live_controls']))
    if (not trace or trace[0].get('method') != 'reset' or trace[0].get('step') != 0
            or trace[0].get('observation_id') != 'libero_obs_0'
            or trace[0].get('session_id') != live.get('session_id')
            or trace[0].get('images', {}).get('0000_agentview_image.png') != declaration['transform']['source_rgb']['sha256']):
        raise ValueError('Source RGB does not bind the actual reset observation')
    if report.get('status') != 'COMPLETED' or not isinstance(report.get('cases'), list):
        raise ValueError('Complete original Agent report required')
    rows = [row for row in report['cases'] if row.get('case', {}).get('id') == live.get('session_id')]
    if len(rows) != 1:
        raise ValueError('Exactly one matching original root result required')
    result = rows[0]
    raw_controls = _read(workspace, refs['controller_jsonl'])
    if (hashlib.sha256(raw_controls).hexdigest() != live.get('controller_jsonl_sha256')
            or [json.loads(x) for x in raw_controls.splitlines()] != live.get('controls')):
        raise ValueError('Actual controller JSONL differs from source recording')
    events = [json.loads(x) for x in _read(workspace, refs['target_events']).splitlines()]
    roots = [row for row in events if row.get('event') == 'native_observation' and row.get('session_id') == live.get('session_id')]
    terminal = result.get('native_terminal_by_session', {}).get(live.get('session_id'))
    if not roots or roots[-1] != terminal or roots[-1].get('native_success') is not True:
        raise ValueError('Actual target terminal does not confirm source root success')
    provenance = dict(result_sha256=refs['result']['sha256'], trace_sha256=refs['trace']['sha256'],
                      controller_trace_sha256=refs['live_controls']['sha256'])
    initialization_records = None
    if 'native_sampler_binding' in result['case']:
        if not all(name in refs for name in SAMPLER_RECORDS):
            raise ValueError('Sampler source requires its complete initialization references')
        restores = [row for row in events if row.get('event') == 'native_sampler_restore'
                    and row.get('session_id') == live.get('session_id')]
        if len(restores) != 1:
            raise ValueError('Exactly one actual root sampler restoration required')
        initialization_records = dict(restore_event=restores[0],
            sampler_binding=_read(workspace, refs['sampler_binding']),
            registration=_read(workspace, refs['sampler_registration']),
            collection_manifest=_read(workspace, refs['sampler_manifest']),
            sample_receipt=_read(workspace, refs['sampler_receipt']),
            public_observation=_read(workspace, refs['sampler_observation']))
    elif any(name in refs for name in SAMPLER_RECORDS):
        raise ValueError('Sampler metadata cannot relabel a fixed source')
    reconstructed = compile_live_success(result, trace, live, provenance=provenance,
                                        initialization_records=initialization_records)
    if reconstructed != source:
        raise ValueError('Candidate differs from its actually executed source records')
    origins = set(source['source_evidence']['execution_origins'])
    if not origins or origins == {'goal_skill_action'}:
        raise ValueError('Pure replay is not a new capability source')
    return source, dict(status='ACTUAL_DEVELOPMENT_SOURCE_RECOMPILED',
        source_records=copy.deepcopy(refs), source_canonical_sha256=canonical_sha(source),
        execution_origins=sorted(origins), seconds=time.perf_counter()-start,
        transfer_certified=False, learning_gain_claimed=False)


def prepare_component(workspace, declaration, *, expected_goal,
                      expected_model_config_sha256, expected_target_ref, current_rgb=None):
    from .native_goal_component_v1 import _read, _digest, bind_initial_rgb
    from .dual_anchor_retarget_v2 import retarget_dual_anchors
    import cv2
    import numpy as np
    value = validate_declaration(declaration)
    binding = value['bindings']
    _digest(expected_model_config_sha256, 'expected_model_config')
    if (binding['source_goal'] != expected_goal or binding['model_config_sha256'] != expected_model_config_sha256
            or binding['target_ref'] != expected_target_ref):
        raise ValueError('Current goal/model/target differs')
    source, qualification = qualified_source(workspace, value)
    if source['source_goal'] != expected_goal:
        raise ValueError('Source actual goal differs')
    transform = value['transform']
    image_blob = _read(workspace, transform['source_rgb'])
    calibration = json.loads(_read(workspace, transform['calibration']))
    image = cv2.imdecode(np.frombuffer(image_blob, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Source RGB is not decodable')
    source_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    shape = (calibration['height'], calibration['width'], 3)
    if source_rgb.shape != shape or any(roi[2] > shape[1] or roi[3] > shape[0] for roi in transform['source_rois_xyxy']):
        raise ValueError('Source RGB/calibration/regions differ')
    # Check source commands and annotations even on reference-only reads.
    zero = {p['anchor']: [0., 0., 0.] for p in transform['anchor_plan']}
    retarget_dual_anchors(source, transform['anchor_plan'], zero,
        max_offset_m=transform['max_offset_m'], expected_source_sha256=canonical_sha(source))
    receipt = dict(version=VERSION, declaration_sha256=canonical_sha(value),
        source_artifact_sha256=value['source_artifact']['sha256'],
        source_waypoints_sha256=canonical_sha(source['waypoints']),
        source_rgb_raw_sha256=transform['source_rgb']['sha256'],
        source_rgb_array_sha256=hashlib.sha256(source_rgb.tobytes()).hexdigest(),
        calibration_raw_sha256=transform['calibration']['sha256'],
        model_config_sha256=binding['model_config_sha256'],
        model_identity_scope='EXTERNAL_SERVICE_MODEL_CONTRACT_NOT_PI_WEIGHTS_OR_SOURCE_VLA_PROVENANCE',
        transform_version=TRANSFORM_VERSION, current_rgb_array_sha256=None,
        current_rgb_shape=None, current_rgb_dtype=None, offsets=None,
        derived_waypoints_sha256=None, matching=None, source_qualification=qualification,
        status='REFERENCES_VALIDATED_NOT_DERIVED', execution_applicability_certified=False,
        semantic_identity_certified=False, physical_release_status='UNKNOWN', native_goal_status='NOT_EVALUATED',
        source_candidate_unchanged=True, original_default_changed=False, model_calls=0, physics_steps=0)
    prepared = dict(version=VERSION, declaration=value, source_candidate=copy.deepcopy(source),
        source_artifact_sha256=value['source_artifact']['sha256'],
        source_waypoints_sha256=receipt['source_waypoints_sha256'], calibration=calibration,
        derived_waypoints=None, derivation_receipt=receipt)
    if current_rgb is not None:
        if not isinstance(current_rgb, np.ndarray) or current_rgb.dtype != np.uint8 or current_rgb.shape != shape:
            raise ValueError('Current raw front RGB must match declared HWC uint8 dimensions')
        offsets, matching = bind_initial_rgb(source_rgb, current_rgb, transform['source_rois_xyxy'], calibration)
        named = {p['anchor']: offset for p, offset in zip(transform['anchor_plan'], offsets)}
        geometry = retarget_dual_anchors(source, transform['anchor_plan'], named,
            max_offset_m=transform['max_offset_m'], expected_source_sha256=canonical_sha(source))
        prepared['derived_waypoints'] = geometry['waypoints']
        receipt.update(status='UNVALIDATED_DERIVED_OPERATION_CANDIDATE',
            current_rgb_array_sha256=hashlib.sha256(current_rgb.tobytes()).hexdigest(),
            current_rgb_shape=list(current_rgb.shape), current_rgb_dtype=str(current_rgb.dtype),
            offsets=named, derived_waypoints_sha256=geometry['result_waypoints_sha256'], matching=matching,
            geometry={k: v for k, v in geometry.items() if k != 'waypoints'})
    return prepared

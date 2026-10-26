"""Explicit native target consumer for a registered, derived robot program.

No image matching, learned acceptance, reset, goal replacement or native-success
mutation occurs here. Untagged native policy actions retain their original path.
"""
import copy
import hashlib
import inspect
import json
import re
import time
from pathlib import Path

import numpy as np

from .dual_anchor_retarget_v1 import retarget_dual_anchors, _sha
from .experience_compiler import canonical_sha
from .goal_skill import execute, marker
from .live_controller_trace import controller_config
from .planar_binding import camera_rotation


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_file(root, relative):
    root = Path(root).resolve()
    value = Path(relative)
    if value.is_absolute() or '..' in value.parts:
        raise ValueError('Component reference must be workspace relative')
    path = root/value
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError('Symlink component reference')
    if not path.is_file():
        raise ValueError('Missing component file')
    return path


def registry_snapshot(workspace):
    from .native_registry_files_v1 import read_registry_files
    return {name: hashlib.sha256(raw).hexdigest()
            for name, raw in read_registry_files(workspace).items()}


def yaml_document(path):
    import yaml
    match = re.search(r'```ya?ml[^\n]*\n(?P<body>.*?)\n```', Path(path).read_text(), re.S)
    if match is None:
        raise ValueError('Missing native YAML block')
    return yaml.safe_load(match.group('body'))


def verify_derived_geometry(source, recomputed, received, receipt_geometry):
    """Verify numerical equivalence without replacing cryptographically bound bytes.

    Only transformed XY may differ by <= 1e-12m, rtol=0. All other source
    values, and complete weight-zero rows, remain exactly equal to source.
    """
    if not isinstance(received, list) or len(received) != len(source):
        raise ValueError('Derived waypoint extent differs')
    if receipt_geometry['result_waypoints_sha256'] != _sha(received):
        raise ValueError('Received geometry SHA mismatch')
    for key in ('version', 'status', 'source_waypoints_sha256', 'max_offset_m',
                'destination_assumption', 'terminal', 'online_input_scope',
                'hidden_state_used', 'model_calls', 'physics_steps'):
        if receipt_geometry[key] != recomputed[key]:
            raise ValueError('Geometry source/transform metadata differs: '+key)
    if len(receipt_geometry['segments']) != len(recomputed['segments']) or len(receipt_geometry['rows']) != len(source):
        raise ValueError('Geometry annotation extent differs')
    for actual, expected in zip(receipt_geometry['segments'], recomputed['segments']):
        if set(actual) != set(expected) or any(actual[k] != expected[k] for k in expected if k != 'weights'):
            raise ValueError('Geometry anchor definition differs')
    max_error = 0.0
    for i, (original, expected, actual) in enumerate(zip(source, recomputed['waypoints'], received)):
        if not isinstance(actual, dict) or set(actual) != set(original):
            raise ValueError('Derived waypoint fields differ')
        if any(actual[k] != original[k] for k in original if k != 'target_pos'):
            raise ValueError('Source orientation/gripper changed')
        values = np.asarray(actual['target_pos'], dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all() or actual['target_pos'][2] != original['target_pos'][2]:
            raise ValueError('Source Z or finite pose changed')
        metadata = receipt_geometry['rows'][i]
        for key in ('index', 'anchor', 'phase', 'source_target_pos'):
            if metadata[key] != recomputed['rows'][i][key]:
                raise ValueError('Geometry row source definition differs')
        if metadata['target_pos'] != actual['target_pos']:
            raise ValueError('Received geometry annotation differs from executed target')
        if recomputed['rows'][i]['weight'] == 0 and actual != original:
            raise ValueError('Unchanged destination/source endpoint changed')
        error = float(np.max(np.abs(values[:2]-np.asarray(expected['target_pos'][:2]))))
        if error > 1e-12:
            raise ValueError('Transformed XY exceeds numerical verification bound')
        max_error = max(max_error, error)
    return dict(version='derived_xy_numerical_verification_v1',
                received_waypoints_sha256=_sha(received),
                recomputed_waypoints_sha256=_sha(recomputed['waypoints']),
                byte_identical=_sha(received) == _sha(recomputed['waypoints']),
                numerical_match=True, max_transformed_xy_error_m=max_error,
                absolute_tolerance_m=1e-12, relative_tolerance=0,
                immutable_source_fields_exact=True, weight_zero_rows_exact=True,
                executed_artifact='RECEIVED_SHA_BOUND_WAYPOINTS')


class RegisteredGoalTarget:
    def __init__(self, config, *, runtime_source):
        self.config = copy.deepcopy(config)
        self.workspace = Path(config['workspace']).resolve()
        expected = self.workspace/'reports/NATIVE_GOAL_COMPONENT_SELECTION.json'
        if Path(config['selection_path']) != expected:
            raise ValueError('Selection must use the fixed workspace sidecar')
        self.selection_path = expected
        self.runtime_source = str(runtime_source)
        self.bound = None
        self.root_native_started = False

    def _selection(self, runtime, tag):
        raw = local_file(self.workspace, 'reports/NATIVE_GOAL_COMPONENT_SELECTION.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != tag['component_selection_sha256']:
            raise ValueError('Raw selection SHA mismatch')
        doc = json.loads(raw)
        cfg = self.config
        if (doc['version'] != 'native_goal_selection_v1' or doc['workspace'] != str(self.workspace)
                or doc['root_session'] != cfg['root_session'] or doc['session_id'] != runtime.session_id
                or runtime.session_id != cfg['root_session'] or doc['target_ref'] != cfg['target_ref']):
            raise ValueError('Selection Session/target/workspace mismatch')
        if file_sha(cfg['model_config']) != cfg['model_config_sha256']:
            raise ValueError('Model configuration changed')
        if registry_snapshot(self.workspace) != doc['registry_sha256']:
            raise ValueError('Native registry changed')
        targets = yaml_document(self.workspace/'TARGETS.md')['targets']
        target_matches = [t for t in targets if 'target://'+t['id'] == doc['target_ref']]
        if len(target_matches) != 1 or doc['skillruntime_ref'].split('://', 1)[1] not in target_matches[0]['supported_skillruntimes']:
            raise ValueError('Target does not support the selected native skill')
        skills = yaml_document(self.workspace/'SKILLRUNTIME.md')['skillruntimes']
        matches = [s for s in skills if 'skillruntime://'+s['id'] == doc['skillruntime_ref']]
        if len(matches) != 1:
            raise ValueError('Selected native skill missing or duplicated')
        from .native_goal_component_v1 import extract_component
        declaration = extract_component(matches[0])
        if declaration != doc['declaration']:
            raise ValueError('Selected native declaration differs from sidecar')
        return doc

    def _prepare(self, runtime, doc):
        if runtime.step_idx != 0 or doc['initial_observation_id'] != 'libero_obs_0':
            raise ValueError('Derived program must bind at initial native observation')
        for camera in ('agentview_image', 'robot0_eye_in_hand_image'):
            pixels = np.ascontiguousarray(runtime._last_obs[camera])
            actual = dict(sha256=hashlib.sha256(pixels.tobytes()).hexdigest(), shape=list(pixels.shape), dtype=str(pixels.dtype))
            if actual != doc['initial_rgb_by_camera'][camera] or pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[-1] != 3:
                raise ValueError('Initial native camera pixels differ from selection')
        source, calibration, plan = self._artifacts(doc)
        if (runtime.config.get('benchmark_name') != source['source_evidence']['suite'] or
                runtime.config.get('task_id') != source['source_evidence']['task_id'] or
                runtime.config.get('control_mode') != 'relative' or runtime.config.get('max_steps') != 400):
            raise ValueError('Native benchmark or execution contract differs')
        controller = runtime.env.env.robots[0].controller
        expected = source['source_evidence']
        if (file_sha(self.runtime_source) != expected['source_code']['runtime_sha256']
                or file_sha(inspect.getsourcefile(type(controller))) != expected['source_code']['controller_sha256']
                or canonical_sha(controller_config(controller)) != expected['controller_config_sha256']):
            raise ValueError('Native runtime/controller contract differs from source')
        env = runtime.env.env
        camera = env.sim.model.camera_name2id('agentview')
        np.testing.assert_allclose(env.sim.data.cam_xpos[camera], calibration['position'], atol=1e-8, rtol=0)
        np.testing.assert_allclose(env.sim.data.cam_xmat[camera].reshape(3, 3), camera_rotation(calibration['quaternion_wxyz']), atol=1e-7, rtol=0)
        if abs(float(env.sim.model.cam_fovy[camera])-calibration['fovy_degrees']) > 1e-8:
            raise ValueError('Camera fovy mismatch')
        if list(runtime._last_obs['agentview_image'].shape) != [calibration['height'], calibration['width'], 3] or calibration['image_convention'] != 'libero_policy_180_from_bottom_up':
            raise ValueError('Camera dimensions/orientation contract mismatch')
        offsets = doc['derivation_receipt']['offsets']
        if doc['declaration']['version'] == 'native_goal_component_v2':
            from .dual_anchor_retarget_v2 import retarget_dual_anchors as retarget_successor
            derived = retarget_successor(source, plan['anchor_plan'], offsets,
                max_offset_m=plan['max_offset_m'], expected_source_sha256=canonical_sha(source))
        else:
            derived = retarget_dual_anchors(source['waypoints'], plan['anchor_plan'], offsets, max_offset_m=plan['max_offset_m'])
        verification = verify_derived_geometry(source['waypoints'], derived, doc['derived_waypoints'], doc['derivation_receipt']['geometry'])
        derived['verification'] = verification
        program_steps = doc.get('program_steps_limit', len(source['waypoints']))
        # This is deliberately not a mutated compiled/source-success artifact.
        return {'waypoints': copy.deepcopy(doc['derived_waypoints'][:program_steps]), 'kind': 'derived_registered_program'}, derived

    def _artifacts(self, doc):
        from .native_goal_component_v1 import prepare_component
        prepared = prepare_component(self.workspace, doc['declaration'],
            expected_goal=self.config['task_description'],
            expected_model_config_sha256=self.config['model_config_sha256'],
            expected_target_ref=self.config['target_ref'])
        expected_steps = prepared.get('program_steps_limit', len(prepared['source_candidate']['waypoints']))
        declared_steps = doc.get('program_steps_limit', expected_steps)
        if ('execution_contract' in doc['declaration'] and 'program_steps_limit' not in doc
                or type(declared_steps) is not int or declared_steps != expected_steps):
            raise ValueError('Selection execution range differs from the validated component')
        receipt = doc['derivation_receipt']
        if 'execution_contract' in doc['declaration']:
            for key in ('program_steps_limit', 'execution_contract', 'execution_evidence', 'execution_evidence_scope'):
                if receipt.get(key) != prepared['derivation_receipt'].get(key):
                    raise ValueError('Persistent execution evidence differs: '+key)
        for key in ('version', 'declaration_sha256', 'source_artifact_sha256', 'source_waypoints_sha256',
                    'source_rgb_raw_sha256', 'source_rgb_array_sha256', 'calibration_raw_sha256',
                    'model_config_sha256', 'transform_version'):
            if receipt[key] != prepared['derivation_receipt'][key]:
                raise ValueError('Derivation source binding mismatch: '+key)
        if doc['source_artifact_sha256'] != prepared['source_artifact_sha256']:
            raise ValueError('Source raw SHA mismatch')
        rgb = doc['initial_rgb_by_camera']['agentview_image']
        if (receipt['current_rgb_array_sha256'] != rgb['sha256'] or
                receipt['current_rgb_shape'] != rgb['shape'] or receipt['current_rgb_dtype'] != rgb['dtype'] or
                receipt['derived_waypoints_sha256'] != _sha(doc['derived_waypoints'])):
            raise ValueError('Derivation current pixels or geometry digest mismatch')
        return prepared['source_candidate'], prepared['calibration'], doc['declaration']['transform']

    def execute(self, runtime, chunk, original):
        tag = chunk.get('provenance', {}).get('goal_skill')
        if tag is None:
            if runtime.session_id == self.config['root_session']:
                self.root_native_started = True
            return original(runtime, chunk), None
        started = time.perf_counter()
        if self.root_native_started:
            raise ValueError('Cannot return to registered program after native fallback')
        if runtime.done:
            raise ValueError('Tagged program after native terminal')
        required = {'version', 'sha256', 'session_id', 'start_step', 'count', 'component_selection_sha256', 'derived_waypoints_sha256'}
        if not isinstance(tag, dict) or set(tag) != required:
            raise ValueError('Registered program marker shape mismatch')
        doc = self._selection(runtime, tag)
        if self.bound is None:
            program, geometry = self._prepare(runtime, doc)
            self.bound = dict(runtime=runtime, env=runtime.env, selection=tag['component_selection_sha256'], program=program, geometry=geometry, next_step=0)
        bound = self.bound
        if runtime is not bound['runtime'] or runtime.env is not bound['env'] or tag['component_selection_sha256'] != bound['selection']:
            raise ValueError('Program runtime or selection rebinding')
        if runtime.step_idx != bound['next_step']:
            raise ValueError('Program physical history changed or reset')
        waypoints = bound['program']['waypoints']
        expected = marker(doc['source_artifact_sha256'], runtime.session_id, runtime.step_idx, steps=len(waypoints))
        extra = dict(component_selection_sha256=bound['selection'], derived_waypoints_sha256=_sha(doc['derived_waypoints']))
        if tag != dict(expected, **extra):
            raise ValueError('Registered program marker binding mismatch')
        stripped = copy.deepcopy(chunk)
        stripped['provenance']['goal_skill'] = expected
        out, receipt = execute(runtime, stripped, bound['program'], doc['source_artifact_sha256'], self.config['root_session'], True, original)
        bound['next_step'] = runtime.step_idx
        receipt.update(component_selection_sha256=bound['selection'], derived_waypoints_sha256=_sha(doc['derived_waypoints']),
                       executed_prefix_waypoints_sha256=_sha(waypoints), program_steps_limit=len(waypoints),
                       original_registered_marker=copy.deepcopy(tag), program_origin='DERIVED_REGISTERED_PROGRAM',
                       source_artifact_sha256=doc['source_artifact_sha256'], component_consumer_seconds=time.perf_counter()-started,
                       candidate_sha256=canonical_sha(bound['program']), physical_release='UNKNOWN',
                       geometry_verification=copy.deepcopy(bound['geometry']['verification']))
        return out, receipt

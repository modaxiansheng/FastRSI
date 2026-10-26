"""Opt-in native reset restoration for preregistered sampler-extension inputs.

No task, success predicate, Agent input privilege or frozen SDK is changed.
The original reset performs all settling, bookkeeping and observation formatting.
Saved simulator state/XML remain evaluator-only; public observations are original.
"""
import copy
import hashlib
import inspect
import json
from pathlib import Path
import time

VERSION = 'native_sampler_restore_v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bounded_path(root, relative):
    root = Path(root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Collection references must remain within fixed root')
    path = root/rel
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError('Symlink collection reference rejected')
    if not path.is_file():
        raise ValueError('Missing collection artifact')
    return path


def array_binding(value):
    import numpy as np
    array = np.ascontiguousarray(value)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError('Nonfinite or malformed saved state')
    return dict(sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                dtype=str(array.dtype), shape=list(array.shape), nbytes=array.nbytes,
                encoding='sim.get_state().flatten().tobytes(order=C)')


def native_bddl_path(runtime):
    from libero.libero import get_libero_path
    return Path(get_libero_path('bddl_files'))/runtime.task.problem_folder/runtime.task.bddl_file


def registration_role(reg):
    """Explicit input partitions; seeds are pairing IDs, never features."""
    version = reg.get('version')
    if version == 'task64_native_action_adaptation_eval_sampler_v1':
        if reg.get('seeds') != list(range(640781,640787)) or reg.get('role') != 'PREREGISTERED_EVALUATION_INPUT':
            raise ValueError('Action adaptation evaluation sampler partition differs')
        return 'PREREGISTERED_EVALUATION_INPUT'
    if version == 'task64_native_local_bowl_batch_development_sampler_v1':
        if reg.get('seeds') != list(range(640771,640777)) or reg.get('role') != 'DEVELOPMENT_INPUT':
            raise ValueError('Local bowl batch development sampler partition differs')
        return 'DEVELOPMENT_INPUT'
    if version == 'task64_native_sampler_v1':
        if reg.get('seeds') != [640701,640702,640703,640704] or reg.get('role') not in (None,'PREREGISTERED_EVALUATION_INPUT'):
            raise ValueError('Legacy evaluation sampler partition differs')
        return 'PREREGISTERED_EVALUATION_INPUT'
    if version == 'task64_native_development_sampler_v1':
        if reg.get('seeds') != [640711,640712] or reg.get('role') != 'DEVELOPMENT_INPUT':
            raise ValueError('Development sampler partition differs')
        return 'DEVELOPMENT_INPUT'
    if version == 'task64_native_continuation_development_sampler_v1':
        if reg.get('seeds') != [640713,640714] or reg.get('role') != 'DEVELOPMENT_INPUT':
            raise ValueError('Continuation development sampler partition differs')
        return 'DEVELOPMENT_INPUT'
    if version == 'task64_native_successor_development_sampler_v1':
        if reg.get('seeds') != list(range(640715, 640723)) or reg.get('role') != 'DEVELOPMENT_INPUT':
            raise ValueError('Successor development sampler partition differs')
        return 'DEVELOPMENT_INPUT'
    if version == 'task64_native_scope_continuation_learning_sampler_v1':
        if reg.get('seeds') != [640731,640732] or reg.get('role') != 'DEVELOPMENT_INPUT':
            raise ValueError('Scope continuation learning development sampler partition differs')
        return 'DEVELOPMENT_INPUT'
    if version == 'task64_native_early60_reserved_sampler_v1':
        if reg.get('seeds') != [640741,640742] or reg.get('role') != 'PREREGISTERED_EVALUATION_INPUT':
            raise ValueError('Early60 fixed-version reserved evaluation partition differs')
        return 'PREREGISTERED_EVALUATION_INPUT'
    if version == 'task64_native_fresh_cost_development_sampler_v1':
        if reg.get('seeds') != [640751,640752] or reg.get('role') != 'DEVELOPMENT_INPUT':
            raise ValueError('Fresh cost development sampler partition differs')
        return 'DEVELOPMENT_INPUT'
    raise ValueError('Unregistered sampler protocol')


class NativeSamplerRestore:
    def __init__(self, config, *, runtime_source, robot8):
        import numpy as np
        self.config = copy.deepcopy(config)
        if self.config.get('version') != VERSION:
            raise ValueError('Unknown sampler restoration mode')
        self.runtime_source = str(runtime_source)
        self.robot8 = robot8
        self.last_receipt = None
        self.used = False
        self.registration = json.loads(Path(config['registration']).read_text())
        if digest(config['registration']) != config['registration_sha256']:
            raise ValueError('Sampler registration SHA mismatch')
        reg = self.registration
        self.input_role = registration_role(reg)
        if (reg['suite'],reg['task_id'],reg['num_steps_wait'],reg['evaluation_max_steps']) != ('libero_90',64,10,400):
            raise ValueError('Unsupported task or reset protocol')
        if digest(runtime_source) != reg['source_bindings']['runtime_sha256']:
            raise ValueError('Native runtime source differs')
        self.root = Path(config['collection_root']).resolve()
        manifest_path = bounded_path(self.root, 'MANIFEST.json')
        if digest(manifest_path) != config['collection_manifest_sha256']:
            raise ValueError('Collection manifest SHA mismatch')
        self.manifest = json.loads(manifest_path.read_text())
        sample = config['sample_id']
        if sample not in ['sample_'+str(seed) for seed in reg['seeds']]:
            raise ValueError('Sample not in the preregistered native draws')
        self.sample = sample
        # Bind the complete sample, including state/XML and public reference.
        for name, expected in self.manifest.items():
            if name.startswith(sample+'/') or name == 'RESULT.json':
                if digest(bounded_path(self.root, name)) != expected:
                    raise ValueError('Collection bytes changed: '+name)
        collection_result = json.loads(bounded_path(self.root, 'RESULT.json').read_text())
        if (collection_result['registration_sha256'] != config['registration_sha256'] or
                collection_result['generator_sha256'] != reg['generator_sha256']):
            raise ValueError('Collection registration/generator lineage mismatch')
        result_role = collection_result.get('role')
        if (result_role != self.input_role and
                not (reg['version'] == 'task64_native_sampler_v1' and result_role is None)):
            raise ValueError('Collection result role does not match registered partition')
        self.receipt = self._read('COLLECTION_RECEIPT.json')
        if self.receipt.get('role') != self.input_role:
            raise ValueError('Sample role does not match registered partition')
        if self.receipt['status'] != 'COLLECTED_RETAINED' or self.receipt['settling_actions'] != 10:
            raise ValueError('Sample was not fully collected with the shared settling rule')
        self.public = self._read('public_observation/OBSERVATION.json')
        if self.public['task_description'] != reg['task_description']:
            raise ValueError('Sample task goal changed')
        self.xml = self._read_bytes('evaluation_only/MODEL.xml').decode()
        self.state = np.load(self._file('evaluation_only/PRE_SETTLE_STATE.npy'), allow_pickle=False)
        self.post_state = np.load(self._file('evaluation_only/POST_SETTLE_STATE.npy'), allow_pickle=False)
        if array_binding(self.state) != self.receipt['pre_settle_state'] or array_binding(self.post_state) != self.receipt['post_settle_state']:
            raise ValueError('Saved native state binding mismatch')
        self.assets = self._read('evaluation_only/ASSET_MANIFEST.json')
        if not self.assets:
            raise ValueError('Missing bound native assets')
        for path, expected in self.assets.items():
            if digest(path) != expected['sha256'] or Path(path).stat().st_size != expected['size']:
                raise ValueError('Frozen native asset differs')

    def _file(self, relative):
        name = self.sample+'/'+relative
        if name not in self.manifest:
            raise ValueError('Sample artifact absent from fixed collection manifest')
        path = bounded_path(self.root, name)
        if digest(path) != self.manifest[name]:
            raise ValueError('Sample artifact changed after binding')
        return path

    def _read_bytes(self, relative):
        return self._file(relative).read_bytes()

    def _read(self, relative):
        return json.loads(self._read_bytes(relative))

    def reset(self, runtime, context, original_reset):
        import numpy as np
        if self.used:
            raise ValueError('Sampler restoration is single-use per target process')
        sid = context.get('session_id', runtime.session_id)
        if sid != self.config['root_session']:
            raise ValueError('Sampler input cannot silently bind a child/recovery Session')
        cfg = {**runtime.config, **context.get('libero', {})}
        if (cfg.get('benchmark_name'),cfg.get('task_id'),cfg.get('camera_height'),cfg.get('camera_width'),
                cfg.get('num_steps_wait'),cfg.get('max_steps'),cfg.get('control_mode')) != ('libero_90',64,256,256,10,400,'relative'):
            raise ValueError('Native reset changed task/camera/settling/budget/control contract')
        # Recheck immutable source inputs before any reset, without disclosing
        # them in context or returned observations.
        if digest(self.config['registration']) != self.config['registration_sha256'] or digest(self.root/'MANIFEST.json') != self.config['collection_manifest_sha256']:
            raise ValueError('Restoration registration or manifest changed')
        for name in ('evaluation_only/MODEL.xml','evaluation_only/PRE_SETTLE_STATE.npy','evaluation_only/POST_SETTLE_STATE.npy','public_observation/OBSERVATION.json'):
            self._file(name)
        self.used = True
        start = time.perf_counter()
        receipt = self.last_receipt = dict(version=VERSION,status='RUNNING',session_id=sid,
            collection_manifest_sha256=self.config['collection_manifest_sha256'],
            sample_id=self.sample,scope='Evaluator-only sampler input identity, never an Agent feature or official fixed50 score',
            xml_restore_calls=0,state_restore_calls=0,settling_steps=0,settling_attempts=0,
            native_reset_bookkeeping_reused=True,policy_actions=0,model_calls=0)
        wrapper = None
        old_reset = old_state = old_step = None
        try:
            runtime.config.update(context.get('libero', {}))
            runtime._ensure_env()
            wrapper = runtime.env
            if digest(native_bddl_path(runtime)) != self.registration['source_bindings']['bddl_sha256']:
                raise ValueError('Loaded original BDDL bytes differ')
            if runtime.language != self.registration['task_description']:
                raise ValueError('Loaded native task language differs')
            wrapper_source = inspect.getsourcefile(type(wrapper))
            if digest(wrapper_source) != self.registration['source_bindings']['env_wrapper_sha256']:
                raise ValueError('Native environment wrapper changed')
            old_reset,old_state,old_step = wrapper.reset,wrapper.set_init_state,wrapper.step
            def restored_reset():
                if receipt['xml_restore_calls']:
                    raise ValueError('Unexpected repeated reset')
                receipt['xml_restore_calls']+=1
                return wrapper.reset_from_xml_string(self.xml)
            def restored_state(ignored_official_state):
                if receipt['xml_restore_calls'] != 1 or receipt['state_restore_calls']:
                    raise ValueError('Unexpected repeated state assignment')
                receipt['state_restore_calls']+=1
                return old_state(self.state.copy())
            def settling(action):
                if receipt['state_restore_calls'] != 1 or receipt['settling_steps'] >= 10 or list(action) != [0.,0.,0.,0.,0.,0.,-1.]:
                    raise ValueError('Unexpected action during native reset')
                receipt['settling_attempts']+=1
                before=array_binding(wrapper.get_sim_state())
                try:
                    out = old_step(action)
                except BaseException:
                    receipt['failed_settling_call_state_before']=before
                    receipt['failed_settling_call_state_after']=array_binding(wrapper.get_sim_state())
                    receipt['failed_settling_call_physical_steps']='UNKNOWN; retain state change and full reset cost'
                    raise
                receipt['settling_steps']+=1
                return out
            wrapper.reset,wrapper.set_init_state,wrapper.step = restored_reset,restored_state,settling
            output = original_reset(runtime,context)
            if runtime.env is not wrapper:
                raise ValueError('Native reset replaced the bound environment')
            actual_state = array_binding(wrapper.get_sim_state())
            receipt['post_settle_state']=actual_state
            actual_rgb={}
            for camera in ('agentview_image','robot0_eye_in_hand_image'):
                rgb=np.ascontiguousarray(runtime._last_obs[camera])
                actual_rgb[camera]=dict(shape=list(rgb.shape),dtype=str(rgb.dtype),raw_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
            receipt['public_rgb']=actual_rgb
            actual_robot=self.robot8(runtime._last_obs).tolist()
            receipt['public_robot8']=actual_robot
            if (receipt['xml_restore_calls'],receipt['state_restore_calls'],receipt['settling_steps']) != (1,1,10):
                raise ValueError('Native reset did not perform exactly one restore and10 settling steps')
            if actual_state != self.receipt['post_settle_state']:
                raise ValueError('Post-settle simulator state differs from frozen sample')
            for camera,actual in actual_rgb.items():
                if actual != {k:self.public['rgb'][camera][k] for k in ('shape','dtype','raw_sha256')}:
                    raise ValueError('Post-settle public camera differs: '+camera)
            if actual_robot != self.public['robot8']:
                raise ValueError('Post-settle public robot8 differs')
            if runtime.step_idx != 0 or runtime.success or runtime.done or runtime._episode_chunks:
                raise ValueError('Original reset bookkeeping changed')
            receipt.update(status='MATCHED_FROZEN_INPUT',public_rgb_exact=True,public_robot8_exact=True,
                full_state_exact=True,task_success='NOT_ASSESSED_BY_RESTORER')
            return output
        except BaseException as error:
            receipt.update(status='FAILED',error=repr(error))
            raise
        finally:
            if wrapper is not None and old_reset is not None:
                wrapper.reset,wrapper.set_init_state,wrapper.step=old_reset,old_state,old_step
            receipt['seconds']=time.perf_counter()-start


def install(runtime_module, config):
    """Install only on this imported native module; None leaves it untouched."""
    if config is None:
        return None
    original=runtime_module.LiberoRealRuntime.reset
    restorer=NativeSamplerRestore(config,runtime_source=runtime_module.__file__,robot8=runtime_module._libero_state)
    def reset(runtime,context):
        return restorer.reset(runtime,context,original)
    runtime_module.LiberoRealRuntime.reset=reset
    return restorer

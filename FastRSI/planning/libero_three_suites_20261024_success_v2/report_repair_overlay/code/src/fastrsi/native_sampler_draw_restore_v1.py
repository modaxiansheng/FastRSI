"""Opt-in exact native-draw recovery; retain the failed XML protocol unchanged.

A draw is never selected by observations or task outcome. Registered seed,
pre-state, XML, post-state, dual RGB and robot8 must all match original bytes.
The native reset still owns bookkeeping and the ten settling steps.
"""
import copy
import hashlib
import inspect
import random
from pathlib import Path
import time
import numpy as np
from .native_sampler_restore_v1 import NativeSamplerRestore, array_binding, digest, native_bddl_path

VERSION = 'native_sampler_draw_restore_v1'


class NativeSamplerDrawRestore(NativeSamplerRestore):
    def __init__(self, config, *, runtime_source, robot8):
        if config.get('version') != VERSION:
            raise ValueError('Explicit native draw restore opt-in required')
        base = copy.deepcopy(config)
        base['version'] = 'native_sampler_restore_v1'
        super().__init__(base, runtime_source=runtime_source, robot8=robot8)
        self.config = copy.deepcopy(config)

    def reset(self, runtime, context, original_reset):
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
            native_draw_calls=0,native_draw_attempts=0,state_assignment_bypasses=0,settling_steps=0,settling_attempts=0,
            native_reset_bookkeeping_reused=True,policy_actions=0,model_calls=0)
        wrapper = None
        old_reset = old_state = old_step = None
        try:
            if runtime.env is not None:
                raise ValueError('Draw restoration requires a fresh native runtime')
            # Import before the construction seed, as in the original collector.
            from libero.libero import benchmark, get_libero_path
            from libero.libero.envs import OffScreenRenderEnv
            import libero.libero.envs.bddl_base_domain as domain
            if digest(inspect.getsourcefile(domain)) != self.registration['source_bindings']['bddl_base_domain_sha256']:
                raise ValueError('Native sampling implementation changed')
            if self.registration['construction_seed'] != 7:
                raise ValueError('Unregistered construction seed')
            np.random.seed(7); random.seed(7)
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
            raw_draw = {}
            def restored_reset():
                if receipt['native_draw_calls']:
                    raise ValueError('Unexpected repeated native draw')
                receipt['native_draw_calls'] += 1
                seed = int(self.sample[len('sample_'):])
                wrapper.seed(seed); random.seed(seed)
                original_inner = wrapper.env.reset
                def counted_reset():
                    receipt['native_draw_attempts'] += 1
                    if receipt['native_draw_attempts'] > self.registration['max_native_reset_attempts']:
                        raise ValueError('Registered native rejection limit exceeded')
                    return original_inner()
                wrapper.env.reset = counted_reset
                try:
                    raw = old_reset()
                finally:
                    wrapper.env.reset = original_inner
                receipt['pre_settle_state'] = array_binding(wrapper.get_sim_state())
                receipt['xml_sha256'] = hashlib.sha256(wrapper.env.model.get_xml().encode()).hexdigest()
                if receipt['pre_settle_state'] != self.receipt['pre_settle_state']:
                    raise ValueError('Native draw pre-state differs; no retries allowed')
                if receipt['xml_sha256'] != self.receipt['xml_sha256']:
                    raise ValueError('Native draw XML differs; no retries allowed')
                if receipt['native_draw_attempts'] != self.receipt['sampling_attempts']:
                    raise ValueError('Native sampling rejection history differs')
                raw_draw['raw'] = raw
                return raw
            def restored_state(ignored_official_state):
                if receipt['native_draw_calls'] != 1 or receipt['state_assignment_bypasses'] or 'raw' not in raw_draw:
                    raise ValueError('Unexpected state assignment')
                if array_binding(wrapper.get_sim_state()) != self.receipt['pre_settle_state']:
                    raise ValueError('Native pre-state changed before assignment bypass')
                receipt['state_assignment_bypasses'] += 1
                # The exact registered state already exists. Do not call the
                # original set_init_state/force observables a second time.
                return raw_draw['raw']
            def settling(action):
                if receipt['state_assignment_bypasses'] != 1 or receipt['settling_steps'] >= 10 or list(action) != [0.,0.,0.,0.,0.,0.,-1.]:
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
            if (receipt['native_draw_calls'],receipt['state_assignment_bypasses'],receipt['settling_steps']) != (1,1,10):
                raise ValueError('Native reset did not perform exactly one draw and10 settling steps')
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


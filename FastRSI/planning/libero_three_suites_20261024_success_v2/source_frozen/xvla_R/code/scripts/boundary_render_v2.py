"""Opt-in camera rendering at action-block boundaries, without skipping physics.

Sensor sampling clocks keep their native cadence. Intermediate image callbacks
reuse their last pixels only while the policy executes a precomputed block. The
policy-facing boundary image follows the original sensor callback. Early terminal
success refreshes a current final image. Full-video recording keeps native render.
"""
import functools


def install(module, enabled=False, emit=lambda value:None):
    original=module.LiberoRealRuntime.action_chunk
    @functools.wraps(original)
    def chunk(self,payload):
        env=self.env.env
        names=('agentview_image','robot0_eye_in_hand_image')
        cameras={name:env._observables[name] for name in names}
        if not all(x.is_enabled() and x.is_active() and x.modality=='image' for x in cameras.values()):
            return original(self,payload)
        optimize=enabled and not self.config.get('record_dir') and len(payload['actions'])>1
        sensors={name:obj._sensor for name,obj in cameras.items()}
        state={'allow':True,'renders':0,'reused':0,'steps':0,'fresh_terminal':False}
        for name,obj in cameras.items():
            sensor=sensors[name]
            def measured(cache, sensor=sensor, obj=obj):
                if state['allow']:
                    state['renders']+=1
                    return sensor(cache)
                state['reused']+=1
                return obj.obs
            # Direct callback assignment preserves the original sensor name,
            # enabled/active state, sampling clock, filter and zero-delay cadence.
            obj._sensor=functools.wraps(sensor)(measured)
        step=self.env.step
        def selected(action):
            index=state['steps']+1
            state['allow']=not optimize or index==len(payload['actions']) or self.step_idx+1>=int(self.config['max_steps'])
            observation,reward,done,info=step(action)
            state['steps']=index
            if bool(done) and not state['allow']:
                state['allow']=True
                for obj in cameras.values():
                    obj.update(timestep=env.model_timestep,obs_cache=env._obs_cache,force=True)
                observation=env._get_observations()
                state['fresh_terminal']=True
            return observation,reward,done,info
        self.env.step=selected
        try:
            return original(self,payload)
        finally:
            self.env.step=step
            for name,obj in cameras.items():obj._sensor=sensors[name]
            emit({'event':'boundary_render_receipt','session_id':self.session_id,
                  'enabled':optimize,'physics_steps':state['steps'],'native_camera_renders':state['renders'],
                  'cached_camera_reads':state['reused'],'early_terminal_image_refreshed':state['fresh_terminal'],
                  'scope':'Native dynamics and per-step done check retained; no intermediate policy request or full-video recording suppressed.'})
    module.LiberoRealRuntime.action_chunk=chunk

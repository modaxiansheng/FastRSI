"""Keep the learned adapter until a public gripper close/open transition.

This is a fixed diagnostic scope rule, not a learned selector or a completed
subgoal assertion. The full goal, H10 feedback and native policy remain intact.
"""
import json
import numpy as np
import serve_bowl_lora_source_policy_v1 as parent
from bowl_expert_lora_v1 import enable

native = parent.native
original_init = native.NativeOpenPIPolicy.__init__
original_infer = native.NativeOpenPIPolicy.infer

def initialize(self, **kwargs):
    original_init(self, **kwargs)
    from bowl_expert_lora_v1 import ExpertLoRALinear
    self._phase_modules = {name: m for name, m in self.policy._model.named_modules()
                          if isinstance(m, ExpertLoRALinear)}
    assert len(self._phase_modules)==36
    self._phase_history = {}

def infer(self, observation):
    case = observation['_source_case']
    index = observation['_source_chunk']
    state = np.asarray(observation['observation/state'])
    assert state.shape==(8,) and np.isfinite(state).all()
    # Native Panda finger coordinates have an open endpoint of 0.04 m.
    # These fixed mechanical bands are not fitted to timing/goal labels.
    aperture = float(np.mean(np.abs(state[6:8])))
    if index==0:
        self._phase_history[case] = dict(closed=False, withdrawn=False)
    history = self._phase_history[case]
    if aperture<.02:
        history['closed']=True
    if history['closed'] and aperture>.03:
        history['withdrawn']=True
    active = not history['withdrawn']
    enable(self._phase_modules,active)
    with (parent.source.ROOT/'PHASE_GATE.jsonl').open('a') as f:
        f.write(json.dumps(dict(case=case,chunk=index,public_finger_coordinate_mean=aperture,
            adapter_active=active,subgoal_completion_claim=False,
            rule='Withdraw after first observed finger closing then opening'))+'\n')
    return original_infer(self,observation)

native.NativeOpenPIPolicy.__init__ = initialize
native.NativeOpenPIPolicy.infer = infer
if __name__=='__main__':native.main()

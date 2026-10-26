"""Opt-in inference-length experiment using the frozen native policy loader.

H20 is a different sampler configuration, not an unused tail of H10 output.
No weights, transforms, denoising iterations or frozen source files are edited.
"""
import dataclasses
import hashlib
import json
import threading

import numpy as np


VERSION = "experimental_pi05_inference_horizon_v1"
_LOAD_LOCK = threading.Lock()


def _components():
    from PhyAgentOS.runtime.policy.openpi.native_openpi_server import NativeOpenPIPolicy
    from openpi.training import config
    return NativeOpenPIPolicy, config


def _config_hash(config):
    if not dataclasses.is_dataclass(config):
        raise ValueError("Expected the audited dataclass model config")
    return hashlib.sha256(json.dumps(dataclasses.asdict(config), sort_keys=True,
                                    default=str).encode()).hexdigest()


class ExperimentalHorizonPolicy:
    """Resident-compatible wrapper, with one fixed H for its entire lifetime."""
    def __init__(self, *, checkpoint_dir, action_horizon=10,
                 pytorch_device="cuda:0", torch_seed=7):
        if type(action_horizon) is not int or action_horizon not in (10, 20):
            raise ValueError("Only explicit H10/H20 are supported")
        self.action_horizon = action_horizon
        factory, config_module = _components()
        arguments = dict(policy_config="pi05_libero", checkpoint_dir=str(checkpoint_dir),
                         default_prompt=None, pytorch_device=pytorch_device,
                         disable_torch_compile=True, torch_seed=torch_seed)
        # The server is not serving yet. Serialize this experiment's constructors;
        # restore the module function before constructing any resident RNG snapshot.
        with _LOAD_LOCK:
            if action_horizon == 10:
                native = factory(**arguments)  # Exact original loading route.
            else:
                original = config_module.get_config

                def get_config(name):
                    if name != "pi05_libero":
                        raise ValueError("Unexpected configuration request during H20 loading")
                    base = original(name)
                    if base.model.action_horizon != 10 or base.model.pi05 is not True:
                        raise ValueError("Frozen pi05_libero reference config differs")
                    return dataclasses.replace(base, model=dataclasses.replace(
                        base.model, action_horizon=action_horizon))

                config_module.get_config = get_config
                try:
                    native = factory(**arguments)
                finally:
                    config_module.get_config = original
        self._native = native
        self.policy = native.policy
        self.policy_config = native.policy_config
        self._model = self.policy._model
        self._fixed_config = self._model.config
        self._config_sha256 = _config_hash(self._fixed_config)
        self._fixed_horizon = action_horizon
        self.metadata = {**native.metadata, "experimental_horizon": {
            "version": VERSION, "reference_horizon": 10,
            "inference_horizon": action_horizon,
            "model_config_sha256": self._config_sha256,
            "model_config": dataclasses.asdict(self._fixed_config),
            "same_checkpoint_requested": str(checkpoint_dir),
            "changed_prediction_length": action_horizon != 10,
            "robot_quality_or_speed_established": False,
        }}
        self.check_configuration()

    def check_configuration(self):
        c = self._model.config
        if (self.action_horizon != self._fixed_horizon or self.policy is not self._native.policy
                or self.policy._model is not self._model or c is not self._fixed_config
                or _config_hash(c) != self._config_sha256):
            raise ValueError("Horizon policy configuration changed; reload required")
        if (self.policy_config != "pi05_libero" or self._native.policy_config != "pi05_libero"
                or c.action_horizon != self._fixed_horizon or c.pi05 is not True
                or c.pytorch_compile_mode is not None):
            raise ValueError("Unexpected loaded pi05 inference configuration")
        for metadata in (self._native.metadata, self.metadata):
            if (metadata.get("action_horizon") != self._fixed_horizon
                    or metadata.get("action_dim") != 7
                    or metadata.get("torch_compile_disabled") is not True):
                raise ValueError("Native metadata does not match fixed horizon")

    def infer(self, observation):
        self.check_configuration()
        output = self._native.infer(observation)
        self.check_configuration()
        actions = np.asarray(output.get("actions"))
        if (actions.shape != (self._fixed_horizon, 7)
                or actions.dtype != np.dtype("float32") or not np.isfinite(actions).all()):
            raise ValueError("Expected finite float32 output with exact configured horizon")
        metadata = output.get("policy_meta", {})
        if metadata.get("chunk_size") != self._fixed_horizon or metadata.get("action_dim") != 7:
            raise ValueError("Output metadata does not match action shape")
        return output

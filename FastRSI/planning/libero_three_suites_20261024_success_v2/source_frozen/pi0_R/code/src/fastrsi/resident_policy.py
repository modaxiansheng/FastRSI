"""Experimental, exclusive-session reuse of the audited eager OpenPI PyTorch policy.

This is a common deployment optimization, not a learned policy. Construct directly
after model loading, before ANY inference. Reset restores the actual post-load RNG
snapshot, not a seed. Only the audited pi05 LIBERO implementation is supported.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import inspect
import json
import pickle
import random
import threading
import time
from pathlib import Path

VERSION = "resident_openpi_postload_reset_v2"
ADMIN_KEY = "__fastrsi_resident__"
AUDITED_SOURCES = {
    "openpi.policies.policy": "a2ac5236e23b11d0f5a840d4a8fef148d89dba921d5452ec5dfcd5a6ffa46f20",
    "openpi.models_pytorch.pi0_pytorch": "dcd7b6f508808ce4b7170a1720a4c62c6da04edae3652d16c94a469dca924e16",
    "openpi.models_pytorch.gemma_pytorch": "225092e06509da0ae37bb9dca55c00d2804255bb1496f7c8700304b4c52a95eb",
    "openpi.models_pytorch.preprocessing_pytorch": "6a44e91be2d7a724c6140d2cc98ed0a631b8d38bbb29f63b380a0a19f903bbf6",
    "openpi.transforms": "a1b94e9e72849a18834778f229c6bb389a495eb7fbe0aa800edea728b9424ff4",
    "openpi.policies.libero_policy": "d8679d99facc3fef48da07406081b20709ed872fcefdf7de67ede99422d623fa",
    "openpi.models.tokenizer": "965be8b3c393a6811875bbc32da9e01a5d01cc2f87802de801cf7293e049748c",
}


def _hash(value):
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


class ResidentProtocolError(RuntimeError):
    pass


class RNGSnapshot:
    """No random draws. CUDA includes every visible device, including ROCm torch."""
    def __init__(self, *, numpy_module=None, torch_module=None):
        self.np = numpy_module or importlib.import_module("numpy")
        self.torch = torch_module or importlib.import_module("torch")
        self.python = random.getstate()
        self.numpy = copy.deepcopy(self.np.random.get_state())
        self.cpu = self.torch.get_rng_state().clone()
        self.cuda_count = self.torch.cuda.device_count() if self.torch.cuda.is_available() else 0
        self.cuda = [x.clone() for x in self.torch.cuda.get_rng_state_all()] if self.cuda_count else []
        if len(self.cuda) != self.cuda_count:
            raise ResidentProtocolError("visible CUDA RNG snapshot incomplete")
        self.hashes = {
            "python": _hash(self.python), "numpy": _hash(self.numpy),
            "torch_cpu": _hash(self.cpu.cpu().numpy().tobytes()),
            "torch_cuda": [_hash(x.cpu().numpy().tobytes()) for x in self.cuda],
        }
        self.digest = _hash(self.hashes)

    def restore(self):
        count = self.torch.cuda.device_count() if self.torch.cuda.is_available() else 0
        if count != self.cuda_count:
            raise ResidentProtocolError("visible CUDA device count changed")
        # A reset is a device barrier, not merely a host-side seed operation.
        for i in range(count):
            self.torch.cuda.synchronize(i)
        random.setstate(self.python)
        self.np.random.set_state(copy.deepcopy(self.numpy))
        self.torch.set_rng_state(self.cpu.clone())
        if count:
            self.torch.cuda.set_rng_state_all([x.clone() for x in self.cuda])
        current = RNGSnapshot(numpy_module=self.np, torch_module=self.torch)
        if current.hashes != self.hashes:
            raise ResidentProtocolError("post-reset RNG hash mismatch")
        return current.hashes


class OpenPIStateGuard:
    """Reset only two known attention-config writes; reject unrecognized state.

    In this exact source version KV caches are local infer variables initialized
    from None on every call. JAX _rng, recorders and compiled policies are rejected.
    Tensor identity/version checks detect ordinary model mutation; they are not a
    cryptographic full-weight comparison or proof of numerical reproducibility.
    """
    def __init__(self, native):
        self.native, self.policy = native, native.policy
        p = self.policy
        if (type(p).__module__, type(p).__name__) != ("openpi.policies.policy", "Policy"):
            raise ResidentProtocolError("unsupported policy wrapper")
        if (native.policy_config not in ("pi0_libero", "pi05_libero") or not p._is_pytorch_model
                or hasattr(p, "_rng") or not native.metadata.get("torch_compile_disabled")):
            raise ResidentProtocolError("only eager PyTorch pi05_libero is audited")
        expected_keys = {"_model", "_input_transform", "_output_transform", "_sample_kwargs",
                         "_metadata", "_is_pytorch_model", "_pytorch_device", "_sample_actions"}
        if set(vars(p)) != expected_keys:
            raise ResidentProtocolError("unsupported persistent Policy attributes")
        self.sources = {}
        for module, expected in AUDITED_SOURCES.items():
            path = Path(inspect.getfile(importlib.import_module(module)))
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ResidentProtocolError(f"unaudited source: {module}")
            self.sources[module] = actual
        m = p._model
        if (type(m).__module__, type(m).__name__) != ("openpi.models_pytorch.pi0_pytorch", "PI0Pytorch"):
            raise ResidentProtocolError("unsupported model")
        if m.config.pytorch_compile_mode is not None or not inspect.ismethod(p._sample_actions):
            raise ResidentProtocolError("compiled or replaced sampler unsupported")
        if p._sample_actions.__self__ is not m:
            raise ResidentProtocolError("sampler model binding mismatch")
        self.configs = [m.paligemma_with_expert.paligemma.language_model.config,
                        m.paligemma_with_expert.gemma_expert.model.config]
        self.attention = [c._attn_implementation for c in self.configs]
        self.policy_keys = set(vars(p))
        self.baseline = self._fingerprint()
        self.digest = _hash(self.baseline)

    def _transform_state(self, obj):
        name = f"{type(obj).__module__}.{type(obj).__name__}"
        allowed = {f"openpi.transforms.{n}" for n in (
            "CompositeTransform", "InjectDefaultPrompt", "Normalize", "Unnormalize",
            "ResizeImages", "TokenizePrompt", "PadStatesAndActions", "DeltaActions", "AbsoluteActions")}
        allowed |= {"openpi.policies.libero_policy.LiberoInputs", "openpi.policies.libero_policy.LiberoOutputs"}
        if name not in allowed or not dataclasses.is_dataclass(obj):
            raise ResidentProtocolError(f"unsupported transform {name}")
        if name.endswith(".CompositeTransform"):
            return name, tuple(self._transform_state(t) for t in obj.transforms)
        fields = dict(vars(obj))
        if name.endswith(".TokenizePrompt"):
            tokenizer = fields.pop("tokenizer")
            if (type(tokenizer).__module__, type(tokenizer).__name__) != ("openpi.models.tokenizer", "PaligemmaTokenizer"):
                raise ResidentProtocolError("unsupported tokenizer")
            fields["tokenizer"] = (tokenizer._max_len, tokenizer._tokenizer.serialized_model_proto())
        return name, _hash(fields)

    def _fingerprint(self):
        p, m = self.policy, self.policy._model
        if set(vars(p)) != self.policy_keys:
            raise ResidentProtocolError("policy acquired unsupported state")
        modules = []
        for name, sub in m.named_modules():
            if sub.training:
                raise ResidentProtocolError(f"training module unsupported: {name}")
            for key, value in vars(sub).items():
                if ("cache" in key or "past_key_value" in key) and value is not None:
                    # Default RoPE has an immutable integer max_seq_len_cached.
                    if key not in {"max_seq_len_cached"}:
                        raise ResidentProtocolError(f"persistent cache unsupported: {name}.{key}")
            if getattr(sub, "rope_type", "default") != "default":
                raise ResidentProtocolError("dynamic RoPE state unsupported")
            scalar_state = tuple(sorted((key, value) for key, value in vars(sub).items()
                                        if value is None or type(value) in (bool, int, float, str)))
            modules.append((name, type(sub).__module__, type(sub).__name__,
                            tuple(sorted(vars(sub))), scalar_state))
        tensors = []
        for kind, iterator in (("parameter", m.named_parameters()), ("buffer", m.named_buffers())):
            for name, tensor in iterator:
                tensors.append((kind, name, id(tensor), tensor.data_ptr(), tuple(tensor.shape),
                                str(tensor.dtype), str(tensor.device), tensor._version))
        return (tuple(modules), tuple(tensors), _hash(p._sample_kwargs), _hash(p._metadata),
                self._transform_state(p._input_transform), self._transform_state(p._output_transform))

    def check(self):
        if self._fingerprint() != self.baseline:
            raise ResidentProtocolError("non-RNG model/transform state changed; reload required")

    def reset(self):
        self.check()
        for config, original in zip(self.configs, self.attention):
            if config._attn_implementation not in (original, "eager"):
                raise ResidentProtocolError("unrecognized attention implementation mutation")
            config._attn_implementation = original
        return {"state_sha256": self.digest, "restored_fields": [
            "paligemma.language_model.config._attn_implementation",
            "gemma_expert.model.config._attn_implementation"],
            "persistent_kv_cache": False, "sources_sha256": self.sources}


class ResidentPolicy:
    def __init__(self, native, *, rng=None, guard=None, max_sessions=64, max_inferences=1024):
        if any(type(x) is not int or x < 1 for x in (max_sessions, max_inferences)):
            raise ResidentProtocolError("positive integer server limits required")
        self.native = native
        self.guard = guard if guard is not None else OpenPIStateGuard(native)
        self.rng = rng if rng is not None else RNGSnapshot()
        self.max_sessions, self.max_inferences = max_sessions, max_inferences
        self.lock = threading.RLock()
        self.generation = 0
        self.active = None
        self.connections = {}
        self.receipts = {}
        self.used_sessions = set()
        self.broken = False
        self.metadata = {**native.metadata, "resident": {"version": VERSION,
            "begin_required": True, "exclusive_session": True, "rng_sha256": self.rng.digest}}

    def connect(self, connection_id):
        with self.lock:
            if connection_id in self.connections:
                raise ResidentProtocolError("duplicate connection")
            self.connections[connection_id] = self.generation

    def disconnect(self, connection_id):
        with self.lock:
            self.connections.pop(connection_id, None)

    def admin(self, command):
        with self.lock:
            op = command.get("op")
            if op == "status":
                return {"version": VERSION, "generation": self.generation, "broken": self.broken,
                        "active": copy.deepcopy(self.active), "rng_sha256": self.rng.digest,
                        "owner_connected": bool(self.active and self.active["owner"] in self.connections)}
            if op not in {"begin", "end", "handoff"}:
                raise ResidentProtocolError("unknown admin operation")
            request_id, session_id = command.get("request_id"), command.get("session_id")
            if not all(isinstance(x, str) and 0 < len(x) <= 160 for x in (request_id, session_id)):
                raise ResidentProtocolError("nonempty request_id and session_id required")
            if request_id in self.receipts:
                old, receipt = self.receipts[request_id]
                if old != command:
                    raise ResidentProtocolError("request_id reused with different command")
                return copy.deepcopy(receipt)  # Never rewind on retry, including after inference.
            if self.broken:
                raise ResidentProtocolError("server poisoned; fresh process required")
            started = time.perf_counter()
            if op == "begin":
                limit = command.get("max_inferences")
                if type(limit) is not int or not 1 <= limit <= self.max_inferences:
                    raise ResidentProtocolError("invalid bounded inference limit")
                if self.active is not None or session_id in self.used_sessions:
                    raise ResidentProtocolError("end previous session; never reuse session_id")
                if len(self.used_sessions) >= self.max_sessions:
                    raise ResidentProtocolError("session limit reached")
                try:
                    state = self.guard.reset()
                    hashes = self.rng.restore()
                except Exception:
                    self.broken = True
                    raise
                self.generation += 1
                self.used_sessions.add(session_id)
                self.active = {"session_id": session_id, "generation": self.generation,
                               "max_inferences": limit, "inferences": 0, "owner": None}
                receipt = {"version": VERSION, "op": op, "session_id": session_id,
                           "generation": self.generation, "request_id": request_id,
                           "rng_sha256": self.rng.digest, "rng_hashes": hashes,
                           "policy_state": state, "reset_seconds": time.perf_counter() - started}
            elif op == "handoff":
                a = self.active
                if not a or a["session_id"] != session_id:
                    raise ResidentProtocolError("session not active")
                expected = command.get("expected_generation")
                if type(expected) is not int or expected != a["generation"]:
                    raise ResidentProtocolError("handoff generation mismatch")
                if a["owner"] is None:
                    raise ResidentProtocolError("no previous inference owner to hand off")
                if a["owner"] in self.connections:
                    raise ResidentProtocolError("disconnect inference owner before handoff")
                if a["inferences"] >= a["max_inferences"]:
                    raise ResidentProtocolError("session inference limit reached")
                try:
                    before = RNGSnapshot(numpy_module=self.rng.np, torch_module=self.rng.torch)
                    if before.cuda_count != self.rng.cuda_count:
                        raise ResidentProtocolError("visible CUDA device count changed")
                    self.guard.check()  # No model/config reset within a recovery chain.
                    for device in range(before.cuda_count):
                        self.rng.torch.cuda.synchronize(device)
                    after = RNGSnapshot(numpy_module=self.rng.np, torch_module=self.rng.torch)
                    if before.hashes != after.hashes:
                        raise ResidentProtocolError("handoff changed RNG state")
                except Exception:
                    self.broken = True
                    raise
                old_owner = a["owner"]
                self.generation += 1
                a.update(owner=None, generation=self.generation)
                receipt = {"version": VERSION, "op": op, "session_id": session_id,
                           "request_id": request_id, "previous_generation": expected,
                           "generation": self.generation, "previous_owner": old_owner,
                           "inferences": a["inferences"], "max_inferences": a["max_inferences"],
                           "rng_reset": False, "rng_sha256": self.rng.digest,
                           "rng_before_hashes": before.hashes, "rng_after_hashes": after.hashes,
                           "handoff_seconds": time.perf_counter() - started}
            else:
                if not self.active or self.active["session_id"] != session_id:
                    raise ResidentProtocolError("session not active")
                if self.active["owner"] in self.connections:
                    raise ResidentProtocolError("disconnect inference owner before ending session")
                receipt = {"version": VERSION, "op": op, "request_id": request_id,
                           **self.active, "end_rng_hashes": RNGSnapshot(
                               numpy_module=self.rng.np, torch_module=self.rng.torch).hashes}
                self.active = None
            self.receipts[request_id] = (copy.deepcopy(command), copy.deepcopy(receipt))
            return receipt

    def infer(self, observation, *, connection_id):
        with self.lock:
            a = self.active
            if self.broken or a is None:
                raise ResidentProtocolError("no healthy active session")
            if self.connections.get(connection_id) != a["generation"]:
                raise ResidentProtocolError("connect inference client AFTER begin/handoff receipt")
            if a["owner"] not in (None, connection_id):
                raise ResidentProtocolError("session already bound to another inference client")
            if a["inferences"] >= a["max_inferences"]:
                raise ResidentProtocolError("session inference limit reached")
            a["owner"] = connection_id
            a["inferences"] += 1  # Failed attempts count and poison the session.
            try:
                self.guard.check()
                result = self.native.infer(observation)
                self.guard.check()
            except Exception:
                self.broken = True
                raise
            result = dict(result)
            result["policy_meta"] = {**dict(result.get("policy_meta", {})), "resident": {
                "session_id": a["session_id"], "generation": a["generation"],
                "inference_index": a["inferences"], "rng_sha256": self.rng.digest}}
            return result

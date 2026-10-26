"""Msgpack serialization helpers for runtime RPC messages."""

from __future__ import annotations

from typing import Any

import msgpack
import numpy as np

from PhyAgentOS.runtime.communication.envelope import RuntimeEnvelope


def _pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "data": obj.tobytes(),
            "dtype": obj.dtype.str,
            "shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            "__npgeneric__": True,
            "data": obj.item(),
            "dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj: dict[Any, Any]) -> Any:
    if not isinstance(obj, dict):
        return obj
    is_ndarray = obj.get("__ndarray__") or obj.get(b"__ndarray__")
    is_npgeneric = obj.get("__npgeneric__") or obj.get(b"__npgeneric__")
    if is_ndarray:
        data = obj.get("data", obj.get(b"data"))
        if isinstance(data, str):
            data = data.encode("latin1")
        dtype = obj.get("dtype", obj.get(b"dtype"))
        shape = obj.get("shape", obj.get(b"shape"))
        return np.ndarray(buffer=data, dtype=np.dtype(dtype), shape=shape)
    if is_npgeneric:
        data = obj.get("data", obj.get(b"data"))
        dtype = obj.get("dtype", obj.get(b"dtype"))
        return np.dtype(dtype).type(data)
    return obj


def encode_msgpack(envelope: RuntimeEnvelope | dict[str, Any]) -> bytes:
    payload = envelope.model_dump(mode="python") if isinstance(envelope, RuntimeEnvelope) else envelope
    return msgpack.packb(payload, use_bin_type=True, default=_pack_array)


def decode_msgpack(data: bytes) -> RuntimeEnvelope:
    payload = msgpack.unpackb(data, raw=False, object_hook=_unpack_array)
    return RuntimeEnvelope.model_validate(payload)

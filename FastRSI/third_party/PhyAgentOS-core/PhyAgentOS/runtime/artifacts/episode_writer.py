"""Episode artifact writer."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from PhyAgentOS.runtime.schemas import SessionResult, SessionSpec, TargetSpec
from PhyAgentOS.runtime.state_io.atomic_file import atomic_write_text


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


class EpisodeWriter:
    """Write episode-level runtime artifacts under artifacts/runtime."""

    def __init__(self, artifacts_root: Path):
        self.artifacts_root = artifacts_root

    def write_episode(
        self,
        session: SessionSpec,
        target: TargetSpec,
        skillruntime_id: str,
        result: SessionResult,
    ) -> Path:
        artifact_dir = self.artifacts_root / session.session_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        episode_path = artifact_dir / "episode.json"
        benchmark_episode = _benchmark_episode_summary(result.metadata)
        payload = {
            "session_id": session.session_id,
            "target_id": target.id,
            "skillruntime_id": skillruntime_id,
            "benchmark": benchmark_episode,
            "success": result.success,
            "status": result.status,
            "num_steps": result.num_steps,
            "return_value": result.return_value,
            "policy_latency_ms": {
                "mean": result.mean_policy_latency_ms,
            },
            "error_code": result.error_code,
            "error_message": result.error_message,
            "metadata": result.metadata,
        }
        atomic_write_text(episode_path, json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
        return artifact_dir

    def write_rgb_frames(self, artifact_dir: Path, observation: dict[str, Any] | None, *, phase: str = "final") -> list[Path]:
        """Persist every RGB array in the final raw observation as a PNG."""
        if not observation:
            return []
        rgb_dir = artifact_dir / "rgb"
        paths: list[Path] = []
        for index, (name, array) in enumerate(_find_rgb_arrays(observation), start=1):
            rgb_dir.mkdir(parents=True, exist_ok=True)
            path = rgb_dir / f"{phase}_{index:02d}_{_safe_name(name)}.png"
            path.write_bytes(_encode_rgb_png(array))
            paths.append(path)
        return paths


def _benchmark_episode_summary(metadata: dict[str, Any]) -> dict[str, Any] | None:
    final_status = metadata.get("final_status")
    if not isinstance(final_status, dict):
        return None
    for key in ("benchmark_episode", "episode_summary", "benchmark"):
        summary = final_status.get(key)
        if isinstance(summary, dict):
            return summary
    return None


def _find_rgb_arrays(value: Any, prefix: str = "rgb") -> list[tuple[str, np.ndarray]]:
    found: list[tuple[str, np.ndarray]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(_find_rgb_arrays(item, f"{prefix}_{key}"))
        return found
    if not isinstance(value, np.ndarray) or value.ndim != 3 or value.dtype != np.uint8:
        return found
    array = value
    if array.shape[-1] == 3:
        found.append((prefix, array))
    elif array.shape[0] == 3:
        found.append((prefix, np.transpose(array, (1, 2, 0))))
    return found


def _safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_")
    return cleaned[:64] or "rgb"


def _encode_rgb_png(array: np.ndarray) -> bytes:
    height, width, channels = array.shape
    if channels != 3:
        raise ValueError(f"RGB frame must have three channels, got {array.shape}")
    raw = b"".join(b"\x00" + array[row].tobytes() for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    signature = b"\x89PNG\r\n\x1a\n"
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return signature + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")

"""Standalone process env bootstrap for BEHAVIOR-1K TargetWS server.

Self-contained (no PhyAgentOS import). Mirrors ``behavior1k_env.py`` so behavior
conda uses its own scipy instead of Isaac ``pip_prebundle``.
"""

from __future__ import annotations

import os
from pathlib import Path

_PYTHONPATH_DROP_MARKERS = (
    "/isaacsim",
    "/isaac-sim",
    "pip_prebundle",
    "/extscache/",
)
_SRC_OMNIGIBSON_MARKERS = ("/src/omnigibson/",)
_DEFAULT_BEHAVIOR1K_ROOT = Path("/home/zyserver/work/BEHAVIOR-1K")
_DEFAULT_ISAAC_PATH = "/home/zyserver/isaacsim3"


def resolve_behavior1k_root(root: str | Path | None = None) -> Path:
    if root:
        path = Path(root).expanduser().resolve()
        if path.is_dir():
            return path
    env_root = os.environ.get("BEHAVIOR1K_ROOT")
    if env_root:
        path = Path(env_root).expanduser().resolve()
        if path.is_dir():
            return path
    if _DEFAULT_BEHAVIOR1K_ROOT.is_dir():
        return _DEFAULT_BEHAVIOR1K_ROOT.resolve()
    raise FileNotFoundError("BEHAVIOR-1K root not found; set BEHAVIOR1K_ROOT")


def behavior1k_omnigibson_paths(behavior1k_root: Path) -> list[str]:
    primary = (behavior1k_root / "OmniGibson").resolve()
    if primary.is_dir():
        return [str(primary)]
    fallback = (behavior1k_root / "src" / "omnigibson" / "OmniGibson").resolve()
    if fallback.is_dir():
        return [str(fallback)]
    return []


def behavior1k_python_paths(behavior1k_root: Path) -> list[str]:
    paths: list[str] = []
    joylo = (behavior1k_root / "joylo").resolve()
    if joylo.is_dir() and (joylo / "gello").is_dir():
        paths.append(str(joylo))
    fallback_joylo = (behavior1k_root / "src" / "omnigibson" / "joylo").resolve()
    if fallback_joylo.is_dir() and (fallback_joylo / "gello").is_dir():
        if str(fallback_joylo) not in paths:
            paths.append(str(fallback_joylo))
    paths.extend(behavior1k_omnigibson_paths(behavior1k_root))
    return paths


def sanitize_pythonpath(existing: str, *, behavior1k_root: Path) -> str:
    kept: list[str] = []
    for part in existing.split(os.pathsep):
        if not part:
            continue
        if any(marker in part for marker in _PYTHONPATH_DROP_MARKERS):
            continue
        if any(marker in part for marker in _SRC_OMNIGIBSON_MARKERS):
            continue
        if part not in kept:
            kept.append(part)
    for extra in reversed(behavior1k_python_paths(behavior1k_root)):
        if extra not in kept:
            kept.insert(0, extra)
    return os.pathsep.join(kept)


def isaac_runtime_python_paths(isaac_path: str) -> list[str]:
    root = Path(isaac_path.rstrip("/"))
    candidates = [
        root / "python_packages",
        root / "exts" / "isaacsim.simulation_app",
        root / "extsDeprecated" / "omni.isaac.kit",
        root / "kit" / "kernel" / "py",
        root / "kit" / "plugins" / "bindings-python",
    ]
    return [str(path) for path in candidates if path.is_dir()]


def _isaac_ld_library_prefix(isaac_path: str) -> list[str]:
    root = isaac_path.rstrip("/")
    candidates = [
        root,
        f"{root}/.",
        f"{root}/exts/omni.usd.schema.isaac/plugins/IsaacSensorSchema/lib",
        f"{root}/exts/omni.usd.schema.isaac/plugins/RangeSensorSchema/lib",
        f"{root}/kit",
        f"{root}/kit/kernel/plugins",
        f"{root}/kit/libs/iray",
        f"{root}/kit/plugins",
        f"{root}/kit/plugins/bindings-python",
        f"{root}/kit/plugins/carb_gfx",
        f"{root}/kit/plugins/rtx",
        f"{root}/kit/plugins/gpu.foundation",
    ]
    return [path for path in candidates if Path(path).exists()]


def sanitize_ld_library_path(existing: str, *, isaac_path: str) -> str:
    kept: list[str] = []
    for part in existing.split(os.pathsep):
        if not part:
            continue
        if "/isaacsim" in part or "/isaac-sim" in part:
            continue
        if part not in kept:
            kept.append(part)
    return os.pathsep.join(_isaac_ld_library_prefix(isaac_path) + kept)


def _path_is_polluted(path: str) -> bool:
    if not path:
        return False
    if any(marker in path for marker in _PYTHONPATH_DROP_MARKERS):
        return True
    return any(marker in path for marker in _SRC_OMNIGIBSON_MARKERS)


def sanitize_sys_path(*, behavior1k_root: Path, isaac_path: str) -> None:
    """Remove stale Isaac prebundle entries already injected at interpreter startup."""
    import sys

    cleaned = [p for p in sys.path if not _path_is_polluted(p)]
    prepend: list[str] = []
    for extra in behavior1k_python_paths(behavior1k_root):
        if extra not in prepend:
            prepend.append(extra)
    for isaac_py in isaac_runtime_python_paths(isaac_path):
        if isaac_py not in prepend and isaac_py not in cleaned:
            cleaned.append(isaac_py)
    sys.path[:] = prepend + [p for p in cleaned if p not in prepend]


def apply_behavior1k_process_env(
    *,
    isaac_path: str | None = None,
    behavior1k_root: str | Path | None = None,
    headless: bool = True,
    display: str | None = None,
) -> dict[str, str]:
    """Sanitize and apply env vars before ``import omnigibson``."""
    b1k_root = resolve_behavior1k_root(behavior1k_root)
    isaac = (isaac_path or os.environ.get("B1K_ISAAC_PATH") or _DEFAULT_ISAAC_PATH).rstrip("/")

    os.environ["ISAAC_PATH"] = isaac
    os.environ["CARB_APP_PATH"] = f"{isaac}/kit"
    os.environ["EXP_PATH"] = f"{isaac}/apps"
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    os.environ["BEHAVIOR1K_ROOT"] = str(b1k_root)

    os.environ["LD_LIBRARY_PATH"] = sanitize_ld_library_path(
        os.environ.get("LD_LIBRARY_PATH", ""),
        isaac_path=isaac,
    )

    py_parts = [p for p in sanitize_pythonpath(os.environ.get("PYTHONPATH", ""), behavior1k_root=b1k_root).split(os.pathsep) if p]
    for isaac_py in isaac_runtime_python_paths(isaac):
        if isaac_py not in py_parts:
            py_parts.append(isaac_py)
    os.environ["PYTHONPATH"] = os.pathsep.join(py_parts)

    sanitize_sys_path(behavior1k_root=b1k_root, isaac_path=isaac)

    if not headless:
        os.environ["DISPLAY"] = display or os.environ.get("DISPLAY") or ":1"

    return dict(os.environ)

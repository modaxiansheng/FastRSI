"""Runtime target registry schemas."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TargetPerceptionRefs(BaseModel):
    enabled: bool = False
    strict_preflight: Literal[True] = True
    sensor_config_ref: Path | None = None
    perception_config_ref: Path | None = None
    artifact_dir: Path | None = None
    config_version: str | None = None


class TargetRuntimeSpec(BaseModel):
    target_runtime: str
    target_endpoint: str | None = None
    target_adapter: str
    runtime_contract_ref: Path


class TargetObservationContract(BaseModel):
    observation_type: Literal[
        "empty",
        "structured",
        "visual",
        "proprioceptive",
        "multimodal",
        "environment_only",
    ] = "multimodal"
    empty_observation_allowed: bool = False
    empty_observation_semantics: str | None = None


class TargetBenchmarkExecutionMode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["policy_loop", "target_native"]
    interface: Literal["rollout_episode_v1", "target_benchmark_job_v1"]
    reset_owner: Literal["session_runner", "skillruntime"]


class TargetBenchmarkCapability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark_id: str
    suites: list[str] = Field(default_factory=list)
    execution_modes: list[TargetBenchmarkExecutionMode]


class TargetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    target_class: Literal["local", "remote"]
    target_kind: Literal["game", "debug", "simulation", "real_robot"]
    embodiment: str | None = None
    enabled: bool = True
    workspace: str
    supported_skillruntimes: list[str] = Field(default_factory=list)
    benchmark_capabilities: list[TargetBenchmarkCapability] = Field(default_factory=list)
    runtime: TargetRuntimeSpec
    observation: TargetObservationContract = Field(default_factory=TargetObservationContract)
    perception: TargetPerceptionRefs = Field(default_factory=TargetPerceptionRefs)
    config: dict[str, Any] = Field(default_factory=dict)


class TargetsDocument(BaseModel):
    version: Literal["runtime_target_registry_v1"] = "runtime_target_registry_v1"
    targets: list[TargetSpec] = Field(default_factory=list)

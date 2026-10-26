"""Typed target-native benchmark job protocol."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class BenchmarkJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["benchmark_job_request_v1"] = "benchmark_job_request_v1"
    benchmark_id: str
    suite_id: str
    run_id: str
    policy_endpoint: str
    episodes: list[dict[str, Any]]
    max_steps: int = Field(gt=0)
    policy_timeout_s: float = Field(gt=0)
    control_mode: str
    evidence_mode: Literal["none", "failed", "all"] = "none"
    verification_profile: Literal["strict", "audit", "recovery"] = "strict"
    verification_endpoint: str | None = None
    verification_token: str | None = None
    verification_timeout_s: float = Field(default=180.0, gt=0)
    max_replans_per_episode: int = Field(default=2, ge=0)
    max_verifier_calls_per_run: int = Field(default=50, ge=0)
    retry_instruction_mode: Literal["original", "verifier_rewrite"] = "original"
    options: dict[str, Any] = Field(default_factory=dict)


class BenchmarkJobRef(BaseModel):
    schema_version: Literal["benchmark_job_ref_v1"] = "benchmark_job_ref_v1"
    job_id: str


class BenchmarkJobStatus(BaseModel):
    schema_version: Literal["benchmark_job_status_v1"] = "benchmark_job_status_v1"
    job_id: str
    state: Literal["pending", "running", "succeeded", "failed", "cancelling", "cancelled"]
    completed_episodes: int = 0
    total_episodes: int = 0
    heartbeat_ns: int
    error_code: str | None = None
    error_message: str | None = None


class BenchmarkExecutionResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["benchmark_execution_result_v1"] = "benchmark_execution_result_v1"
    status: Literal["succeeded", "failed", "timed_out", "cancelled"]
    run_id: str
    suite: str
    execution_mode: Literal["target_native"] = "target_native"
    verification_profile: Literal["strict", "audit", "recovery"] = "strict"
    control_mode: str | None = None
    max_steps: int | None = None
    successes: int = 0
    first_attempt_successes: int = 0
    final_successes: int = 0
    total_episodes: int = 0
    completed_episodes: int = 0
    valid_episodes: int = 0
    total_attempts: int = 0
    success_rate: float = 0.0
    final_success_rate: float = 0.0
    first_attempt_score: float = 0.0
    assisted_final_score: float = 0.0
    episodes_replanned: int = 0
    recovered_after_replan: int = 0
    verifier_calls: int = 0
    configured_max_verifier_calls_per_run: int = 0
    effective_max_verifier_calls_per_run: int = 0
    remaining_verifier_calls: int = 0
    verifier_skipped: int = 0
    num_steps: int = 0
    mean_policy_latency_ms: float | None = None
    elapsed_s: float | None = None
    episodes: list[dict[str, Any]] = Field(default_factory=list)
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    verifier_attempts: list[dict[str, Any]] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None

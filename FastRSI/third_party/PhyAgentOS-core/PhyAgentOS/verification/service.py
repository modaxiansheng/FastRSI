"""Agent-owned HTTP service for scoped benchmark episode verification."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib import error as url_error
from urllib import request as url_request

import numpy as np

from PhyAgentOS.runtime.artifacts.episode_writer import _encode_rgb_png
from PhyAgentOS.verification.engine import VerificationEngine

VERIFICATION_CLIENT_TIMEOUT_GRACE_S = 15.0


EPISODE_PROMPT = """You are a benchmark episode verifier. Return one JSON object with keys verdict
(success|replan|failure), evidence, failure_reason, replan_task_description, and lesson. evidence must be an array of short strings.
Compare the initial and final observations. Use success only when the original task is already complete.
When the task is not complete, prefer replan whenever the scene still contains the required target objects,
the robot/environment can plausibly continue from the final observation, and another attempt with the original
task instruction could still complete the task. This includes wrong object picks, missed grasps, incomplete
placements, unopened drawers, or reaching max steps while the goal remains physically possible. For every
replan verdict, replan_task_description must be a non-empty executable instruction that describes how the next
attempt should recover while preserving the original task goal. Use failure only when continuing from the final
observation is physically impossible or clearly unrecoverable, such as the required object being absent,
inaccessible, irreversibly displaced out of the workspace, or the scene state preventing the original task from
being completed."""

SESSION_PROMPT = """Verify the Agent session and return one JSON object with keys verdict
(success|replan|failure), evidence, failure_reason, replan_task_description, and lesson. evidence must be a
non-empty array of non-empty strings. Every replan verdict must include a non-empty executable
replan_task_description. Every failure verdict must include a non-empty failure_reason. Choose replan only when
another executable session can correct the task; choose failure when the task cannot be corrected by replanning."""


class VerificationServiceError(RuntimeError):
    """Raised when the Agent-owned verification service cannot return a verdict."""


class VerificationServiceProcess:
    def __init__(self, *, engine: VerificationEngine, host: str, port: int, episode_token: str, provider_spec: dict | None = None) -> None:
        self.engine = engine
        self.host = host
        self.port = int(port)
        self.episode_token = episode_token
        self.session_token = hashlib.sha256((episode_token + ":session").encode()).hexdigest()
        self.provider_spec = provider_spec
        self._process: subprocess.Popen | None = None
        self._lifecycle_lock = threading.Lock()
        self._closed = False

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            if self._process is not None and self._process.poll() is None:
                return
            if self.provider_spec is None:
                raise RuntimeError("Verification Service requires a serializable provider specification")
            settings = {
                "provider": self.provider_spec, "host": self.host, "port": self.port,
                "episode_token": self.episode_token, "session_token": self.session_token,
                "timeout_s": self.engine.timeout_s,
            }
            env = dict(os.environ)
            env["PAOS_VERIFICATION_SERVICE_CONFIG"] = json.dumps(settings)
            self._process = subprocess.Popen(
                [sys.executable, "-m", "PhyAgentOS.verification.service"],
                env=env,
                stdin=subprocess.DEVNULL,
            )
        deadline = time.monotonic() + 5.0
        from urllib import request
        opener = request.build_opener(request.ProxyHandler({}))
        probe_host = "127.0.0.1" if self.host == "0.0.0.0" else self.host
        while time.monotonic() < deadline:
            with self._lifecycle_lock:
                if self._closed:
                    return
                process = self._process
            if process is None or process.poll() is not None:
                raise RuntimeError("Verification Service exited before readiness")
            try:
                with opener.open(f"http://{probe_host}:{self.port}/healthz", timeout=0.2) as response:  # noqa: S310
                    if response.status == 200:
                        return
            except OSError:
                time.sleep(0.05)
        self.stop()
        raise TimeoutError("Verification Service readiness timed out")

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._closed = True
            process = self._process
            self._process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)

    def verify_session(self, content: list[dict[str, Any]]) -> dict[str, Any]:
        """Submit Agent session evidence through the child verification service."""
        service_host = "127.0.0.1" if self.host == "0.0.0.0" else self.host
        endpoint = f"http://{service_host}:{self.port}/v1/verify-session"
        payload = json.dumps(
            {"version": "agent_session_verification_v2", "content": content},
            ensure_ascii=False,
        ).encode("utf-8")
        req = url_request.Request(
            endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "X-PAOS-Admin-Token": self.session_token,
            },
            method="POST",
        )
        opener = url_request.build_opener(url_request.ProxyHandler({}))
        client_timeout_s = max(
            1.0,
            float(self.engine.timeout_s) + VERIFICATION_CLIENT_TIMEOUT_GRACE_S,
        )
        try:
            with opener.open(req, timeout=client_timeout_s) as response:  # noqa: S310 - local Agent service
                body = response.read().decode("utf-8")
        except url_error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace").strip()
            except Exception:  # noqa: BLE001
                detail = ""
            message = detail or str(exc.reason) or type(exc).__name__
            raise VerificationServiceError(
                f"session verification service returned HTTP {exc.code}: {message[:500]}"
            ) from exc
        except (OSError, TimeoutError, url_error.URLError) as exc:
            raise VerificationServiceError(
                f"session verification service request failed: {str(exc) or type(exc).__name__}"
            ) from exc
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise VerificationServiceError("session verification service returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise VerificationServiceError("session verification service response must be a JSON object")
        return data


def serve_verification_service(engine: VerificationEngine, host: str, port: int, episode_token: str, session_token: str) -> None:
    server = ThreadingHTTPServer((host, port), _handler(engine, episode_token, session_token))
    server.serve_forever(poll_interval=0.2)


def _handler(engine: VerificationEngine, episode_token: str, session_token: str):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self._send({"ok": self.path == "/healthz"}, 200 if self.path == "/healthz" else 404)

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/v1/verify-episode":
                self._verify_episode()
                return
            if self.path == "/v1/verify-session":
                if self.headers.get("X-PAOS-Admin-Token") != session_token:
                    self._send({"error": "session verification is restricted to the Agent wrapper"}, 403)
                    return
                self._verify_session()
                return
            self._send({"error": "not found"}, 404)

        def _verify_episode(self) -> None:
            try:
                bundle = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                if bundle.get("version") != "benchmark_episode_verification_v1":
                    raise ValueError("unsupported episode verification bundle")
                token = str(self.headers.get("Authorization") or "").removeprefix("Bearer ")
                if not _valid_episode_token(token, episode_token, bundle):
                    self._send({"error": "invalid or expired episode verification token"}, 403)
                    return
            except Exception as exc:  # noqa: BLE001
                self._send({"error": str(exc) or type(exc).__name__}, 400)
                return
            try:
                data = asyncio.run(engine.complete(system_prompt=EPISODE_PROMPT, content=_episode_content(bundle)))
                self._send(_normalize(data), 200)
            except TimeoutError as exc:
                self._send(_error_payload(exc), 504)
            except Exception as exc:  # noqa: BLE001
                self._send(_error_payload(exc), 500)

        def _verify_session(self) -> None:
            try:
                bundle = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                if bundle.get("version") != "agent_session_verification_v2" or not isinstance(bundle.get("content"), list):
                    raise ValueError("unsupported session verification bundle")
            except Exception as exc:  # noqa: BLE001
                self._send({"error": str(exc) or type(exc).__name__}, 400)
                return
            try:
                data = asyncio.run(engine.complete(system_prompt=SESSION_PROMPT, content=bundle["content"]))
                self._send(_normalize(data), 200)
            except TimeoutError as exc:
                self._send(_error_payload(exc), 504)
            except Exception as exc:  # noqa: BLE001
                self._send(_error_payload(exc), 500)

        def _send(self, payload: dict, status: int) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, socket.timeout):
                return

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def _episode_content(bundle: dict) -> list[dict]:
    context = {key: value for key, value in bundle.items() if key not in {"initial_observation", "final_observation"}}
    content: list[dict] = [{"type": "text", "text": json.dumps(context, ensure_ascii=False)}]
    for phase in ("initial_observation", "final_observation"):
        for name, image in _rgb_arrays(bundle.get(phase), phase):
            png = _encode_rgb_png(image)
            content.extend([{"type": "text", "text": name}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}])
    return content


def _rgb_arrays(value: Any, prefix: str) -> list[tuple[str, np.ndarray]]:
    found: list[tuple[str, np.ndarray]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(_rgb_arrays(item, prefix + "." + str(key)))
        return found
    try:
        array = np.asarray(value, dtype=np.uint8)
    except Exception:
        return found
    if array.ndim == 3 and array.shape[-1] == 3:
        found.append((prefix, np.ascontiguousarray(array)))
    return found


def _normalize(data: dict) -> dict:
    verdict = str(data.get("verdict") or "")
    if verdict not in {"success", "replan", "failure"}:
        return _invalid_response("unsupported verifier verdict")

    evidence = data.get("evidence")
    if (
        not isinstance(evidence, list)
        or not evidence
        or any(not isinstance(item, str) or not item.strip() for item in evidence)
    ):
        return _invalid_response("evidence must be a non-empty array of non-empty strings")

    rewrite = str(data.get("replan_task_description") or "").strip()
    if verdict == "replan" and not rewrite:
        return _invalid_response("replan requires non-empty replan_task_description")

    reason = str(data.get("failure_reason") or "").strip()
    if verdict == "failure" and not reason:
        return _invalid_response("failure requires non-empty failure_reason")

    lesson = data.get("lesson")
    if not isinstance(lesson, str) or not lesson.strip():
        return _invalid_response("lesson must be a non-empty string")

    return {
        "verdict": verdict,
        "evidence": [item.strip() for item in evidence],
        "failure_reason": reason or None,
        "replan_task_description": rewrite or None,
        "lesson": lesson.strip(),
        "verifier_status": "completed",
    }


def _invalid_response(reason: str) -> dict:
    return {
        "verdict": "failure",
        "evidence": ["invalid verifier response: %s" % reason],
        "failure_reason": reason,
        "replan_task_description": None,
        "lesson": "Verifier returned an invalid response.",
        "verifier_status": "invalid_response",
    }


def _error_payload(exc: Exception) -> dict:
    reason = str(exc) or type(exc).__name__
    return {
        "verdict": "failure",
        "evidence": ["verifier infrastructure error: %s" % reason],
        "failure_reason": reason,
        "replan_task_description": None,
        "lesson": "Verification could not be completed.",
        "verifier_status": "error",
    }


def _valid_episode_token(token: str, secret: str, bundle: dict) -> bool:
    try:
        encoded, signature = token.split(".", 1)
        expected = hmac.new(secret.encode(), encoded.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return False
        padded = encoded + "=" * (-len(encoded) % 4)
        claims = json.loads(base64.urlsafe_b64decode(padded))
        return (
            int(claims["expires_at"]) >= int(time.time())
            and str(claims["run_id"]) == str(bundle.get("run_id"))
            and str(claims["session_id"]) == str(bundle.get("session_id"))
        )
    except Exception:
        return False


def _provider(spec: dict, timeout_s: float | None = None):
    from PhyAgentOS.providers.base import GenerationSettings

    name = str(spec["provider_name"])
    model = str(spec["model"])
    if name == "custom":
        from PhyAgentOS.providers.custom_provider import CustomProvider
        provider = CustomProvider(
            api_key=spec.get("api_key") or "no-key",
            api_base=spec.get("api_base") or "http://localhost:8000/v1",
            default_model=model,
            timeout_s=float(timeout_s or spec.get("timeout_s") or os.environ.get("PAOS_VERIFICATION_TIMEOUT_S", 180.0)),
        )
    elif name == "azure_openai":
        from PhyAgentOS.providers.azure_openai_provider import AzureOpenAIProvider
        provider = AzureOpenAIProvider(api_key=spec.get("api_key"), api_base=spec.get("api_base"), default_model=model)
    elif name == "openai_codex":
        from PhyAgentOS.providers.openai_codex_provider import OpenAICodexProvider
        provider = OpenAICodexProvider(default_model=model)
    else:
        from PhyAgentOS.providers.litellm_provider import LiteLLMProvider
        provider = LiteLLMProvider(api_key=spec.get("api_key"), api_base=spec.get("api_base"), default_model=model, extra_headers=spec.get("extra_headers"), provider_name=name)
    provider.generation = GenerationSettings(temperature=float(spec.get("temperature", 0.0)), max_tokens=int(spec.get("max_tokens", 1024)), reasoning_effort=spec.get("reasoning_effort"))
    return provider


def main() -> int:
    settings = json.loads(os.environ["PAOS_VERIFICATION_SERVICE_CONFIG"])
    timeout_s = float(settings["timeout_s"])
    provider = _provider(settings["provider"], timeout_s=timeout_s)
    engine = VerificationEngine(provider=provider, model=settings["provider"]["model"], timeout_s=timeout_s)
    serve_verification_service(engine, settings["host"], int(settings["port"]), settings["episode_token"], settings["session_token"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

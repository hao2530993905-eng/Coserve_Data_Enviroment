"""Call the existing CoServe Agent from Python with a configurable model API.

CoServe owns prompt assembly, tools, the agent loop and Session persistence.
This adapter uses its Python SDK; it does not implement a second agent loop.
"""

from __future__ import annotations

import importlib
import os
import shlex
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from collector import Collector, utc_now, write_json

if TYPE_CHECKING:
    from deepseek_harness import DeepSeekHarness


@dataclass(frozen=True)
class ModelAPI:
    """An OpenAI Chat Completions endpoint used by CoServe's existing adapter.

    Supply the API key directly or through ``api_key_env``. For a server that
    ignores authentication, explicitly pass a placeholder such as ``EMPTY``.
    The key is passed to the child environment and excluded from saved config.
    """

    base_url: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    api_key_env: str = "COSERVE_MODEL_API_KEY"
    context_window: int = 32768
    max_tokens: int = 4096

    def __post_init__(self) -> None:
        url = urlsplit(self.base_url)
        if url.scheme not in ("http", "https") or not url.netloc or url.username or url.password:
            raise ValueError("base_url must be an HTTP(S) model endpoint without embedded credentials")
        if url.query or url.fragment:
            raise ValueError("base_url must not contain a query or fragment")
        if not self.model.strip() or not self.api_key_env.strip():
            raise ValueError("model and api_key_env must not be blank")
        if self.context_window <= 0 or self.max_tokens <= 0 or self.max_tokens > self.context_window:
            raise ValueError("Require 0 < max_tokens <= context_window")

    def key(self) -> str:
        """Resolve the credential without serializing the process environment."""
        value = self.api_key if self.api_key is not None else os.environ.get(self.api_key_env)
        if not value:
            raise ValueError(f"Provide api_key or set {self.api_key_env}; use EMPTY for a keyless local server")
        return value

    def public_config(self) -> dict[str, object]:
        """Return model settings safe to include in the run's request record."""
        return {
            "base_url": self.base_url, "model": self.model,
            "context_window": self.context_window, "max_tokens": self.max_tokens,
            "api_key_env": self.api_key_env,
        }


@dataclass(frozen=True)
class Rollout:
    """An execution outcome; completed means the Agent turn ended normally.

    Engineering success is not assessed here. Inspect the saved events and
    artifacts before treating the execution as a useful training example.
    """

    run_id: str
    session_id: str
    status: str
    finish_reason: str | None
    final_response: str
    directory: Path


class CoServeRunError(RuntimeError):
    """An execution or collection failure whose partial records remain on disk."""

    def __init__(self, directory: Path, cause: Exception) -> None:
        self.directory = directory
        super().__init__(f"CoServe run failed; partial records: {directory}: {cause}")


def _sdk_class(repo: Path) -> type[DeepSeekHarness]:
    sdk_source = repo / "python" / "sdk" / "src"
    if not (sdk_source / "deepseek_harness" / "__init__.py").is_file():
        raise FileNotFoundError(f"CoServe Python SDK not found in {sdk_source}")
    loaded = sys.modules.get("deepseek_harness")
    if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(sdk_source):
        raise RuntimeError("This Python process already loaded an SDK from another checkout; use a new process")
    if str(sdk_source) not in sys.path:
        sys.path.insert(0, str(sdk_source))
    return importlib.import_module("deepseek_harness").DeepSeekHarness


def _launcher(repo: Path, directory: Path) -> tuple[Path, dict[str, str]]:
    """Adapt the source dsh executable to the SDK's public dsh_bin option."""
    node = shutil.which("node")
    entry = repo / "apps" / "cli" / "src" / "bin.ts"
    loader = repo / "node_modules" / "tsx" / "dist" / "esm" / "index.mjs"
    for required in (entry, loader, repo / "apps" / "cli" / "src" / "sdk-source.cordis.patch.yml"):
        if not required.is_file():
            raise FileNotFoundError(f"Missing {required}; run pnpm install in the CoServe checkout")
    if node is None:
        raise FileNotFoundError("Node.js >=24.9 is required on PATH for the CoServe source runtime")
    arguments = [node, "--import", loader.as_uri(), str(entry)]
    if os.name == "nt":
        launcher = directory / "dsh.cmd"
        launcher.write_bytes(
            b'@echo off\r\nsetlocal DisableDelayedExpansion\r\n'
            b'"%COSERVE_ROLLOUT_NODE%" --import "%COSERVE_ROLLOUT_LOADER%" "%COSERVE_ROLLOUT_ENTRY%" %*\r\n'
        )
        environment = {"COSERVE_ROLLOUT_NODE": node, "COSERVE_ROLLOUT_LOADER": loader.as_uri(),
                       "COSERVE_ROLLOUT_ENTRY": str(entry)}
    else:
        launcher = directory / "dsh"
        launcher.write_text("#!/bin/sh\nexec " + shlex.join(arguments) + ' "$@"\n', encoding="utf-8")
        launcher.chmod(0o700)
        environment = {}
    return launcher, environment


class CoServeAgent:
    """Run manually supplied tasks through a local CoServe checkout.

    Every run creates a fresh SDK runtime, Session and Harness home. The
    workspace is used in place, so its mutations persist between runs.
    External CAStudio services must already be prepared by the caller.
    One Python process can bind to one CoServe SDK checkout.
    """

    def __init__(
        self, *, repo: str | Path, workspace: str | Path, model: ModelAPI,
        output_dir: str | Path = "runs", patches: tuple[str | Path, ...] = (),
        permission_mode: str = "workspace-write", initialize_timeout: float = 60,
    ) -> None:
        self.repo = Path(repo).expanduser().resolve()
        self.workspace = Path(workspace).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.model = model
        self.patches = tuple(Path(patch).expanduser().resolve() for patch in patches)
        self.permission_mode = permission_mode
        self.initialize_timeout = initialize_timeout
        if not self.workspace.is_dir():
            raise NotADirectoryError(self.workspace)
        for patch in self.patches:
            if not patch.is_file():
                raise FileNotFoundError(patch)

    def run(self, task: str, *, artifacts: tuple[str, ...] = ()) -> Rollout:
        """Execute one task, save its trace and copy explicitly selected artifacts.

        Agent-reported non-completion returns a failed Rollout. SDK or output
        collection exceptions raise CoServeRunError with the output directory.
        Interrupts are recorded and re-raised after SDK shutdown.
        """
        if not task.strip():
            raise ValueError("task must not be blank")
        key = self.model.key()
        run_id = "run-" + uuid.uuid4().hex
        directory = self.output_dir / run_id
        session_id = run_id
        collector = Collector(directory, {
            "run_id": run_id, "session_id": session_id, "task": task,
            "workspace": str(self.workspace), "coserve_repo": str(self.repo),
            "model": self.model.public_config(), "created_at": utc_now(),
            "patches": [str(path) for path in self.patches],
            "permission_mode": self.permission_mode,
        })
        outcome: dict[str, object] = {"run_id": run_id, "session_id": session_id}
        try:
            sdk = _sdk_class(self.repo)
            launcher, launch_env = _launcher(self.repo, directory)
            patch = directory / "runtime.patch.json"
            write_json(patch, [
                {"id": "llm-pi-ai", "config": {"providers": {"coserve-rollout": {
                    "api": "openai-completions", "baseURL": self.model.base_url,
                    "apiKeyEnv": "COSERVE_ROLLOUT_API_KEY",
                    "models": [{"id": self.model.model, "name": self.model.model,
                                "contextWindow": self.model.context_window,
                                "maxTokens": self.model.max_tokens, "reasoningEfforts": False}],
                }}}},
                {"id": "agent-default-model", "config": {
                    "provider": "coserve-rollout", "model": self.model.model,
                }},
                {"id": "session-persistence-jsonl", "config": {
                    "root": str(directory / "sessions"), "compression": "none",
                }},
            ])
            harness = sdk(
                provider="coserve-rollout", model=self.model.model,
                max_tokens=self.model.max_tokens, cwd=str(self.workspace),
                runtime_cwd=str(self.repo), dsh_bin=str(launcher),
                profile="sdk", coserve_home=str(directory / "runtime-home"),
                patches=(str(self.repo / "apps/cli/src/sdk-source.cordis.patch.yml"),
                         *(str(path) for path in self.patches), str(patch)),
                env={**launch_env, "COSERVE_ROLLOUT_API_KEY": key,
                     "TSX_TSCONFIG_PATH": str(self.repo / "apps/cli/tsconfig.json"),
                     "DSH_PERMISSION_MODE": self.permission_mode},
                initialize_timeout_seconds=self.initialize_timeout,
                shutdown_timeout_seconds=10,
            )
            try:
                result = harness.run(task, session_id=session_id, on_notification=collector.record)
            finally:
                harness.close()
            status = "completed" if result.finish_reason == "completed" else "failed"
            outcome.update(status=status, finish_reason=result.finish_reason,
                           final_response=result.final_response)
            (directory / "final_response.txt").write_text(result.final_response, encoding="utf-8")
            outcome["artifacts"] = collector.collect_artifacts(self.workspace, artifacts)
            rollout = Rollout(run_id, session_id, status, result.finish_reason, result.final_response, directory)
        except BaseException as error:
            outcome.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                           error={"type": type(error).__name__, "message": str(error)})
            if isinstance(error, Exception):
                raise CoServeRunError(directory, error) from error
            raise
        finally:
            collector.finish(outcome)
        return rollout

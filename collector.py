"""Save one CoServe execution without converting its events into training samples."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepseek_harness import Notification


def utc_now() -> str:
    """Return an ISO timestamp with an explicit UTC offset."""
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    """Replace a JSON document after its complete UTF-8 contents are written."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class Collector:
    """Own the output directory and open event stream for a single run.

    Each notification is flushed before returning to the SDK. Session files
    retain CoServe's own event representation, including events before the
    SDK run interval. Neither source is a provider-wire or token-ID capture.
    """

    def __init__(self, directory: Path, request: dict[str, object]) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "request.json", request)
        write_json(directory / "result.json", {"status": "running", "started_at": utc_now()})
        self.stream = (directory / "notifications.jsonl").open("w", encoding="utf-8", newline="\n")

    def record(self, notification: Notification) -> None:
        """Append one untruncated SDK notification, retaining its Session id."""
        self.stream.write(json.dumps({
            "method": notification.method,
            "payload": notification.payload,
        }, ensure_ascii=False) + "\n")
        self.stream.flush()

    def collect_artifacts(self, workspace: Path, paths: tuple[str, ...]) -> list[dict[str, object]]:
        """Copy explicitly selected workspace files or directories and record hashes.

        Missing paths and symlinks raise. Partial copies remain in the failed
        run directory. Paths must be relative to the original workspace.
        """
        manifest: list[dict[str, object]] = []
        for name in paths:
            relative = Path(name)
            if relative.is_absolute() or not relative.parts or ".." in relative.parts:
                raise ValueError(f"Artifact path must be relative to the workspace: {name}")
            source = workspace / relative
            if source.is_symlink() or not source.resolve().is_relative_to(workspace):
                raise ValueError(f"Artifact path leaves the workspace or is a symlink: {name}")
            if not source.exists():
                raise FileNotFoundError(f"Requested artifact does not exist: {source}")
            if self.directory.is_relative_to(source.resolve()):
                raise ValueError(f"Artifact path contains the run directory: {name}")
            files = sorted(source.rglob("*")) if source.is_dir() else [source]
            for file in files:
                if file.is_symlink():
                    raise ValueError(f"Artifact contains a symlink: {file}")
                if file.is_dir():
                    continue
                destination = self.directory / "artifacts" / file.relative_to(workspace)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(file, destination)
                with destination.open("rb") as copied:
                    digest = hashlib.file_digest(copied, "sha256").hexdigest()
                manifest.append({
                    "path": destination.relative_to(self.directory).as_posix(),
                    "bytes": destination.stat().st_size,
                    "sha256": digest,
                })
        return manifest

    def finish(self, result: dict[str, object]) -> None:
        """Close the stream and save the outcome, including partial-run failures."""
        self.stream.close()
        write_json(self.directory / "result.json", {**result, "finished_at": utc_now()})

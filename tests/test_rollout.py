"""Verify record retention, credential handling and SDK ownership without a model."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collector import Collector
from coserve_agent import CoServeAgent, CoServeRunError, ModelAPI


class CollectorTests(unittest.TestCase):
    def test_long_child_event_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            collector = Collector(output, {"task": "中文任务"})
            text = "工具结果" * 10000
            collector.record(SimpleNamespace(method="session.event", payload={
                "sessionId": "child", "event": {"type": "tool/result", "data": {"text": text}},
            }))
            collector.finish({"status": "failed"})
            saved = json.loads((output / "notifications.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(saved["payload"]["event"]["data"]["text"], text)
            self.assertEqual(saved["payload"]["sessionId"], "child")

    def test_artifact_paths_cannot_escape_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            collector = Collector(workspace / "run", {})
            try:
                with self.assertRaises(ValueError):
                    collector.collect_artifacts(workspace, ("../outside.txt",))
            finally:
                collector.finish({"status": "failed"})


class AgentTests(unittest.TestCase):
    def exercise(self, *, finish_reason: str = "completed", fail: bool = False,
                 missing_artifact: bool = False) -> tuple[dict, Path, MagicMock]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        workspace = root / "workspace"
        workspace.mkdir()
        runtime = MagicMock()

        def run(task, *, session_id, on_notification):
            on_notification(SimpleNamespace(method="session.event", payload={
                "sessionId": session_id, "event": {"type": "user/message", "data": {"text": task}},
            }))
            if fail:
                raise ConnectionError("model endpoint unavailable")
            if not missing_artifact:
                (workspace / "answer.txt").write_text("工程结果", encoding="utf-8")
            return SimpleNamespace(finish_reason=finish_reason, final_response="done")

        runtime.run.side_effect = run
        sdk = MagicMock(return_value=runtime)
        agent = CoServeAgent(repo=root, workspace=workspace, output_dir=root / "runs",
                             model=ModelAPI("http://localhost:8000/v1", "trained", api_key="private-test-key"))
        with patch("coserve_agent._sdk_class", return_value=sdk), patch(
            "coserve_agent._launcher", return_value=(root / "dsh", {}),
        ):
            if fail or missing_artifact:
                with self.assertRaises(CoServeRunError) as caught:
                    agent.run("test", artifacts=("answer.txt",))
                directory = caught.exception.directory
            else:
                directory = agent.run("test", artifacts=("answer.txt",)).directory
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        return result, directory, sdk

    def test_endpoint_is_configured_in_coserve_and_key_stays_in_environment(self) -> None:
        result, directory, sdk = self.exercise()
        self.assertEqual(result["status"], "completed")
        options = sdk.call_args.kwargs
        self.assertEqual(options["profile"], "sdk")
        self.assertEqual(options["model"], "trained")
        self.assertEqual(options["env"]["COSERVE_ROLLOUT_API_KEY"], "private-test-key")
        for file in directory.rglob("*"):
            if file.is_file():
                self.assertNotIn(b"private-test-key", file.read_bytes())
        self.assertEqual((directory / "artifacts/answer.txt").read_text(encoding="utf-8"), "工程结果")
        self.assertEqual(len(result["artifacts"][0]["sha256"]), 64)
        sdk.return_value.close.assert_called_once()

    def test_failed_model_keeps_events_and_closes_runtime(self) -> None:
        result, directory, sdk = self.exercise(fail=True)
        self.assertEqual(result["error"]["type"], "ConnectionError")
        self.assertTrue((directory / "notifications.jsonl").read_text(encoding="utf-8"))
        sdk.return_value.close.assert_called_once()

    def test_noncompleted_turn_is_not_reported_as_completed(self) -> None:
        result, _, _ = self.exercise(finish_reason="max-tokens")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["finish_reason"], "max-tokens")

    def test_missing_artifact_keeps_agent_outcome_and_reports_failure(self) -> None:
        result, _, _ = self.exercise(missing_artifact=True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["finish_reason"], "completed")
        self.assertEqual(result["error"]["type"], "FileNotFoundError")

    def test_model_url_rejects_embedded_credentials(self) -> None:
        with self.assertRaises(ValueError):
            ModelAPI("https://user:secret@example.com/v1", "trained")


if __name__ == "__main__":
    unittest.main()

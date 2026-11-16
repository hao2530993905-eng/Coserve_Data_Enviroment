"""Drive the real CoServe SDK and tools through a local OpenAI-compatible test endpoint."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from coserve_agent import CoServeAgent, ModelAPI


def main() -> None:
    """Require a real read/write cycle and preserve its evidence under .smoke/."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=PROJECT.parent)
    args = parser.parse_args()
    work = PROJECT / ".smoke" / uuid.uuid4().hex
    workspace = work / "workspace"
    workspace.mkdir(parents=True)
    challenge = "host-only-" + uuid.uuid4().hex
    (workspace / "input.txt").write_text(challenge, encoding="utf-8")
    requests: list[dict] = []
    final_text = "SMOKE_OK 中文轨迹 " + "x" * 12000

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append({"path": self.path, "body": body})
            tools = {tool["function"]["name"] for tool in body.get("tools", [])}
            observed = [message for message in body["messages"] if message["role"] == "tool"]
            call = None
            if "read" in tools and not observed:
                call = {"id": "smoke-read", "name": "read", "arguments": {"file_path": "input.txt"}}
            elif "write" in tools and observed[-1]["tool_call_id"] == "smoke-read":
                if challenge not in json.dumps(observed[-1]):
                    self.send_error(500, "CoServe did not return the file observation")
                    return
                call = {"id": "smoke-write", "name": "write",
                        "arguments": {"file_path": "answer.txt", "content": challenge}}
            delta = {"content": final_text if tools else "smoke title"}
            reason = "stop"
            if call:
                delta = {"tool_calls": [{"index": 0, "id": call["id"], "type": "function",
                    "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}}]}
                reason = "tool_calls"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for content, finish in ((delta, None), ({}, reason)):
                event = {"id": "smoke", "object": "chat.completion.chunk", "created": 0,
                         "model": body["model"], "choices": [{"index": 0, "delta": content,
                         "finish_reason": finish}]}
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, _format: str, *_args: object) -> None:
            """Keep the smoke output limited to its validation result."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        agent = CoServeAgent(
            repo=args.repo, workspace=workspace, output_dir=work / "runs",
            model=ModelAPI(f"http://127.0.0.1:{server.server_port}/v1", "coserve-test-model",
                           api_key="test-key", max_tokens=8192),
        )
        rollout = agent.run("Read input.txt with read, write its contents to answer.txt with write, then finish.",
                            artifacts=("answer.txt",))
        assert rollout.status == "completed", rollout
        assert rollout.final_response == final_text
        assert (workspace / "answer.txt").read_text(encoding="utf-8") == challenge
        assert (rollout.directory / "artifacts/answer.txt").read_text(encoding="utf-8") == challenge
        sessions = list((rollout.directory / "sessions").rglob("*.jsonl"))
        assert sessions, "CoServe did not persist its raw Session files"
        raw = "\n".join(path.read_text(encoding="utf-8") for path in sessions)
        for event in ("system/message", "request/header", "tool/call", "tool/result", "assistant/message"):
            assert event in raw, f"Missing {event} in raw Session files"
        assert final_text in (rollout.directory / "notifications.jsonl").read_text(encoding="utf-8")
        assert all(request["path"] == "/v1/chat/completions" for request in requests), requests
        assert len([request for request in requests if request["body"].get("tools")]) == 3
        print(f"PASS: Python -> CoServe SDK -> model API -> read/write -> next model request")
        print(f"Untruncated final response: {len(final_text)} characters")
        print(f"Records: {rollout.directory}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()

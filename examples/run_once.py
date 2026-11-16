"""Configure a model API and run one task inside the real CoServe framework."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from coserve_agent import CoServeAgent, ModelAPI


def main() -> int:
    """Run a manually supplied task with the configured model endpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("--repo", type=Path, default=PROJECT.parent)
    parser.add_argument("--workspace", type=Path, default=PROJECT / "examples" / "workspace")
    parser.add_argument("--base-url", default=os.environ.get("COSERVE_MODEL_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("COSERVE_MODEL", "coserve-agent"))
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "runs")
    parser.add_argument("--artifact", action="append", default=[])
    args = parser.parse_args()
    agent = CoServeAgent(
        repo=args.repo, workspace=args.workspace, output_dir=args.output_dir,
        model=ModelAPI(base_url=args.base_url, model=args.model),
    )
    rollout = agent.run(args.task, artifacts=tuple(args.artifact))
    print(rollout.final_response)
    print(f"status={rollout.status}\nrecords={rollout.directory}")
    return 0 if rollout.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import asyncio

from config import CONFIG

from .pipeline import build_harness


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Harness knowledge tree.")
    parser.add_argument(
        "--provider",
        choices=["heuristic", "openai-compatible"],
        help="Override the configured LLM provider.",
    )
    args = parser.parse_args()
    if args.provider:
        CONFIG.llm.provider = args.provider
    asyncio.run(build_harness(CONFIG))

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .demo import run_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coastal-segformer-demo",
        description="Run a public-safe coastal segmentation inference demo.",
    )
    parser.add_argument("run", nargs="?", default="run", choices=["run"])
    parser.add_argument("--config", default="configs/example_demo.yml")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(Path(args.config))
    result = run_demo(config)
    print(f"output_dir={result.output_dir}")
    for artifact in result.artifacts:
        print(f"artifact={artifact}")
    return 0


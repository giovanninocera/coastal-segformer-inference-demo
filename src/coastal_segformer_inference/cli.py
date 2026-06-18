from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .inference import run_inference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coastal-segformer-inference",
        description="Run a coastal segmentation inference workflow.",
    )
    parser.add_argument("run", nargs="?", default="run", choices=["run"])
    parser.add_argument("--config", default="configs/example_inference.yml")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(Path(args.config))
    result = run_inference(config)
    print(f"output_dir={result.output_dir}")
    for artifact in result.artifacts:
        print(f"artifact={artifact}")
    return 0

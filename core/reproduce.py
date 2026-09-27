"""Re-run or resume an experiment from its recorded resolved configuration."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf, open_dict

from core.evaluate import _checkpoint_path
from core.train import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-only", action="store_true", help="Disable external loggers")
    args = parser.parse_args()

    source = args.experiment_dir.resolve()
    config_path = source / "experiment" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing recorded config: {config_path}")
    cfg = OmegaConf.load(config_path)

    if args.resume:
        output_dir = source
        with open_dict(cfg):
            cfg.trainer.resume_from = str(_checkpoint_path(source, "last"))
    else:
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (args.output_dir or source.parent / f"{source.name}_reproduced_{suffix}").resolve()
        with open_dict(cfg):
            cfg.run_name = output_dir.name
            cfg.trainer.resume_from = None

    if args.local_only:
        with open_dict(cfg):
            cfg.loggers = {}

    run(cfg, output_dir)


if __name__ == "__main__":
    main()

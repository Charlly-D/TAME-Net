"""Train TimesNet multimodal models from scratch on target zones."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.timesnet_zone_adaptation.strategy_runner import add_common_args, run_strategy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train local TimesNet multimodal models from scratch.")
    add_common_args(parser)
    run_strategy(parser.parse_args(), "local_scratch")

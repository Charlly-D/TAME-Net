# TAME-Net

This repository contains the core implementation and principal experimental scripts for **TAME-Net**.

## Contents

- `models/`: TimesNet and multimodal TAME-Net components.
- `timesnet_ablation_suite/`: ablation, sensitivity, and forecasting-horizon experiments.
- `timesnet_zone_adaptation/`: global-to-local regional adaptation experiments.
- `chronos2_finetune/`: Chronos-2 fine-tuning and checkpoint evaluation.
- `main_table_multiseed/`: multi-seed experiment orchestration used for the main comparison.

## Environment

The experiments were developed with Python and PyTorch. Install the recorded dependencies with:

```bash
pip install -r requirements.txt
```

If necessary, install the PyTorch build appropriate for your CUDA version before installing the remaining packages.

## Data configuration

The datasets and trained checkpoints are not included. Set the paths to the hourly load data, ERA5 meteorological grids, and GOES satellite imagery in `config.json`. Regional experiments additionally use the JSON files in `timesnet_zone_adaptation/configs/`.

The scripts expose their experiment-specific options through `--help`. Outputs, checkpoints, and evaluation summaries are written to the directory supplied by the corresponding command-line argument.

## Availability

This is the review-oriented release of the implementation and principal experiment scripts. The complete project codebase and data are subject to the ongoing project's management requirements.


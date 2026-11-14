"""Run stochastic main-table models for multiple random seeds.

The coordinator launches every training job in a separate Python process,
skips valid completed outputs, evaluates each trained model on the test split,
and writes an aggregate summary. Console output is inherited by default and is
only saved when explicit per-job logging is requested.
TAME-Net and its TimesNet backbone are trained together so the two reported
methods use the same seed and Stage-1a backbone training protocol.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
BASELINE_SCRIPT = ROOT / "experiments" / "baseline_suite" / "train_baseline.py"
DEFAULT_SUITE_CONFIG = (
    ROOT / "experiments" / "baseline_suite" / "configs" / "baseline_suite_template.json"
)

DEFAULT_MODELS = [
    "tame_timesnet",
    "dlinear",
    "patchtst",
    "gtrmlp",
    "autoformer_load",
    "autoformer_weather",
    "itransformer",
    "crossfusion",
    "convlstm",
]

BASELINE_SPECS = {
    "dlinear": "DLinearLoadOnly",
    "patchtst": "PatchTSTLoadOnly",
    "gtrmlp": "GTRMLPLoadOnly",
    "autoformer_load": "AutoformerLoadOnly",
    "autoformer_weather": "Autoformer",
    "itransformer": "iTransformer",
    "crossfusion": "CrossFusion",
    "convlstm": "ConvLSTM",
}


@dataclass(frozen=True)
class Job:
    name: str
    seed: int
    command: list[str]
    log_path: Path
    outputs: tuple[Path, ...]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    temporary.replace(path)


def baseline_metrics_path(run_dir: Path, model_name: str) -> Path:
    return run_dir / f"{model_name.lower()}_metrics.json"


def metrics_are_valid(path: Path, key: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = load_json(path)
        metrics = payload.get(key)
        return isinstance(metrics, dict) and all(
            metric in metrics and metrics[metric] is not None
            for metric in ("MAE", "RMSE", "MAPE(%)")
        )
    except (OSError, ValueError, TypeError):
        return False


def job_is_complete(job: Job) -> bool:
    if job.name == "tame_timesnet":
        return metrics_are_valid(job.outputs[0], "final_metrics") and metrics_are_valid(
            job.outputs[1], "metrics"
        )
    return metrics_are_valid(job.outputs[0], "metrics")


def run_tame_worker(args: argparse.Namespace) -> int:
    import torch

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from experiments.timesnet_multimodal.execute_training import (
        LoadPredictorTrainer,
        execute_training,
    )
    from experiments.timesnet_multimodal.run_param_sweep import (
        fix_random_seed,
        load_flat_config,
    )

    config_path = Path(args.config).resolve()
    run_dir = Path(args.worker_run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    base_config, data_paths = load_flat_config(str(config_path))
    training_results_path = run_dir / "training_stage_results.json"
    selected_test_path = run_dir / "selected_test_metrics.json"
    final_config_path = run_dir / "final_config.json"

    # Parameter-sweep runs may store the separately evaluated test result under
    # selected_test_metrics.json. Normalize that record so the coordinator can
    # reuse the run without retraining TAME-Net.
    if (
        not metrics_are_valid(training_results_path, "final_metrics")
        and metrics_are_valid(selected_test_path, "test_metrics")
    ):
        selected_payload = load_json(selected_test_path)
        training_payload = load_json(training_results_path) if training_results_path.is_file() else {}
        training_payload["final_metrics"] = selected_payload["test_metrics"]
        training_payload["test_metrics_source"] = str(selected_test_path)
        atomic_write_json(training_results_path, training_payload)

    existing_tame_run = (
        metrics_are_valid(training_results_path, "final_metrics")
        and final_config_path.is_file()
        and (run_dir / "stage1a_backbone.pth").is_file()
    )
    if existing_tame_run:
        print("[MainTable] Reusing existing TAME-Net test result; training is skipped.")
        config = load_json(final_config_path)
    else:
        config = base_config
        config.pop("stage1_weights_path", None)
        config.pop("stage1a_weights_path", None)
        config.pop("stage1b_weights_path", None)
        config.pop("stage1c_weights_path", None)
        config.update(
            {
                "ignore_stage1_weights": True,
                "pred_len": args.pred_len,
                "aux_delta_scale": args.aux_delta_scale,
                "router_init_bias": args.router_init_bias,
            }
        )
        if args.label_len is not None:
            config["label_len"] = args.label_len
        if args.batch_size is not None:
            config["batch_size"] = args.batch_size
        if args.num_workers is not None:
            config["num_workers"] = args.num_workers

        fix_random_seed(args.worker_seed)
        result = execute_training(data_paths, config, str(run_dir), evaluate_test=True)
        if result.get("status") != "success":
            print(result.get("traceback", result.get("message", "TAME-Net training failed.")))
            return 1

    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    # Evaluate the paired Stage-1a TimesNet backbone on exactly the same split.
    checkpoint_path = run_dir / "stage1a_backbone.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    cache_dir = run_dir / "timesnet_test_cache"
    trainer = LoadPredictorTrainer(config, str(cache_dir), device_ids=[args.device_id])
    trainer.prepare_data(data_paths)
    state = torch.load(checkpoint_path, map_location=trainer.device, weights_only=True)
    backbone_state = {key: value for key, value in state.items() if key.startswith("backbone.")}
    if not backbone_state:
        raise RuntimeError(f"No backbone tensors found in {checkpoint_path}")
    incompatible = trainer.model.load_state_dict(backbone_state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected backbone tensors: {unexpected}")
    trainer.model.fusion_mode = "base_only"
    timesnet_metrics = trainer.test_model(mode="pretrain_backbone")
    atomic_write_json(
        run_dir / "timesnet_backbone_test_metrics.json",
        {
            "model": "TimesNet (paired TAME-Net Stage 1a backbone)",
            "seed": args.worker_seed,
            "split": "test",
            "checkpoint": str(checkpoint_path),
            "metrics": timesnet_metrics,
        },
    )
    return 0


def build_jobs(args: argparse.Namespace) -> list[Job]:
    output_root = Path(args.output_root).resolve()
    config = str(Path(args.config).resolve())
    suite_config = str(Path(args.suite_config).resolve())
    jobs: list[Job] = []

    for seed in args.seeds:
        seed_dir = output_root / f"seed_{seed}"
        tame_dir = seed_dir / "tame_net"
        if "tame_timesnet" in args.models:
            command = [
                sys.executable,
                "-u",
                str(SCRIPT),
                "--worker-tame",
                "--config",
                config,
                "--worker-run-dir",
                str(tame_dir),
                "--worker-seed",
                str(seed),
                "--pred-len",
                str(args.pred_len),
                "--aux-delta-scale",
                str(args.aux_delta_scale),
                "--router-init-bias",
                str(args.router_init_bias),
                "--device-id",
                str(args.device_id),
            ]
            if args.label_len is not None:
                command.extend(["--label-len", str(args.label_len)])
            if args.batch_size is not None:
                command.extend(["--batch-size", str(args.batch_size)])
            if args.num_workers is not None:
                command.extend(["--num-workers", str(args.num_workers)])
            jobs.append(
                Job(
                    name="tame_timesnet",
                    seed=seed,
                    command=command,
                    log_path=output_root / "logs" / f"seed_{seed}_tame_timesnet.log",
                    outputs=(
                        tame_dir / "training_stage_results.json",
                        tame_dir / "timesnet_backbone_test_metrics.json",
                    ),
                )
            )

        for short_name in args.models:
            model_name = BASELINE_SPECS.get(short_name)
            if model_name is None:
                continue
            run_dir = seed_dir / short_name
            command = [
                sys.executable,
                "-u",
                str(BASELINE_SCRIPT),
                "--model",
                model_name,
                "--config",
                config,
                "--suite-config",
                suite_config,
                "--save-dir",
                str(run_dir),
                "--seed",
                str(seed),
                "--pred-len",
                str(args.pred_len),
            ]
            if args.label_len is not None:
                command.extend(["--label-len", str(args.label_len)])
            if args.batch_size is not None:
                command.extend(["--batch-size", str(args.batch_size)])
            if args.num_workers is not None:
                command.extend(["--num-workers", str(args.num_workers)])
            if args.device_ids:
                command.extend(["--device-ids", args.device_ids])

            # Match Cross-Fusion's load backbone initialization to the paired
            # TAME-Net Stage-1a checkpoint. Training remains fully enabled.
            if short_name == "crossfusion" and args.reuse_tame_backbone:
                command.extend(["--backbone-checkpoint", str(tame_dir / "stage1a_backbone.pth")])

            jobs.append(
                Job(
                    name=short_name,
                    seed=seed,
                    command=command,
                    log_path=output_root / "logs" / f"seed_{seed}_{short_name}.log",
                    outputs=(baseline_metrics_path(run_dir, model_name),),
                )
            )
    return jobs


def extract_available_rows(
    output_root: Path,
    models: list[str],
    seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for seed in seeds:
        seed_dir = output_root / f"seed_{seed}"
        if "tame_timesnet" in models:
            tame_path = seed_dir / "tame_net" / "training_stage_results.json"
            if metrics_are_valid(tame_path, "final_metrics"):
                tame_payload = load_json(tame_path)
                rows.append({"model": "TAME-Net", "seed": seed, **tame_payload["final_metrics"]})
            else:
                missing.append({"model": "TAME-Net", "seed": seed, "expected_path": str(tame_path)})

            timesnet_path = seed_dir / "tame_net" / "timesnet_backbone_test_metrics.json"
            if metrics_are_valid(timesnet_path, "metrics"):
                timesnet_payload = load_json(timesnet_path)
                rows.append({"model": "TimesNet", "seed": seed, **timesnet_payload["metrics"]})
            else:
                missing.append({"model": "TimesNet", "seed": seed, "expected_path": str(timesnet_path)})
        for short_name in models:
            model_name = BASELINE_SPECS.get(short_name)
            if model_name is None:
                continue
            path = baseline_metrics_path(seed_dir / short_name, model_name)
            if metrics_are_valid(path, "metrics"):
                payload = load_json(path)
                rows.append({"model": short_name, "seed": seed, **payload["metrics"]})
            else:
                missing.append({"model": short_name, "seed": seed, "expected_path": str(path)})
    return rows, missing


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        record: dict[str, Any] = {
            "model": model,
            "n": len(model_rows),
            "seeds": [row["seed"] for row in model_rows],
        }
        for metric in ("MAE", "RMSE", "MAPE(%)", "R2", "Corr"):
            values = [float(row[metric]) for row in model_rows if row.get(metric) is not None]
            if not values:
                continue
            record[metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "values": values,
            }
        aggregates.append(record)
    return aggregates


def run_coordinator(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)
    if args.dry_run:
        for index, job in enumerate(jobs, start=1):
            print(f"[{index}/{len(jobs)}] {job.name} seed={job.seed}")
            print("  " + subprocess.list2cmdline(job.command))
            if args.capture_training_output:
                print(f"  log: {job.log_path}")
        return 0
    status_path = output_root / "run_status.json"
    status: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": str(Path(args.config).resolve()),
        "models": args.models,
        "seeds": args.seeds,
        "jobs": [],
    }

    failures = 0
    for index, job in enumerate(jobs, start=1):
        if not args.force and job_is_complete(job):
            print(f"[{index}/{len(jobs)}] SKIP {job.name} seed={job.seed}: valid outputs exist")
            outcome = "skipped_complete"
            return_code = 0
        else:
            print(f"[{index}/{len(jobs)}] RUN  {job.name} seed={job.seed}")
            if args.capture_training_output:
                job.log_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"  log: {job.log_path}")
                with job.log_path.open("a", encoding="utf-8") as log:
                    log.write("\n" + "=" * 100 + "\n")
                    log.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    log.write("Command: " + subprocess.list2cmdline(job.command) + "\n")
                    log.flush()
                    completed = subprocess.run(
                        job.command,
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
            else:
                completed = subprocess.run(
                    job.command,
                    cwd=ROOT,
                )
            return_code = completed.returncode
            if return_code == 0 and job_is_complete(job):
                outcome = "completed"
            else:
                outcome = "failed"
                failures += 1
                if args.capture_training_output:
                    print(f"  FAILED (exit={return_code}); inspect {job.log_path}")
                else:
                    print(f"  FAILED (exit={return_code})")

        status["jobs"].append(
            {
                "name": job.name,
                "seed": job.seed,
                "status": outcome,
                "return_code": return_code,
                "log": str(job.log_path) if args.capture_training_output else None,
                "outputs": [str(path) for path in job.outputs],
            }
        )
        atomic_write_json(status_path, status)
        if failures and not args.continue_on_error:
            break

    completed_jobs = [job for job in jobs if job_is_complete(job)]
    rows, missing_results = extract_available_rows(output_root, args.models, args.seeds)
    summary_path = output_root / "main_table_multiseed_summary.json"
    atomic_write_json(
        summary_path,
        {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "complete": len(completed_jobs) == len(jobs),
            "completed_jobs": len(completed_jobs),
            "expected_jobs": len(jobs),
            "protocol": {
                "seeds": args.seeds,
                "pred_len": args.pred_len,
                "aux_delta_scale": args.aux_delta_scale,
                "router_init_bias": args.router_init_bias,
                "sample_std_ddof": 1,
            },
            "per_seed": rows,
            "aggregate": aggregate_rows(rows),
            "missing_results": missing_results,
        },
    )
    print(f"Summary: {summary_path}")
    if missing_results:
        print("Some results are incomplete; rerun the same command to resume only missing jobs.")

    status["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status["failure_count"] = failures
    atomic_write_json(status_path, status)
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all stochastic models in the main comparison for multiple seeds."
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--suite-config", default=str(DEFAULT_SUITE_CONFIG))
    parser.add_argument(
        "--output-root",
        default="experiments/main_table_multiseed/checkpoints/main_p96",
    )
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Random seeds to run. Seed 0 is omitted by default because it already exists.",
    )
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--label-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--aux-delta-scale", type=float, default=0.1)
    parser.add_argument("--router-init-bias", type=float, default=-5.0)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--device-ids", default=None)
    parser.set_defaults(reuse_tame_backbone=True, continue_on_error=True)
    parser.add_argument(
        "--reuse-tame-backbone",
        dest="reuse_tame_backbone",
        action="store_true",
        help="Initialize Cross-Fusion from the same seed's trained Stage-1a TimesNet backbone (default).",
    )
    parser.add_argument(
        "--no-reuse-tame-backbone",
        dest="reuse_tame_backbone",
        action="store_false",
        help="Train Cross-Fusion without loading the paired Stage-1a backbone.",
    )
    parser.add_argument("--force", action="store_true", help="Rerun completed jobs.")
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without training.")
    parser.add_argument(
        "--capture-training-output",
        action="store_true",
        help="Save verbose per-job stdout/stderr logs. Disabled by default.",
    )
    parser.add_argument("--continue-on-error", dest="continue_on_error", action="store_true")
    parser.add_argument("--stop-on-error", dest="continue_on_error", action="store_false")

    # Internal worker arguments used by the coordinator subprocess.
    parser.add_argument("--worker-tame", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-run-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, default=0, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker_tame:
        if not args.worker_run_dir:
            raise ValueError("--worker-run-dir is required with --worker-tame")
        return run_tame_worker(args)
    return run_coordinator(args)


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import sys
from pathlib import Path
from typing import Optional

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.common import banner, log, setup_nnunet_environment
from source.preprocess import ensure_preprocessed_ready
from source.train.specs import TrainModelSpec, load_train_specs
from source.train.train_cmd import run_nnunet_train


def parse_args():
    here = THIS_DIR
    parser = argparse.ArgumentParser("Vesuvius Surface Detection - Train Only")
    parser.add_argument("--working_dir", type=str, default="./work")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument(
        "--active_train_yaml",
        type=str,
        default=str(here / "configs" / "train" / "models_active_set2.yaml"),
        help="Active training models yaml (used).",
    )
    parser.add_argument(
        "--unused_train_yaml",
        type=str,
        default=str(here / "configs" / "train" / "models_unused.yaml"),
        help="Unused training models yaml (kept separate by request, not executed).",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="1,2,5,7",
        help="Comma-separated modes to train, must exist in active_train_yaml.",
    )
    parser.add_argument("--configuration", type=str, default="3d_fullres")
    parser.add_argument("--fold", type=str, default="all")
    parser.add_argument("--command_logs_dir", type=str, default="")
    parser.add_argument("--timeout_sec", type=int, default=0)
    return parser.parse_args()


def _parse_modes(raw: str, available: list[int]) -> list[int]:
    if not raw.strip():
        return sorted(set(int(x) for x in available))
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _model_output_dir(results_root: Path, spec: TrainModelSpec, configuration: str, fold: str) -> Path:
    return results_root / spec.dataset_name / f"{spec.trainer}__{spec.plans_name}__{configuration}" / f"fold_{fold}"


def _resolve_pretrained_checkpoint(
    results_root: Path,
    source_spec: TrainModelSpec,
    configuration: str,
    fold: str,
) -> Path:
    fold_dir = _model_output_dir(results_root, source_spec, configuration, fold)
    c_final = fold_dir / "checkpoint_final.pth"
    c_best = fold_dir / "checkpoint_best.pth"
    if c_final.exists():
        return c_final
    if c_best.exists():
        return c_best
    raise FileNotFoundError(f"pretrained checkpoint not found: {fold_dir}")


def main():
    args = parse_args()
    specs = load_train_specs(Path(args.active_train_yaml).resolve())
    modes = _parse_modes(args.modes, list(specs.keys()))

    for m in modes:
        if m not in specs:
            raise ValueError(f"mode {m} is not defined in {args.active_train_yaml}")

    working_dir = Path(args.working_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    logs_dir = Path(args.command_logs_dir).resolve() if args.command_logs_dir else (output_dir / "logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    timeout_sec: Optional[int] = args.timeout_sec if args.timeout_sec > 0 else None

    banner("Vesuvius nnUNet Train Only")
    log(f"working_dir={working_dir}")
    log(f"output_dir={output_dir}")
    log(f"active_train_yaml={Path(args.active_train_yaml).resolve()}")
    log(f"modes={modes}")
    log(f"configuration={args.configuration}, fold={args.fold}")
    log(f"command_logs_dir={logs_dir}")

    unused_yaml = Path(args.unused_train_yaml).resolve()
    if unused_yaml.exists():
        log(f"unused_train_yaml exists (kept separate): {unused_yaml}")

    paths = setup_nnunet_environment(working_dir=working_dir, output_dir=output_dir, compile_flag="true")

    for idx, mode in enumerate(modes, start=1):
        spec = specs[mode]
        log("-" * 60)
        log(f"[{idx}/{len(modes)}] mode={mode} | {spec.tag}")
        log(
            f"dataset={spec.dataset_name} plans={spec.plans_name} trainer={spec.trainer} "
            f"patch={spec.patch_size} batch={spec.batch_size}"
        )

        ensure_preprocessed_ready(
            preprocessed_root=paths.preprocessed,
            dataset_name=spec.dataset_name,
            configuration=args.configuration,
        )

        pretrained_weights = None
        if spec.pretrained_from_mode is not None:
            if spec.pretrained_from_mode not in specs:
                raise ValueError(
                    f"mode {mode} requires pretrained_from_mode={spec.pretrained_from_mode}, "
                    "but it is not in active yaml."
                )
            pretrained_weights = _resolve_pretrained_checkpoint(
                results_root=paths.results,
                source_spec=specs[spec.pretrained_from_mode],
                configuration=args.configuration,
                fold=args.fold,
            )
            log(f"pretrained_weights={pretrained_weights}")

        ok = run_nnunet_train(
            dataset_id=spec.dataset_id,
            configuration=args.configuration,
            fold=args.fold,
            plans_name=spec.plans_name,
            trainer=spec.trainer,
            pretrained_weights=pretrained_weights,
            timeout_sec=timeout_sec,
            logs_dir=logs_dir,
        )
        if not ok:
            raise RuntimeError(f"training failed for mode {mode}")

    log("all requested training modes finished")


if __name__ == "__main__":
    main()


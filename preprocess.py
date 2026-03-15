import argparse
import json
import pickle
import shutil
import sys
import zipfile
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.common import (
    banner,
    CommandProgress,
    create_dataset_json,
    create_spacing_json,
    ensure_spacing_sidecars,
    log,
    resolve_command,
    run_command,
    setup_nnunet_environment,
)
from source.train.specs import load_train_specs


@dataclass
class StageHealthReport:
    ok: bool
    missing_case_ids: list[str]
    broken_case_ids: list[str]
    issues: list[str]


def parse_args():
    here = THIS_DIR
    parser = argparse.ArgumentParser("Vesuvius Surface Detection - Preprocess Only")
    parser.add_argument("--input_dir", type=str, default="./data")
    parser.add_argument("--working_dir", type=str, default="./work")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument(
        "--active_train_yaml",
        type=str,
        default=str(here / "configs" / "train" / "models_active_set2.yaml"),
        help="Active training models yaml (used).",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="1,2,5,7",
        help="Comma-separated modes to preprocess, must exist in active_train_yaml.",
    )
    parser.add_argument("--configuration", type=str, default="3d_fullres")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--use_copy", action="store_true")
    parser.add_argument("--force_rebuild_raw", action="store_true")
    parser.add_argument("--force_rebuild_preprocessed", action="store_true")
    parser.add_argument("--command_logs_dir", type=str, default="")
    parser.add_argument("--timeout_sec", type=int, default=0)
    return parser.parse_args()


def _parse_modes(raw: str, available: list[int]) -> list[int]:
    if not raw.strip():
        return sorted(set(int(x) for x in available))
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _prepare_single_pair(
    image_path: Path,
    train_labels_dir: Path,
    images_dir: Path,
    labels_dir: Path,
    use_symlinks: bool,
) -> bool:
    import tifffile

    case_id = image_path.stem
    label_path = train_labels_dir / image_path.name
    if not label_path.exists():
        return False

    try:
        # Validate TIFF readability via metadata only to avoid full-volume decode.
        with tifffile.TiffFile(str(image_path)) as tif_img:
            if len(tif_img.pages) == 0:
                raise ValueError("empty tiff")
            _ = tif_img.pages[0].shape
        with tifffile.TiffFile(str(label_path)) as tif_lbl:
            if len(tif_lbl.pages) == 0:
                raise ValueError("empty tiff")
            _ = tif_lbl.pages[0].shape
    except Exception as e:
        log(f"failed to read {image_path.name}: {e}")
        return False

    img_dst = images_dir / f"{case_id}_0000.tif"
    img_json = images_dir / f"{case_id}_0000.json"
    lbl_dst = labels_dir / f"{case_id}.tif"
    lbl_json = labels_dir / f"{case_id}.json"

    if use_symlinks:
        if not img_dst.exists():
            img_dst.symlink_to(image_path.resolve())
        if not lbl_dst.exists():
            lbl_dst.symlink_to(label_path.resolve())
    else:
        shutil.copy2(image_path, img_dst)
        shutil.copy2(label_path, lbl_dst)

    create_spacing_json(img_json)
    create_spacing_json(lbl_json)
    return True


def prepare_raw_dataset(
    input_dir: Path,
    raw_root: Path,
    dataset_name: str,
    num_workers: int,
    use_symlinks: bool,
    force_rebuild: bool,
    max_cases: Optional[int] = None,
) -> Path:
    from tqdm.auto import tqdm

    dataset_dir = raw_root / dataset_name
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"

    if force_rebuild and dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    train_images_dir = input_dir / "train_images"
    train_labels_dir = input_dir / "train_labels"
    if not train_images_dir.exists() or not train_labels_dir.exists():
        raise FileNotFoundError(f"missing train_images/train_labels under {input_dir}")

    image_files = sorted(train_images_dir.glob("*.tif"))
    if max_cases is not None and max_cases > 0:
        image_files = image_files[:max_cases]

    log(f"raw dataset={dataset_name} | cases={len(image_files)} | workers={num_workers}")
    worker = partial(
        _prepare_single_pair,
        train_labels_dir=train_labels_dir,
        images_dir=images_dir,
        labels_dir=labels_dir,
        use_symlinks=use_symlinks,
    )

    if num_workers > 1 and len(image_files) > 1:
        with Pool(num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(worker, image_files),
                    total=len(image_files),
                    desc=f"Prepare {dataset_name}",
                )
            )
    else:
        results = [worker(p) for p in tqdm(image_files, desc=f"Prepare {dataset_name}")]

    converted = sum(1 for x in results if x)
    create_dataset_json(dataset_dir, num_training=converted)
    log(f"raw dataset ready: {dataset_dir} (converted={converted})")
    if converted == 0:
        raise RuntimeError(f"no valid training pair for {dataset_name}")
    return dataset_dir


def run_plan_and_preprocess(
    dataset_id: int,
    planner: str,
    configuration: str,
    num_workers: int,
    stage_dir: Path,
    expected_cases: Optional[int],
    timeout_sec: Optional[int],
    logs_dir: Optional[Path],
) -> bool:
    exe = resolve_command("nnUNetv2_plan_and_preprocess")
    cmd: list[str] = [
        exe,
        "-d",
        f"{dataset_id:03d}",
        "-np",
        str(num_workers),
        "-pl",
        planner,
        "-c",
        configuration,
    ]
    ok, _ = run_command(
        cmd=cmd,
        name="Preprocessing",
        timeout_sec=timeout_sec,
        logs_dir=logs_dir,
        progress=CommandProgress(
            label="nnUNet preprocess",
            total=expected_cases if (expected_cases is not None and expected_cases > 0) else None,
            unit="cases",
            watch_dir=stage_dir,
            watch_glob="*.pkl",
            heartbeat_sec=20,
            emit_every=1,
        ),
        shell=False,
    )
    return ok


def _is_valid_pickle(path: Path) -> tuple[bool, str]:
    try:
        if path.stat().st_size <= 0:
            return False, "empty file"
        with path.open("rb") as f:
            _ = pickle.load(f)
        return True, ""
    except Exception as e:
        return False, repr(e)


def _is_valid_npz(path: Path) -> tuple[bool, str]:
    try:
        if path.stat().st_size <= 0:
            return False, "empty file"
        import numpy as np

        with np.load(str(path), allow_pickle=False) as z:
            keys = list(z.keys())
            if not keys:
                return False, "npz contains no arrays"
            arr = z[keys[0]]
            if getattr(arr, "size", 0) <= 0:
                return False, "first array is empty"
        return True, ""
    except zipfile.BadZipFile as e:
        return False, repr(e)
    except Exception as e:
        return False, repr(e)


def _is_valid_b2nd(path: Path) -> tuple[bool, str]:
    try:
        if path.stat().st_size <= 0:
            return False, "empty file"
        try:
            import blosc2
        except Exception:
            # Fallback when blosc2 is unavailable in the current environment.
            return True, ""

        arr = blosc2.open(urlpath=str(path), mode="r")
        shape = getattr(arr, "shape", None)
        if shape is None:
            return False, "cannot read b2nd shape"
        if len(shape) == 0:
            return False, "b2nd shape is empty"
        return True, ""
    except Exception as e:
        return False, repr(e)


def inspect_preprocessed_stage(stage_dir: Path, expected_case_ids: list[str]) -> StageHealthReport:
    if not stage_dir.exists():
        return StageHealthReport(
            ok=False,
            missing_case_ids=list(expected_case_ids),
            broken_case_ids=[],
            issues=[f"missing stage dir: {stage_dir}"],
        )

    missing_case_ids: set[str] = set()
    broken_case_ids: set[str] = set()
    issues: list[str] = []

    for case_id in expected_case_ids:
        pkl_file = stage_dir / f"{case_id}.pkl"
        npz_file = stage_dir / f"{case_id}.npz"
        data_b2nd_file = stage_dir / f"{case_id}.b2nd"
        seg_b2nd_file = stage_dir / f"{case_id}_seg.b2nd"

        has_b2nd = data_b2nd_file.exists() or seg_b2nd_file.exists()
        has_pair = (
            pkl_file.exists()
            and ((data_b2nd_file.exists() and seg_b2nd_file.exists()) or npz_file.exists())
        )
        if not has_pair:
            missing_case_ids.add(case_id)
            missing_parts: list[str] = []
            if not pkl_file.exists():
                missing_parts.append(pkl_file.name)
            if has_b2nd:
                if not data_b2nd_file.exists():
                    missing_parts.append(data_b2nd_file.name)
                if not seg_b2nd_file.exists():
                    missing_parts.append(seg_b2nd_file.name)
            elif not npz_file.exists():
                missing_parts.append(f"{case_id}.npz_or_b2nd_pair")
            issues.append(f"{case_id}: missing {'/'.join(missing_parts)}")
            continue

        pkl_ok, pkl_msg = _is_valid_pickle(pkl_file)
        if not pkl_ok:
            broken_case_ids.add(case_id)
            issues.append(f"{case_id}: broken {pkl_file.name} ({pkl_msg})")
            continue

        if has_b2nd:
            data_ok, data_msg = _is_valid_b2nd(data_b2nd_file)
            seg_ok, seg_msg = _is_valid_b2nd(seg_b2nd_file)
            if not data_ok:
                broken_case_ids.add(case_id)
                issues.append(f"{case_id}: broken {data_b2nd_file.name} ({data_msg})")
            if not seg_ok:
                broken_case_ids.add(case_id)
                issues.append(f"{case_id}: broken {seg_b2nd_file.name} ({seg_msg})")
        else:
            npz_ok, npz_msg = _is_valid_npz(npz_file)
            if not npz_ok:
                broken_case_ids.add(case_id)
                issues.append(f"{case_id}: broken {npz_file.name} ({npz_msg})")

    missing = sorted(missing_case_ids)
    broken = sorted(broken_case_ids)
    return StageHealthReport(
        ok=(len(missing) == 0 and len(broken) == 0),
        missing_case_ids=missing,
        broken_case_ids=broken,
        issues=issues,
    )


def _remove_case_preprocessed_files(stage_dir: Path, case_id: str) -> int:
    removed = 0
    names = [
        f"{case_id}.pkl",
        f"{case_id}.npz",
        f"{case_id}.npy",
        f"{case_id}.b2nd",
        f"{case_id}_seg.b2nd",
        f"{case_id}_seg.npy",
        f"{case_id}_seg.npz",
    ]
    for name in names:
        p = stage_dir / name
        if p.exists():
            p.unlink()
            removed += 1
    return removed


def repair_preprocessed_stage(stage_dir: Path, case_ids: list[str]) -> int:
    total_removed = 0
    for case_id in sorted(set(case_ids)):
        total_removed += _remove_case_preprocessed_files(stage_dir, case_id)
    return total_removed


def validate_preprocessed_stage(
    stage_dir: Path,
    expected_case_ids: Optional[list[str]] = None,
) -> tuple[bool, str]:
    if expected_case_ids is None:
        pkl_files = sorted(stage_dir.glob("*.pkl"))
        if not pkl_files:
            return False, f"no case metadata (.pkl) found in {stage_dir}"
        expected_case_ids = [p.stem for p in pkl_files]

    report = inspect_preprocessed_stage(stage_dir, expected_case_ids)
    if report.ok:
        return True, ""
    sample = "; ".join(report.issues[:10])
    return (
        False,
        f"preprocessed stage invalid: missing_cases={len(report.missing_case_ids)}, "
        f"broken_cases={len(report.broken_case_ids)} | sample: {sample}",
    )


def ensure_preprocessed_ready(
    preprocessed_root: Path,
    dataset_name: str,
    configuration: str,
    expected_case_ids: Optional[list[str]] = None,
):
    stage_dir = preprocessed_root / dataset_name / f"nnUNetPlans_{configuration}"
    if not stage_dir.exists():
        raise RuntimeError(f"missing preprocessed stage: {stage_dir}")
    ok, msg = validate_preprocessed_stage(stage_dir, expected_case_ids=expected_case_ids)
    if not ok:
        raise RuntimeError(msg)
    log(f"preprocessed stage ready: {stage_dir}")


def _find_plans_file(dataset_dir: Path, plans_name: str) -> Optional[Path]:
    candidates = [
        dataset_dir / f"{plans_name}.json",
        dataset_dir / "nnUNetPlans.json",
        dataset_dir / "plans.json",
    ]
    for c in candidates:
        if c.exists():
            return c

    for c in sorted(dataset_dir.glob("*.json")):
        if c.name == "dataset.json":
            continue
        return c
    return None


def override_patch_and_batch(
    preprocessed_root: Path,
    dataset_name: str,
    plans_name: str,
    configuration: str,
    patch_size: int,
    batch_size: int,
):
    dataset_dir = preprocessed_root / dataset_name
    plans_file = _find_plans_file(dataset_dir, plans_name)
    if plans_file is None:
        raise RuntimeError(f"plans file not found under {dataset_dir}")

    plans = json.loads(plans_file.read_text(encoding="utf-8"))
    configs = plans.get("configurations")
    if not isinstance(configs, dict):
        raise RuntimeError(f"invalid plans format in {plans_file}")
    conf = configs.get(configuration)
    if not isinstance(conf, dict):
        raise RuntimeError(f"configuration '{configuration}' not found in {plans_file}")

    old_patch = conf.get("patch_size")
    old_batch = conf.get("batch_size")
    new_patch = [int(patch_size), int(patch_size), int(patch_size)]
    new_batch = int(batch_size)

    same_patch = False
    if isinstance(old_patch, (list, tuple)):
        try:
            same_patch = [int(v) for v in old_patch] == new_patch
        except Exception:
            same_patch = False

    same_batch = False
    try:
        same_batch = int(old_batch) == new_batch
    except Exception:
        same_batch = False

    if same_patch and same_batch:
        log(
            f"plans already match ({plans_file.name}, {configuration}): "
            f"patch_size={new_patch}, batch_size={new_batch}"
        )
        return

    conf["patch_size"] = new_patch
    conf["batch_size"] = new_batch
    plans_file.write_text(json.dumps(plans, indent=2), encoding="utf-8")

    log(
        f"override plans ({plans_file.name}, {configuration}): "
        f"patch_size {old_patch} -> {new_patch}, "
        f"batch_size {old_batch} -> {new_batch}"
    )


def main():
    args = parse_args()
    specs = load_train_specs(Path(args.active_train_yaml).resolve())
    modes = _parse_modes(args.modes, list(specs.keys()))
    for m in modes:
        if m not in specs:
            raise ValueError(f"mode {m} is not defined in {args.active_train_yaml}")

    input_dir = Path(args.input_dir).resolve()
    working_dir = Path(args.working_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    logs_dir = Path(args.command_logs_dir).resolve() if args.command_logs_dir else (output_dir / "logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    timeout_sec: Optional[int] = args.timeout_sec if args.timeout_sec > 0 else None

    banner("Vesuvius nnUNet Preprocess Only")
    log(f"input_dir={input_dir}")
    log(f"working_dir={working_dir}")
    log(f"output_dir={output_dir}")
    log(f"active_train_yaml={Path(args.active_train_yaml).resolve()}")
    log(f"modes={modes}")
    log(f"configuration={args.configuration}, num_workers={args.num_workers}")
    log(f"command_logs_dir={logs_dir}")

    paths = setup_nnunet_environment(working_dir=working_dir, output_dir=output_dir, compile_flag="true")

    for idx, mode in enumerate(modes, start=1):
        spec = specs[mode]
        log("-" * 60)
        log(f"[{idx}/{len(modes)}] mode={mode} | {spec.tag}")
        log(
            f"dataset={spec.dataset_name} planner={spec.planner} plans={spec.plans_name} "
            f"patch={spec.patch_size} batch={spec.batch_size}"
        )

        dataset_dir = paths.raw / spec.dataset_name
        if dataset_dir.exists() and not args.force_rebuild_raw:
            created = ensure_spacing_sidecars(dataset_dir)
            if created > 0:
                log(f"created missing spacing sidecars: {created}")
            else:
                log(f"raw dataset exists: {dataset_dir}")
        else:
            prepare_raw_dataset(
                input_dir=input_dir,
                raw_root=paths.raw,
                dataset_name=spec.dataset_name,
                num_workers=args.num_workers,
                use_symlinks=not args.use_copy,
                force_rebuild=args.force_rebuild_raw,
                max_cases=args.max_cases if args.max_cases > 0 else None,
            )

        preprocessed_dataset_dir = paths.preprocessed / spec.dataset_name
        if args.force_rebuild_preprocessed and preprocessed_dataset_dir.exists():
            log(f"force_rebuild_preprocessed=True, removing: {preprocessed_dataset_dir}")
            shutil.rmtree(preprocessed_dataset_dir)

        stage_dir = preprocessed_dataset_dir / f"nnUNetPlans_{args.configuration}"
        expected_case_ids = sorted(
            p.name.replace("_0000.tif", "") for p in (dataset_dir / "imagesTr").glob("*_0000.tif")
        )
        expected_cases = len(expected_case_ids)

        needs_preprocess = not stage_dir.exists()
        if not needs_preprocess:
            report = inspect_preprocessed_stage(stage_dir, expected_case_ids=expected_case_ids)
            if report.ok:
                log(
                    f"preprocessed stage is healthy: {stage_dir} "
                    f"(cases={expected_cases})"
                )
            else:
                needs_preprocess = True
                sample = "; ".join(report.issues[:10])
                log(
                    "preprocessed stage has missing/broken files, auto-repair enabled: "
                    f"missing_cases={len(report.missing_case_ids)}, "
                    f"broken_cases={len(report.broken_case_ids)}"
                )
                if sample:
                    log(f"issue sample: {sample}")
                cases_to_rebuild = sorted(set(report.missing_case_ids + report.broken_case_ids))
                removed = repair_preprocessed_stage(stage_dir, case_ids=cases_to_rebuild)
                log(
                    f"removed stale/corrupted files: {removed} "
                    f"(cases_to_rebuild={len(cases_to_rebuild)})"
                )

        if needs_preprocess:
            ok = run_plan_and_preprocess(
                dataset_id=spec.dataset_id,
                planner=spec.planner,
                configuration=args.configuration,
                num_workers=args.num_workers,
                stage_dir=stage_dir,
                expected_cases=expected_cases,
                timeout_sec=timeout_sec,
                logs_dir=logs_dir,
            )
            if not ok:
                raise RuntimeError(f"preprocess failed for mode {mode}")

        ensure_preprocessed_ready(
            preprocessed_root=paths.preprocessed,
            dataset_name=spec.dataset_name,
            configuration=args.configuration,
            expected_case_ids=expected_case_ids,
        )
        override_patch_and_batch(
            preprocessed_root=paths.preprocessed,
            dataset_name=spec.dataset_name,
            plans_name=spec.plans_name,
            configuration=args.configuration,
            patch_size=spec.patch_size,
            batch_size=spec.batch_size,
        )

    log("all requested preprocessing modes finished")


if __name__ == "__main__":
    main()


import json
from pathlib import Path


def create_spacing_json(path: Path, spacing=(1.0, 1.0, 1.0)):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"spacing": list(spacing)}, f)


def ensure_spacing_sidecars(dataset_dir: Path) -> int:
    created = 0
    for sub in ("imagesTr", "labelsTr", "imagesTs"):
        folder = dataset_dir / sub
        if not folder.exists():
            continue
        for tif_path in folder.glob("*.tif"):
            spacing_path = tif_path.with_suffix(".json")
            if not spacing_path.exists():
                create_spacing_json(spacing_path)
                created += 1
    return created


def create_dataset_json(output_dir: Path, num_training: int):
    payload = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "surface": 1, "ignore": 2},
        "numTraining": num_training,
        "file_ending": ".tif",
        "overwrite_image_reader_writer": "SimpleTiffIO",
    }
    with open(output_dir / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


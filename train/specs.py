from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from source.common import get_dataset_name, load_yaml


@dataclass
class TrainModelSpec:
    mode: int
    tag: str
    planner: str
    plans_name: str
    trainer: str
    patch_size: int
    batch_size: int
    pretrained_from_mode: Optional[int]

    @property
    def dataset_id(self) -> int:
        return 100 + int(self.mode)

    @property
    def dataset_name(self) -> str:
        return get_dataset_name(self.dataset_id, suffix="VesuviusSurface", mode=self.mode)


def _parse_spec(raw: dict, context: str) -> TrainModelSpec:
    try:
        return TrainModelSpec(
            mode=int(raw["mode"]),
            tag=str(raw["tag"]),
            planner=str(raw["planner"]),
            plans_name=str(raw["plans_name"]),
            trainer=str(raw["trainer"]),
            patch_size=int(raw["patch_size"]),
            batch_size=int(raw["batch_size"]),
            pretrained_from_mode=(
                int(raw["pretrained_from_mode"])
                if raw.get("pretrained_from_mode") is not None
                else None
            ),
        )
    except KeyError as e:
        raise KeyError(f"Missing key {e} in train model config: {context}") from e


def _resolve_yaml_path(base_yaml: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (base_yaml.parent / p).resolve()


def load_train_specs(path: Path) -> dict[int, TrainModelSpec]:
    payload = load_yaml(path)
    specs: dict[int, TrainModelSpec] = {}
    if isinstance(payload, dict) and "model_yamls" in payload:
        model_yamls = payload.get("model_yamls", [])
        for rel in model_yamls:
            model_yaml = _resolve_yaml_path(path, str(rel))
            model_payload = load_yaml(model_yaml)
            if not isinstance(model_payload, dict):
                raise ValueError(f"model yaml must be a dict: {model_yaml}")
            raw = model_payload.get("model", model_payload)
            if not isinstance(raw, dict):
                raise ValueError(f"model yaml 'model' field must be a dict: {model_yaml}")
            spec = _parse_spec(raw, str(model_yaml))
            if spec.mode in specs:
                raise ValueError(f"duplicate mode {spec.mode} in {path}")
            specs[spec.mode] = spec
        return specs

    models = payload.get("models", []) if isinstance(payload, dict) else []
    for m in models:
        spec = _parse_spec(m, str(path))
        if spec.mode in specs:
            raise ValueError(f"duplicate mode {spec.mode} in {path}")
        specs[spec.mode] = spec
    return specs

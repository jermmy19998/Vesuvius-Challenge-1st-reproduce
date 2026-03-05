from pathlib import Path
from typing import Any


def load_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError(
            "PyYAML is required. Install with: python -m pip install pyyaml"
        ) from e

    if not path.exists():
        raise FileNotFoundError(f"yaml file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


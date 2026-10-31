from pathlib import Path
import json


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_class_names(names, path: str | Path) -> None:
    Path(path).write_text(json.dumps(list(names), indent=2), encoding="utf-8")


def load_class_names(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

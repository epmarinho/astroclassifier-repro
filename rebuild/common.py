import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def output_path(path):
    target = Path(path).resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError("Outputs must remain under rebuild/")
    if target.is_relative_to(ROOT / "reports/reaudit_20261016"):
        raise ValueError("The reconciled audit is immutable")
    return target


def write_json(path, value):
    target = output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)

import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys

from .common import PROJECT, ROOT, file_hash, read_json, write_json


def git_state():
    environment = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    commands = {
        "status": ["status", "--porcelain=v2", "--branch", "--untracked-files=all", "--", ".", ":(exclude)rebuild"],
        "ignored": ["status", "--short", "--ignored=matching", "--", ".", ":(exclude)rebuild"],
    }
    state = {name: subprocess.check_output(["git", *arguments], cwd=PROJECT, env=environment, text=True) for name, arguments in commands.items()}
    state["index_sha256"] = file_hash(PROJECT / ".git/index")
    return state


def source_snapshot():
    files = {}
    directories = []
    for folder, subdirectories, filenames in os.walk(PROJECT, followlinks=False):
        if Path(folder) == PROJECT:
            subdirectories[:] = [name for name in subdirectories if name != "rebuild"]
        for name in sorted(subdirectories + filenames):
            path = Path(folder) / name
            relative = path.relative_to(PROJECT).as_posix()
            if path.is_symlink():
                files[relative] = {"kind": "symlink", "target": os.readlink(path)}
            elif path.is_file():
                files[relative] = {"kind": "file", "bytes": path.stat().st_size, "sha256": file_hash(path)}
            elif path.is_dir():
                directories.append(relative)
    return {"files": dict(sorted(files.items())), "directories": sorted(directories), "git": git_state()}


def environment_report():
    libraries = {}
    for name in ["torch", "torchvision", "numpy", "Pillow", "pytest", "h5py", "pillow-avif-plugin"]:
        try:
            libraries[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            libraries[name] = None
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "libraries": libraries}


def verify(initial, current):
    before, after = initial["files"], current["files"]
    return {
        "missing": sorted(before.keys() - after.keys()),
        "unexpected_outside_rebuild": sorted(after.keys() - before.keys()),
        "changed": sorted(name for name in before.keys() & after.keys() if before[name] != after[name]),
        "directories_unchanged": initial["directories"] == current["directories"],
        "git_status_unchanged_excluding_authorized_rebuild": initial["git"] == current["git"],
        "index_sha256": current["git"]["index_sha256"],
        "checked_preexisting_files": len(before),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["snapshot", "verify", "inventory"])
    arguments = parser.parse_args()
    if arguments.action == "snapshot":
        write_json(ROOT / "reports/initial_state.json", source_snapshot())
        write_json(ROOT / "reports/environment.json", environment_report())
    elif arguments.action == "verify":
        result = verify(read_json(ROOT / "reports/initial_state.json"), source_snapshot())
        write_json(ROOT / "reports/preservation_validation.json", result)
        print(result)
        if result["missing"] or result["unexpected_outside_rebuild"] or result["changed"] or not result["directories_unchanged"] or not result["git_status_unchanged_excluding_authorized_rebuild"]:
            raise SystemExit(1)
    else:
        target = ROOT / "reports/created_files.sha256"
        paths = sorted(path for path in ROOT.rglob("*") if path.is_file() and path != target)
        with target.open("x", encoding="utf-8") as stream:
            for path in paths:
                stream.write(f"{file_hash(path)}  {path.relative_to(PROJECT).as_posix()}\n")
        print("Inventory excludes itself; its SHA-256:", file_hash(target))


if __name__ == "__main__":
    main()

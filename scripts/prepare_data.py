# Prepare and verify the paper's fixed Adult and Bank benchmark inputs.

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path


def array_hashes(folder):
    import numpy as np
    result = {}
    for dataset in ["adult", "bank"]:
        with np.load(folder / f"{dataset}.npz") as data:
            result[dataset] = {key: dict(shape=list(data[key].shape), dtype=str(data[key].dtype),
                sha256=hashlib.sha256(data[key].tobytes(order="C")).hexdigest()) for key in data.files}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = args.output.resolve() if args.output else root / "data"
    if folder != root / "data" and (folder == root or root not in folder.parents):
        raise ValueError("Choose the data directory or a separate directory within the repository.")
    expected = json.loads((root / "data/array_hashes.json").read_text())
    if all((folder / name).is_file() for name in ["adult.npz", "bank.npz"]):
        if array_hashes(folder) != expected:
            raise AssertionError("Processed data differ from the paper's fixed splits.")
        print("Adult and Bank input arrays match the fixed paper splits.")
        return
    sys.path.insert(0, str(root / "code"))
    import data
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="benchmark_inputs_", dir=root) as temporary:
        scratch = Path(temporary)
        for name in ["adult.zip", "bank.zip"]:
            if (root / "data" / name).is_file():
                shutil.copy2(root / "data" / name, scratch / name)
        data.CACHE = scratch
        data.prepare()
        if array_hashes(scratch) != expected:
            raise AssertionError("Downloaded/preprocessed arrays do not match the fixed paper splits.")
        reference = json.loads((root / "data/manifest.json").read_text())["sha256"]
        for name, expected_hash in reference.items():
            if hashlib.sha256((scratch / name).read_bytes()).hexdigest() != expected_hash:
                raise AssertionError(f"Benchmark archive checksum mismatch: {name}")
        folder.mkdir(parents=True, exist_ok=True)
        for name in ["adult.zip", "bank.zip", "adult.npz", "bank.npz"]:
            if (folder / name).exists():
                if (folder / name).read_bytes() != (scratch / name).read_bytes():
                    raise FileExistsError(f"Conflicting input file: {folder / name}")
            else:
                shutil.copy2(scratch / name, folder / name)
        print("Prepared Adult and Bank inputs; arrays and archive hashes verified.")


if __name__ == "__main__":
    main()

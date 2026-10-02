# Verify code, fixed configurations, numerical coverage, figures, and stored results.

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import tokenize
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", action="store_true", help="Check every locally stored JSON/NPZ result.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checksums = read(root / "checksums.json")
    for row in checksums:
        path = root / row["path"]
        require(path.is_file() and digest(path) == row["sha256"], f"Checksum mismatch: {path}")
    python_files = [path for name in ["code", "scripts", "experiments"]
                    for path in (root / name).rglob("*.py")]
    for path in python_files:
        source = path.read_text()
        ast.parse(source)
        comments = [t for t in tokenize.generate_tokens(io.StringIO(source).readline) if t.type == tokenize.COMMENT]
        require(len(comments) == 1 and comments[0].start[0] == 1, f"Incorrect comment convention: {path}")
    import pandas as pd
    rows = pd.read_csv(root / "results/crossover/runs.csv", float_precision="round_trip")
    require(len(rows) == 2448 and rows.rho.nunique() == 17, "Incomplete crossover grid.")
    require(set(rows.tau) == {0., .015, .03, .06} and rows.rho.min() == 1e-5 and rows.rho.max() == 1.,
            "Crossover temperatures or endpoints differ.")
    for _, group in rows.groupby(["dataset", "tau", "rho"]):
        require(len(group) == 12 and group.seed.nunique() == 12, "Incomplete paired crossover cell.")
    temperature = pd.read_csv(root / "results/sensitivity/temperature.csv")
    require(len(temperature) == 1320 and temperature.tau.nunique() == 22, "Incorrect sensitivity coverage.")
    require((temperature.exact_soft <= temperature.bound_soft + 1e-10).all(), "Measured sensitivity bound violation.")
    tail = temperature[temperature.tau.isin([20, 50, 100])]
    require(len(tail) == 180 and (tail[["exact_soft", "bound_soft"]] < .01 * math.sqrt(2)).all().all(),
            "Sensitivity tail criterion failed.")
    for filename, nested in [("adult_bank_settings.json", True), ("xor3_settings.json", False)]:
        settings = read(root / "results/equal_noise" / filename)
        for row in (settings["metadata"] if nested else settings)["settings"]:
            require(abs(math.tanh(row["reach"] / (2 * row["tau"])) - row["q"]) < 1e-12,
                    "Noise-ratio metadata mismatch.")
    summary = pd.read_csv(root / "results/generalization/confirmation_summary.csv")
    require(len(summary) == 680 and summary.n_pairs.eq(12).all(), "Incomplete primary Holm family.")
    figures = read(root / "figures/manifest.json")
    require(len(figures) == 30 and len({row["standard_stem"] for row in figures}) == 30,
            "Incorrect paper figure coverage.")
    for row in figures:
        for extension in ["png", "pdf"]:
            item = row["files"][extension]
            require(digest(root / item["path"]) == item["sha256"], "Paper figure changed.")
    if args.raw:
        with (root / "results/raw/index.csv").open(newline="", encoding="utf-8") as handle:
            for i, row in enumerate(csv.DictReader(handle), 1):
                path = root / row["path"]
                require(path.is_file() and path.stat().st_size == int(row["bytes"])
                        and digest(path) == row["sha256"], f"Stored result mismatch: {path}")
                if i % 5000 == 0:
                    print(f"Verified {i} raw result files", flush=True)
    print(json.dumps(dict(passed=True, code_files=len(python_files), integrity_files=len(checksums),
                         figures=30, crossover_rows=2448, sensitivity_rows=1320,
                         primary_confirmation_contrasts=680, raw_results_checked=args.raw), indent=2))


if __name__ == "__main__":
    main()

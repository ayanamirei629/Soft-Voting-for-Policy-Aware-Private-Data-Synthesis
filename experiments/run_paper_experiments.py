# Run the paper's fixed experiment configurations in separate workspaces.

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib
import json
import re
import shutil
import sys
from pathlib import Path


FAMILIES = ["crossover", "equal_noise", "temperature_sweep", "interaction", "forecast",
            "mechanism", "generalization_pilot", "generalization_confirmation",
            "sensitivity", "reach", "runtime"]
MEASUREMENTS = {"sensitivity", "reach", "runtime"}


def configurations(root, family):
    return json.loads((root / "experiments/configurations" / f"{family}.json").read_text())


def run_job(job):
    workspace = Path(job["workspace"])
    sys.path.insert(0, str(workspace))
    module = importlib.import_module(job["module"])
    cfg = dict(job["config"])
    if job["module"] == "generalization_study":
        cfg["engine_signature"] = module.signature(module.engine_sources())
    result = module.worker(cfg)
    return dict(module=job["module"], configuration=cfg, result=result)


def run_measurement(family, workspace):
    sys.path.insert(0, str(workspace))
    module = importlib.import_module("measurement_experiments")
    module.OUT = workspace / "measurements" / family
    module.OUT.mkdir(parents=True)
    function = {"sensitivity": "temperature_sensitivity", "reach": "ordinal_benchmark",
                "runtime": "runtime_nxn"}[family]
    frame = getattr(module, function)()
    if family == "sensitivity":
        import math
        tail = frame[frame.tau.isin([20, 50, 100])]
        if not (frame.exact_soft <= frame.bound_soft + 1e-10).all():
            raise AssertionError("Measured sensitivity exceeds its bound.")
        passed = (tail[["exact_soft", "bound_soft"]] < .01 * math.sqrt(2)).all().all()
        if len(frame) != 1320 or len(tail) != 180 or not passed:
            raise AssertionError("Sensitivity coverage or tail convergence check failed.")
    return dict(family=family, rows=len(frame), output=str(module.OUT.relative_to(workspace)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=FAMILIES + ["all"], required=True)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    families = FAMILIES if args.family == "all" else [args.family]
    groups = {family: configurations(root, family)["jobs"] for family in families}
    counts = {family: "CPU measurement protocol" if family in MEASUREMENTS else len(jobs)
              for family, jobs in groups.items()}
    print(json.dumps(counts, indent=2), flush=True)
    if args.list or not args.execute:
        return
    if not args.run_id or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", args.run_id):
        parser.error("Choose a new lowercase --run-id.")
    if args.workers < 1:
        parser.error("--workers must be positive.")
    fidelity = any(family not in MEASUREMENTS for family in families)
    if fidelity:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Synthesis workers require CUDA-enabled PyTorch and a compatible GPU.")
        if any(not (root / "data" / name).is_file() for name in ["adult.npz", "bank.npz"]):
            raise FileNotFoundError("Run python scripts/prepare_data.py before synthesis.")
    target = root / "outputs/runs" / args.run_id
    if target.exists():
        raise FileExistsError("Run identifier already exists.")
    workspace = target / "workspace"
    shutil.copytree(root / "code", workspace, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if fidelity:
        shutil.copytree(root / "data", workspace / "data")
    sources = {p.relative_to(workspace).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in workspace.rglob("*") if p.is_file()}
    jobs = {}
    for family, rows in groups.items():
        for row in rows:
            key = json.dumps([row["module"], row["config"]], sort_keys=True)
            jobs.setdefault(key, dict(row, workspace=str(workspace)))
    manifest = dict(families=families, configuration_counts=counts, archived_cache_reuse=False,
                    source_sha256=sources, jobs=list(jobs.values()), completed_jobs=[],
                    measurements=[], finished=False)
    path = target / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    failures = []
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_job, row) for row in jobs.values()]
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    record = future.result()
                except Exception as exc:
                    record = dict(error=repr(exc))
                manifest["completed_jobs"].append(record)
                if record.get("error") or record.get("result", {}).get("error"):
                    failures.append(record)
                if i % 24 == 0 or i == len(futures):
                    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                    print(f"Completed {i}/{len(futures)} configurations; failures={len(failures)}", flush=True)
    try:
        for family in families:
            if family in MEASUREMENTS:
                manifest["measurements"].append(run_measurement(family, workspace))
    except Exception as exc:
        failures.append(dict(error=repr(exc)))
    manifest["failures"] = failures
    manifest["finished"] = not failures
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"Failed work retained in {target}; see run_manifest.json.")
    print(f"Experiment outputs: {target}")


if __name__ == "__main__":
    main()

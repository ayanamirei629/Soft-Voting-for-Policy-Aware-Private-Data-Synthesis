# Reproduce the paper's individual PNG/PDF panels from saved numerical tables.

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/paper_figures"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if root not in output.parents or output == root:
        raise ValueError("Choose a dedicated output directory inside the repository.")
    if output.exists():
        raise FileExistsError("Choose a new directory; previous exports are not overwritten.")
    output.mkdir(parents=True)
    sys.path.insert(0, str(root / "code"))
    import matplotlib.pyplot as plt
    import pandas as pd
    import plot_paper_figures as plots
    plots.RESULTS = root / "results"
    index = json.loads((root / "figures/manifest.json").read_text())
    names = {row["original_stem"]: row["standard_stem"] for row in index}
    exports = []

    def save(fig, name, source=None):
        if name not in names:
            plt.close(fig)
            return None
        if any(ax.get_title() for ax in fig.axes) or fig._suptitle is not None:
            raise AssertionError("Paper panels must not contain top titles.")
        stem = names[name]
        fig.canvas.draw()
        row = dict(figure=stem, axes=[dict(xlim=list(map(float, ax.get_xlim())),
                   ylim=list(map(float, ax.get_ylim())), xscale=ax.get_xscale(),
                   yscale=ax.get_yscale()) for ax in fig.axes], files={})
        for extension in ["png", "pdf"]:
            path = output / f"{stem}.{extension}"
            options = dict(bbox_inches="tight", facecolor="white")
            if extension == "png":
                options["dpi"] = 320
            fig.savefig(path, **options)
            row["files"][extension] = hashlib.sha256(path.read_bytes()).hexdigest()
        if source is not None:
            folder = output / "source_data"
            folder.mkdir(exist_ok=True)
            source.to_csv(folder / f"{stem}.csv", index=False)
        exports.append(row)
        plt.close(fig)
        print(f"Exported {stem}.png and .pdf", flush=True)
        return output / f"{stem}.png"

    plots.save = lambda fig, name, export_bundle=False: save(fig, name)
    for procedure in ["figure02_budget_panels", "figure03_direct_sensitivity", "figure04_theta_runtime",
                      "figure05_sweet_panels", "figure06_mechanism_panels", "figure07_xor_interaction",
                      "figure08_equal_noise", "figure09_forecast_panels"]:
        getattr(plots, procedure)([])
    import plot_generalization as extra
    extra.save = lambda fig, name, source, caption, family, **metadata: save(fig, name, source)
    manifest = json.loads((root / "experiments/configurations/generalization_confirmation.json").read_text())["metadata"]
    pairs = pd.read_csv(root / "results/generalization/paired_confirmation.csv", float_precision="round_trip")
    extra.new_policy_figures(manifest, pairs, ["adult", "bank"])
    extra.new_size_figures(manifest, pairs)
    if len(exports) != 30 or {row["figure"] for row in exports} != set(names.values()):
        raise AssertionError("Incomplete paper figure coverage.")
    (output / "export_manifest.json").write_text(json.dumps(exports, indent=2), encoding="utf-8")
    print("Reproduced 30 panels in PNG and PDF from saved tables.")


if __name__ == "__main__":
    main()

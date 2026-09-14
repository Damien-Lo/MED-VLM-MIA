"""
Aggregates the tuning_separation.json outputs from tune_and_eval_flickr_llava_mixture_sweep.sh
(one gn_set${std} per std, one mix_p${pp} per mixture fraction) into an 11-curve overlay of
Delta-divergence vs. noise std -- the multi-mixture version of Figure 3 (Delta Renyi-divergence
vs normalized sigma), one curve per reference-set mixture fraction instead of one
estimated-vs-true-optimal pair.

Two figures are produced, one per requested metric:
  - max_k_no_norn_kl_div      ("no normalization")
  - max_k_renyi_inf_kl_div    ("Renyi alpha=inf normalization")

Usage:
    conda activate med_vlm_mia_venv
    python scripts/ORIGINAL_SCRIPTS/plot_mixture_separation_curves.py
"""
import os
import json
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterSciNotation

OUT_ROOT = "/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/additional_ICLR_experiments/results"
FIGURES_OUT_DIR = os.path.join(OUT_ROOT, "figures")

MIXTURE_PS = ["000", "010", "020", "030", "040", "050", "060", "070", "080", "090", "100"]  # must match the sweep script
N_STD_SETS = 33  # gn_set0 .. gn_set32, must match STD_SETS in the sweep script

METRICS = {
    "max_k_no_norn_kl_div": "No normalization",
    "max_k_renyi_inf_kl_div": r"Renyi $\alpha=\infty$ normalization",
}
AUG_NAME = "GaussianNoise"
K_RATIO = 1.0  # which max-k ratio's gap to plot; change if you want a different k slice


def load_metric_curve(metric_name, pp):
    """
    Returns (stds, gaps) sorted by std, for one (metric, mixture) pair, pulled from every
    gn_set*/tuning_separation.json under mix_p{pp} -- reading the 'separation' half (mean_target
    - mean_reference at K_RATIO), not the 'best_by_setting' summary, since we want the whole
    curve across std, not just the best point.
    """
    stds, gaps = [], []
    for set_idx in range(N_STD_SETS):
        path = os.path.join(OUT_ROOT, f"mix_p{pp}", f"gn_set{set_idx}", "tuning_separation.json")
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            data = json.load(f)

        try:
            aug_settings = data["separation"]["img"]["kld_metrics"][metric_name][AUG_NAME]
        except KeyError:
            continue

        for setting_str, entry in aug_settings.items():
            setting = ast.literal_eval(setting_str)
            if setting.get("k_ratio") != K_RATIO:
                continue
            std = float(setting["std"])
            if std == 0:
                continue  # log-scale x-axis, matches the other AUC/TPR plots in this repo
            stds.append(std)
            gaps.append(entry["gap"])

    order = np.argsort(stds)
    return np.asarray(stds)[order], np.asarray(gaps)[order]


def main():
    os.makedirs(FIGURES_OUT_DIR, exist_ok=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(MIXTURE_PS)))

    for metric_name, metric_label in METRICS.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        plotted_any = False

        for pp, color in zip(MIXTURE_PS, colors):
            stds, gaps = load_metric_curve(metric_name, pp)
            if stds.size == 0:
                print(f"[{metric_name}] mix_p{pp}: no data found, skipping")
                continue
            plotted_any = True
            p = int(pp) / 100.0
            ax.plot(stds, gaps, marker="o", markersize=3, color=color,
                     label=f"p={p:.1f} ({int(pp)}% flickr non-members)")

        if not plotted_any:
            print(f"[{metric_name}] no data found for any mixture -- has the sweep been run yet?")
            plt.close(fig)
            continue

        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel(r"Noise $\sigma$ (log scale)")
        ax.set_ylabel(r"$\Delta$ Divergence (mean target-set $-$ mean reference-set)")
        ax.set_title(f"Target-vs-reference separation across the flickr/ShareGPT mixture sweep\n{metric_label}")
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, title="Reference-set mixture")

        out_path = os.path.join(FIGURES_OUT_DIR, f"{metric_name}_mixture_separation.pdf")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

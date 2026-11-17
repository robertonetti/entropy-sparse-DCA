"""
Entropy correction to the lowest-energy decile
===============================================

This script reuses the data, helper functions and figure style of
`entropy_and_energy_landscape.py`. It corrects the entropies of the four DCA models
(meDCA, edDCA, bmDCA, eaDCA) by restricting each Boltzmann distribution to its
lowest-energy decile, which is where the experimentally tested designs were
selected from.

Energy convention (adabmDCA):  P(s) = exp(-E(s)) / Z,  so  S = <E> + log Z.
For any subset A of sequence space with probability mass P(A), the entropy of
the restricted distribution P(s | A) = P(s) / P(A) is

    S_A = <E>_A + log Z + log P(A) = S_tot - (<E> - <E>_A) + log P(A).

Steps
-----
1. Global decile:  A = lowest-energy 10% of the model distribution,
       S_10 = S_tot - (<E> - <E>_10) + log(0.1).
   S_tot is taken as reliable (main-text table) and is not recomputed.
2. Per-distance decile:  inside each divergence bin B, the trusted bin
   entropies S_B (S_<model>_sample_interval.npy) are restricted to the
   lowest-energy decile of the sequences in that bin,
       S_{B,10} = S_B - (<E>_B - <E>_{B,10}) + log(0.1).
   As a consistency check, S_B is also recomputed from scratch with
       S_B = S_tot - (<E> - <E>_B) + log P(B).
3. Figures, saved in figures/intermediate/:
   - barplot of S_tot vs S_10 for every model;
   - the combined figure of entropy_and_energy_landscape.py (energy grid + effective number of
     sequences). The right panel now uses the decile-corrected entropies
     exp(S_{B,10}), and the stars use exp(S_{B,10}) * functional rate.

Run it from the scripts/ directory:
    python scripts/entropy_decile_correction.py
Energies and distances of the large samples are cached in
data/cache/sample_energy_distance_cache.npz, so later runs only
redo the analysis and the plots.
"""

import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")  # Save figures without needing a display
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from adabmDCA.fasta import get_tokens, import_from_fasta
from adabmDCA.io import load_params
from adabmDCA.functional import one_hot
from adabmDCA.statmech import compute_energy
from adabmDCA.utils import get_device


# ===============================================================================
# CONFIGURATION
# ===============================================================================

from paths import (ARCHIVES_DIR, CACHE_DIR, ENTROPY_ARRAYS_DIR, FIGURES_INTERMEDIATE,
                   GENERATED_SAMPLES_DIR, NATURAL_MSA, RELEASED_MODELS_DIR, RESULTS_DIR)

CACHE_FILE = CACHE_DIR / "sample_energy_distance_cache.npz"

MODELS = ["meDCA", "edDCA", "bmDCA", "eaDCA"]

# Total model entropies, taken as reliable (main-text table, same values as in
# verify_analysis.ipynb). They are NOT recomputed with adabmDCA.
S_TOT = {"eaDCA": 113.35, "bmDCA": 121.82, "edDCA": 136.05, "meDCA": 150.13}

# Released model parameters (same files used by verify_analysis_adabmDCA.ipynb)
PARAM_FILES = {m: RELEASED_MODELS_DIR / f"{m}_params.dat" for m in MODELS}

# Large equilibrium samples (2M sequences per model)
SAMPLE_FILES = {
    m: GENERATED_SAMPLES_DIR / f"{m}_large_sample" / "sampling_samples.fasta"
    for m in MODELS
}

# Natural MSA used to measure the divergence of the generated sequences
MSA_FILE = NATURAL_MSA

# Pre-computed arrays of the entropy analysis (folder, or the zip archive if not unpacked)
ARRAYS_DIR = ENTROPY_ARRAYS_DIR
ARRAYS_ZIP = ARCHIVES_DIR / "entropy_logP_distance_arrays.zip"

L = 96                     # Sequence length (Chorismate Mutase alignment)
DECILE = 0.1               # Fraction of lowest-energy sequences kept
BATCH_SIZE = 50_000        # Sequences per batch for energies and distances

# Divergence bins, their midpoints and the three experimentally tested bins
INTERVALS = [(0.2, 0.25), (0.25, 0.3), (0.3, 0.35), (0.35, 0.4),
             (0.4, 0.45), (0.45, 0.5), (0.5, 0.55), (0.55, 0.6), (0.6, 0.65)]
X_PLOT = [0.225, 0.275, 0.325, 0.375, 0.425, 0.475, 0.525, 0.575, 0.625]
TESTED_BINS = [0, 4, 8]    # Indices of [0.20-0.25], [0.40-0.45], [0.60-0.65]
X_STAR = [0.24, 0.44, 0.64]

# Entropy of the natural MSA under each model (origin of the dashed bridge)
S_MSA = {"meDCA": 0.494, "eaDCA": 1.533, "bmDCA": 0.922, "edDCA": 0.809}

# Experimental functional rates in the three tested bins
F_RATE = {
    "meDCA": [0.309, 0.0520, 0.004],
    "edDCA": [0.239, 0.052, 0.0],
    "eaDCA": [0.318, 0.126, 0.02],
    "bmDCA": [0.286, 0.119, 0.013],
}

device = get_device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
tokens = get_tokens("protein")

# ===============================================================================
# PLOT STYLE (same as entropy_and_energy_landscape.py)
# ===============================================================================

USE_TEX = all(shutil.which(p) is not None for p in ("latex", "dvipng", "kpsewhich"))
plt.rcParams.update({
    "text.usetex": USE_TEX,
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "axes.titleweight": "normal",
})
TEXT_COLOR = "#2F2F2F"
palette = sns.color_palette("deep", 4)
colors = {"meDCA": palette[0], "edDCA": palette[1], "bmDCA": palette[2], "eaDCA": palette[3]}


# ===============================================================================
# DATA LOADING
# ===============================================================================

def load_array(name: str) -> np.ndarray:
    """Load a pre-computed .npy array from the unpacked folder or from the zip."""
    path = ARRAYS_DIR / f"{name}.npy"
    if path.is_file():
        return np.load(path)
    with zipfile.ZipFile(ARRAYS_ZIP) as archive:
        return np.load(io.BytesIO(archive.read(f"entropy_logP_distance_arrays/{name}.npy")))


def load_onehot_fasta(path: Path) -> torch.Tensor:
    """Read an aligned FASTA file and return integer-encoded sequences (N, L)."""
    _, sequences = import_from_fasta(str(path), tokens=tokens, filter_sequences=True, remove_duplicates=False)
    return torch.as_tensor(np.asarray(sequences), dtype=torch.int64)


# ===============================================================================
# ENERGIES AND DISTANCES OF THE LARGE SAMPLES
# ===============================================================================

def compute_min_distance_batch(a1: torch.Tensor, a2: torch.Tensor, chunk_n: int = BATCH_SIZE) -> torch.Tensor:
    """
    Minimum Hamming distance between each sequence in `a1` and all sequences in `a2`.
    (Same function as in entropy_and_energy_landscape.py, with the reference matrix
    built only once.)

    Args:
        a1: Tensor (N, L) of integer-encoded query sequences
        a2: Tensor (M, L, C) of one-hot encoded reference sequences
    Returns:
        Tensor (N,) with the distance to the closest reference sequence
    """
    N, L_seq = a1.shape
    M, _, C = a2.shape
    a2f_t = a2.reshape(M, L_seq * C).transpose(0, 1).contiguous()
    min_dist = torch.empty(N, dtype=torch.int32)

    for n in range(0, N, chunk_n):
        batch = one_hot(a1[n:n + chunk_n].to(device), num_classes=C, dtype=a2.dtype)
        matches = batch.reshape(batch.shape[0], L_seq * C) @ a2f_t   # Matching positions
        min_dist[n:n + chunk_n] = (L_seq - matches).min(dim=1).values.round().to(torch.int32).cpu()
    return min_dist


def compute_energies(sequences: torch.Tensor, params: dict, q: int) -> np.ndarray:
    """Energy E(s) of every sequence under the given DCA model, computed in batches."""
    energies = []
    with torch.no_grad():
        for batch in sequences.split(BATCH_SIZE):
            batch_oh = one_hot(batch.to(device), num_classes=q, dtype=dtype)
            energies.append(compute_energy(batch_oh, params).cpu())
    return torch.cat(energies).numpy().astype(np.float64)


def get_sample_energies_and_distances() -> dict:
    """
    For each model, return the energy (under that model) and the divergence
    from the closest natural sequence of every sequence in its large sample.
    Results are cached on disk because this is the only expensive step.
    """
    if CACHE_FILE.is_file():
        print(f"Loading cached energies/distances from {CACHE_FILE.name}")
        cache = np.load(CACHE_FILE)
        return {m: (cache[f"E_{m}"], cache[f"div_{m}"]) for m in MODELS}

    q = len(tokens)
    msa = one_hot(load_onehot_fasta(MSA_FILE).to(device), num_classes=q, dtype=dtype)
    print(f"Natural MSA: {msa.shape[0]} sequences")

    results, to_save = {}, {}
    for model in MODELS:
        print(f"{model}: loading sample and parameters...")
        sample = load_onehot_fasta(SAMPLE_FILES[model])
        params = load_params(str(PARAM_FILES[model]), tokens=tokens, device=device, dtype=dtype)

        energies = compute_energies(sample, params, q)
        divergence = compute_min_distance_batch(sample, msa).numpy() / L
        print(f"{model}: {len(energies)} sequences, <E> = {energies.mean():.3f}")

        results[model] = (energies, divergence)
        to_save[f"E_{model}"], to_save[f"div_{model}"] = energies, divergence

    np.savez_compressed(CACHE_FILE, **to_save)
    return results


# ===============================================================================
# ENTROPY CORRECTIONS
# ===============================================================================

def decile_correction(energies: np.ndarray, fraction: float = DECILE) -> float:
    """
    Entropy change obtained by restricting a Boltzmann distribution, sampled by
    `energies`, to its lowest-energy fraction `fraction`:

        Delta = -(<E> - <E>_low) + log(fraction)      (always negative)

    so that S_low = S + Delta.
    """
    threshold = np.quantile(energies, fraction)
    low_energies = energies[energies <= threshold]
    return -(energies.mean() - low_energies.mean()) + np.log(fraction)


def bin_masks(divergence: np.ndarray) -> list:
    """Boolean masks selecting the sample sequences that fall in each divergence bin."""
    return [(divergence >= lo) & (divergence < hi) for lo, hi in INTERVALS]


def compute_corrected_entropies(samples: dict) -> dict:
    """Compute S_10, per-bin decile-corrected entropies and the from-scratch check."""
    results = {}
    for model in MODELS:
        energies, divergence = samples[model]
        S_bin_trusted = load_array(f"S_{model}_sample_interval")   # Notebook values

        # Step 1: global lowest-energy decile
        S_10 = S_TOT[model] + decile_correction(energies)

        # Step 2: decile correction inside each divergence bin
        delta_bin, S_bin_scratch, n_bin = [], [], []
        for mask in bin_masks(divergence):
            E_bin = energies[mask]
            n_bin.append(int(mask.sum()))
            delta_bin.append(decile_correction(E_bin))
            # Consistency check: S_B = S_tot - (<E> - <E>_B) + log P(B)
            S_bin_scratch.append(S_TOT[model] - (energies.mean() - E_bin.mean()) + np.log(mask.mean()))

        delta_bin = np.array(delta_bin)
        results[model] = {
            "S_tot": S_TOT[model],
            "S_10": S_10,
            "mean_E": energies.mean(),
            "mean_E_10": energies[energies <= np.quantile(energies, DECILE)].mean(),
            "n_bin": n_bin,
            "S_bin": S_bin_trusted,
            "delta_bin": delta_bin,
            "S_bin_10": S_bin_trusted + delta_bin,
            "S_bin_scratch": np.array(S_bin_scratch),
            "S_bin_scratch_10": np.array(S_bin_scratch) + delta_bin,
        }
    return results


# ===============================================================================
# FIGURE 1: S_tot vs S_10 BARPLOT
# ===============================================================================

def style_axes(ax):
    """Grid, ticks and spines as in the published figures."""
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=1)
    ax.tick_params(axis="both", which="major", labelsize=26, direction="in", length=8, width=1.5)
    ax.tick_params(axis="both", which="minor", labelsize=14, direction="in", length=4, width=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color(TEXT_COLOR)


def plot_entropy_barplot(results: dict):
    """Grouped bars: total entropy (light) and lowest-decile entropy (dark) per model."""
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(MODELS))
    width = 0.38

    for i, model in enumerate(MODELS):
        S_tot, S_10 = results[model]["S_tot"], results[model]["S_10"]
        ax.bar(x[i] - width / 2, S_tot, width, color=colors[model], alpha=0.45, edgecolor="black", linewidth=1)
        ax.bar(x[i] + width / 2, S_10, width, color=colors[model], alpha=0.95, edgecolor="black", linewidth=1,
               hatch="//")
        # Label: entropy value and order of magnitude of exp(S), i.e. 10^round(S / ln 10)
        for x_bar, S in ((x[i] - width / 2, S_tot), (x[i] + width / 2, S_10)):
            decade = int(round(S / np.log(10)))
            ax.text(x_bar, S + 1.5, f"{S:.1f}\n" + rf"$\sim 10^{{{decade}}}$",
                    ha="center", va="bottom", fontsize=18, color=TEXT_COLOR, linespacing=1.3)

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=28)
    ax.set_ylabel("Entropy", fontsize=32, labelpad=10, color=TEXT_COLOR)
    ax.set_ylim(0, 1.3 * max(r["S_tot"] for r in results.values()))  # Leave room for the legend
    style_axes(ax)
    ax.tick_params(axis="x", length=0)

    # Right axis: effective number of sequences exp(S). Since S is linear on the
    # left axis, exp(S) on a log scale with limits exp(ylim) lines up exactly.
    ax_exp = ax.twinx()
    ax_exp.set_yscale("log")
    ax_exp.set_ylim(np.exp(ax.get_ylim()))
    max_decade = int(np.log10(ax_exp.get_ylim()[1]))
    ax_exp.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(10.0 ** np.arange(0, max_decade + 1, 10)))
    ax_exp.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterMathtext())
    ax_exp.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax_exp.set_ylabel(r"$e^{S}$ (Effective Number of Sequences)", fontsize=28, labelpad=10, color=TEXT_COLOR)
    ax_exp.tick_params(axis="y", which="major", labelsize=22, direction="in", length=8, width=1.5)

    legend_handles = [
        Patch(facecolor="gray", alpha=0.45, edgecolor="black", label=r"$S_{\mathrm{tot}}$"),
        Patch(facecolor="gray", alpha=0.95, edgecolor="black", hatch="//", label=r"$S_{10}$ (lowest-energy decile)"),
    ]
    ax.legend(handles=legend_handles, fontsize=24, frameon=False, loc="upper center", ncol=2)

    fig.tight_layout()
    fig.savefig(FIGURES_INTERMEDIATE / "entropy_total_vs_decile_barplot.pdf")
    fig.savefig(FIGURES_INTERMEDIATE / "entropy_total_vs_decile_barplot.png", dpi=200)
    plt.close(fig)


# ===============================================================================
# FIGURE 2: COMBINED FIGURE (energy grid + corrected effective sequences)
# ===============================================================================

def panel_to_figure(bounds, panel_bounds):
    """Map axes bounds given inside a panel to figure coordinates."""
    x, y, w, h = bounds
    px, py, pw, ph = panel_bounds
    return [px + x * pw, py + y * ph, w * pw, h * ph]


def plot_effective_sequences(ax, S_interval_by_model: dict, ylabel: str):
    """
    Right panel of the combined figure, drawn with the given per-bin entropies:
    circles = exp(S_B), stars = exp(S_B) * functional rate in the tested bins.
    """
    for model in MODELS:
        S_interval = S_interval_by_model[model]
        # Dashed bridge from the natural MSA (x = 0) to the first divergence bin
        ax.plot([0, X_PLOT[0]], np.exp([S_MSA[model], S_interval[0]]),
                linestyle="--", c=colors[model], linewidth=1, alpha=0.7)
        ax.plot(X_PLOT, np.exp(S_interval), marker="o", markersize=15, linewidth=3, c=colors[model],
                label=model, markeredgewidth=1, markeredgecolor="black", linestyle="--")

        # Number of functional sequences in the three tested bins
        y_star = np.exp(S_interval[TESTED_BINS]) * np.array(F_RATE[model])
        ax.scatter(X_STAR, y_star, marker="*", s=450, c=[colors[model]],
                   edgecolors="black", linewidth=1, zorder=10)

    ax.set_xlabel("Sequence Divergence from Closest Natural", fontsize=32, labelpad=10, color=TEXT_COLOR)
    ax.set_ylabel(ylabel, fontsize=32, labelpad=10, color=TEXT_COLOR)
    for lo, hi in (INTERVALS[i] for i in TESTED_BINS):
        ax.axvspan(lo, hi, color="gray", alpha=0.15, zorder=0)
    ax.set_yscale("log")
    legend = ax.legend(prop={"size": 28}, frameon=False, loc="upper left",
                       bbox_to_anchor=(0.02, 1.035), borderaxespad=0.0, borderpad=1, handlelength=2.5)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    style_axes(ax)


def plot_energy_grid(fig, left_panel):
    """
    Left panel of the combined figure: 2x2 grid of log-probability histograms of
    eaDCA and meDCA designs, scored by eaDCA (left) and meDCA (right), in the
    [0.20-0.25] (top) and [0.40-0.45] (bottom) divergence bins.
    """
    mask_titles = {1: "[0.2–0.25]", 2: "[0.4–0.45]"}
    scorers = ["eaDCA", "meDCA"]
    left_margin, right_margin, bottom_margin, top_margin = 0.08, 1.0, 0.08, 0.93
    col_width = (right_margin - left_margin) / 2
    row_height = (top_margin - bottom_margin) / 2

    axes, x_min, x_max = [], np.inf, -np.inf
    for row_idx, mask_type in enumerate([1, 2]):
        for col_idx, scorer in enumerate(scorers):
            panel_x = left_margin + col_idx * col_width
            panel_y = bottom_margin + (1 - row_idx) * row_height
            ax = fig.add_axes(panel_to_figure([panel_x, panel_y, col_width, row_height], left_panel))
            axes.append(ax)

            # Sequences generated by eaDCA and meDCA, both scored by `scorer`
            ene_eaDCA = load_array(f"ene_eaDCA_{scorer}_{mask_type}")
            ene_meDCA = load_array(f"ene_meDCA_{scorer}_{mask_type}")
            x_min = min(x_min, ene_eaDCA.min(), ene_meDCA.min())
            x_max = max(x_max, ene_eaDCA.max(), ene_meDCA.max())

            for ene, model in ((ene_eaDCA, "eaDCA"), (ene_meDCA, "meDCA")):
                ax.hist(ene, bins=30, alpha=0.5, color=colors[model], density=True)
                ax.hist(ene, bins=30, histtype="step", color="black", density=True, linewidth=1)

            if row_idx == 1:
                ax.set_xlabel(rf"$\log(P_{{\mathrm{{{scorer}}}}})$", labelpad=8, fontsize=32, color=TEXT_COLOR)
            if col_idx == 0:
                ax.text(-0.2, 0.5, mask_titles[mask_type], fontsize=32, color=TEXT_COLOR, rotation=90,
                        va="center", ha="center", transform=ax.transAxes)
            style_axes(ax)
            if col_idx == 1:
                ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    padding = 0.08 * (x_max - x_min)
    for ax in axes:
        ax.set_xlim(x_min - padding, x_max + padding)

    legend_handles = [Patch(facecolor=colors[m], edgecolor="black", linewidth=1, alpha=0.5, label=m)
                      for m in ("eaDCA", "meDCA")]
    legend = fig.legend(handles=legend_handles, loc="upper right", fontsize=28, frameon=False,
                        bbox_to_anchor=(left_panel[0] + 0.4 * left_panel[2],
                                        bottom_margin + 1.035 * (top_margin - bottom_margin)),
                        borderpad=1, handlelength=2.5)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    return right_margin, bottom_margin, top_margin


def plot_combined_figure(S_interval_by_model: dict, filename: str, ylabel: str):
    """Reproduce the final combined figure (Fig. 2) with the given entropies."""
    fig = plt.figure(figsize=(24, 7))
    left_panel = [0.0, 0.0, 10.5 / 22.5, 1.0]
    right_margin, bottom_margin, top_margin = plot_energy_grid(fig, left_panel)

    right_axes_left = left_panel[0] + right_margin * left_panel[2] + 0.075
    ax_right = fig.add_axes([right_axes_left, bottom_margin, 0.965 - right_axes_left, top_margin - bottom_margin])
    plot_effective_sequences(ax_right, S_interval_by_model, ylabel)

    fig.savefig(FIGURES_INTERMEDIATE / f"{filename}.pdf", bbox_inches="tight", pad_inches=0.2)
    fig.savefig(FIGURES_INTERMEDIATE / f"{filename}.png", bbox_inches="tight", pad_inches=0.2, dpi=150)
    plt.close(fig)


# ===============================================================================
# MAIN
# ===============================================================================

def print_and_save_summary(results: dict):
    """Print the numbers used in the figures and save them to a JSON file."""
    print("\nGlobal lowest-energy decile")
    print(f"{'model':<7}{'S_tot':>9}{'<E>':>10}{'<E>_10':>10}{'S_10':>9}")
    for m in MODELS:
        r = results[m]
        print(f"{m:<7}{r['S_tot']:>9.2f}{r['mean_E']:>10.2f}{r['mean_E_10']:>10.2f}{r['S_10']:>9.2f}")

    print("\nPer-bin entropies: trusted S_B | from-scratch S_B | decile shift | corrected S_B,10")
    for m in MODELS:
        r = results[m]
        print(f"\n{m}")
        for i, (lo, hi) in enumerate(INTERVALS):
            print(f"  [{lo:.2f}-{hi:.2f}) n={r['n_bin'][i]:>7d}  S_B={r['S_bin'][i]:7.2f}  "
                  f"scratch={r['S_bin_scratch'][i]:7.2f}  delta={r['delta_bin'][i]:6.2f}  "
                  f"S_B,10={r['S_bin_10'][i]:7.2f}")

    serializable = {m: {k: (v.tolist() if isinstance(v, np.ndarray) else float(v) if np.isscalar(v) and not isinstance(v, int) else v)
                        for k, v in r.items()} for m, r in results.items()}
    with open(RESULTS_DIR / "entropy_correction_results.json", "w") as f:
        json.dump(serializable, f, indent=2)


def main():
    samples = get_sample_energies_and_distances()
    results = compute_corrected_entropies(samples)
    print_and_save_summary(results)

    plot_entropy_barplot(results)
    # Main result: trusted bin entropies corrected to the lowest-energy decile
    plot_combined_figure({m: results[m]["S_bin_10"] for m in MODELS},
                         "effective_sequences_decile_corrected",
                         "Effective Number of Sequences")
    # Same figure with the from-scratch bin entropies, for comparison
    plot_combined_figure({m: results[m]["S_bin_scratch_10"] for m in MODELS},
                         "effective_sequences_decile_corrected_from_scratch",
                         "Effective Number of Sequences")
    print(f"\nFigures saved in {FIGURES_INTERMEDIATE}")


if __name__ == "__main__":
    main()

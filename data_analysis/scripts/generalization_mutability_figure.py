#!/usr/bin/env python3
"""
Published Fig. 4: generalization and mutability panels side by side.

This script recomputes both panels from the raw data instead of reading the figures
produced by leave_one_out_generalization.py and mutability.py, so the merged figure
is reproducible on its own.

Figures
-------
Fig. 4    figures/main/fig04_generalization_and_mutability.pdf

Inputs
------
data/natural_msa/                     natural MSA and sequence weights
data/leave_one_out/                   excluded sequences, filtered MSAs, retrained models
models/                               meDCA and eaDCA, for the mutability panel

Runtime
-------
Tens of minutes, since it repeats the computation of both panels. GPU when available.

# Figure 2

Standalone notebook to recreate the two violin plots used in Figure 2.
- Produces and saves the combined figure as `figure_2.pdf` (and `figure_2.png`) for the paper.
- Uses precomputed model parameters, MSAs, and overfitting analysis outputs to ensure reproducibility of the panels.
- All plotting functions are configured for publication-ready output (consistent palette, LaTeX text).

Converted from the notebook data_analysis/merge_overfitting_mutability_figures.ipynb; the analysis code is unchanged.
Cell boundaries are kept as `# %%`, so the script can also be run cell by cell.
"""

# %% [markdown]
# ## Shared Setup
#
# Load dependencies, configure the output directory, define the shared color palette, and select CPU/GPU devices for the figure-generation workflow.

# %%
# ============================================================================
# FIGURE 2 - MUTABILITY AND TAKE-ONE-OUT ENERGY COMPARISON
# ============================================================================

import gc
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from adabmDCA.fasta import get_tokens, import_from_fasta
from adabmDCA.functional import one_hot
from adabmDCA.io import load_params
from adabmDCA.statmech import compute_energy
from adabmDCA.utils import get_device, get_dtype


# ============================================================================
# SHARED STYLE AND PATHS
# ============================================================================

from paths import (FIGURES_MAIN, FILTERED_MSA_DIR, MODELS_DIR, NATURAL_MSA,
                   NATURAL_WEIGHTS, TAKE_1OUT_MODELS_DIR)

text_gray = "0.35"
plt.rcParams.update({
    "text.usetex": True,
})

palette = sns.color_palette("deep", 4)
colors = {
    "meDCA": palette[0],
    "edDCA": palette[1],
    "bmDCA": palette[2],
    "eaDCA": palette[3],
}

device_cpu = get_device("cpu")
device_accel = get_device("cuda" if torch.cuda.is_available() else "cpu")
dtype = get_dtype("float32")

print(f"Output directory: {FIGURES_MAIN}")
print(f"Acceleration device: {device_accel}")

# %% [markdown]
# ## Mutability Panel Data
#
# Compute CDE values for sampled natural sequences and load the leave-one-out energy distributions that will be combined into the final two-panel Figure 2.

# %%


# ============================================================================
# MUTABILITY PANEL DATA
# ============================================================================

def compute_CDE_full_batch_fast(sequences, model, device=device_accel, dtype=dtype):
    with torch.no_grad():
        batch_size, L, q = sequences.shape
        CDE_full_batch = torch.zeros((batch_size, L), device=device, dtype=dtype)

        chunk_size = 150
        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            sequences_chunk = sequences[start:end]
            chunk_batch_size = sequences_chunk.shape[0]

            model_device = {
                "bias": model["bias"].to(device),
                "coupling_matrix": model["coupling_matrix"].to(device),
            }

            sequence_indices = sequences_chunk.argmax(-1)
            base = sequence_indices.unsqueeze(1).expand(-1, L * q, -1).reshape(chunk_batch_size * L * q, L).clone().to(device)

            positions = torch.arange(L, device=device).repeat_interleave(q).repeat(chunk_batch_size)
            values = torch.arange(q, device=device, dtype=torch.int64).repeat(L).repeat(chunk_batch_size)
            base[torch.arange(chunk_batch_size * L * q, device=device), positions] = values

            dms_oh = one_hot(base, num_classes=q).to(dtype)
            energy_dms = compute_energy(dms_oh, model_device).view(chunk_batch_size, L, q)
            energy_dms -= energy_dms.min(dim=2, keepdim=True)[0]

            exp_energy_dms = torch.exp(-energy_dms)
            p_i = exp_energy_dms / exp_energy_dms.sum(dim=2, keepdim=True)
            CDE_chunk = -torch.sum(p_i * torch.log2(p_i + 1e-10), dim=2)
            CDE_full_batch[start:end] = CDE_chunk

            del model_device, base, positions, values, dms_oh, energy_dms, exp_energy_dms, p_i, CDE_chunk

        del sequences
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return CDE_full_batch.cpu()

mut_tokens = get_tokens("ACDEFGHIKLMNPQRSTVWY-")
mut_data_path = NATURAL_MSA
mut_weights_path = NATURAL_WEIGHTS
mut_model_paths = {
    "meDCA": MODELS_DIR / "Parameters_edDCA_density=0.125_high_entropy.dat",
    "eaDCA": MODELS_DIR / "Parameters_EAALG0.8_new.dat",
}

_, msa_enc_nat = import_from_fasta(str(mut_data_path), tokens=mut_tokens, filter_sequences=True)
msa_enc_nat = torch.tensor(msa_enc_nat, device=device_cpu, dtype=torch.long)
msa_oh_nat = one_hot(msa_enc_nat, num_classes=len(mut_tokens))

weights = torch.from_numpy(np.loadtxt(mut_weights_path)).float()
sampled_indices = torch.multinomial(weights / weights.sum(), num_samples=1000, replacement=True).cpu().numpy()

mut_models = {
    name: load_params(str(path), tokens=mut_tokens, device=device_cpu, dtype=dtype)
    for name, path in mut_model_paths.items()
}

msa_oh_nat_accel = msa_oh_nat.to(device_accel)
CDE_mean_results = {}
for model_name, model in mut_models.items():
    print(f"Computing mutability panel data for {model_name}...")
    CDE_mean_results[model_name] = compute_CDE_full_batch_fast(msa_oh_nat_accel, model).mean(-1)

mutability_order = ["eaDCA", "meDCA"]
sampled_cde_means = [CDE_mean_results[name].cpu().numpy()[sampled_indices] for name in mutability_order]


# ============================================================================
# TAKE-ONE-OUT ENERGY PANEL DATA
# ============================================================================

overfit_tokens = get_tokens("protein")
_, msa_data = import_from_fasta(
    str(NATURAL_MSA),
    tokens=overfit_tokens,
    filter_sequences=True,
)

n = 10
headers2 = [
    "1ECM|A",
    "YP_788889.1",
    "ZP_03546198.1",
    "NP_378270.1",
    "ZP_03716454.1",
    "ZP_07298571.1",
    "YP_872130.1",
    "ZP_03290748.1",
    "YP_001612384.1",
    "YP_003610308.1",
]

msa_paths = [FILTERED_MSA_DIR / f"msa_exclude_{i}.fasta" for i in range(1, n + 1)]
seq_1to_path = FILTERED_MSA_DIR / "selected_sequences.fasta"
model_paths = [TAKE_1OUT_MODELS_DIR / f"model_{i}" / "params.dat" for i in range(1, n + 1)]
models_hs_paths = [TAKE_1OUT_MODELS_DIR / f"model_{i}" / "params_density_0.125.dat" for i in range(1, n + 1)]

msa_t1o = {
    i: import_from_fasta(str(msa_paths[i]), tokens=overfit_tokens, filter_sequences=True)[1]
    for i in range(n)
}
_, seqs = import_from_fasta(str(seq_1to_path), tokens=overfit_tokens, filter_sequences=True)
seq_1to = {i: seqs[i] for i in range(n)}

models_t1o = {
    i: load_params(str(model_paths[i]), device=device_accel, tokens=overfit_tokens, dtype=dtype)
    for i in range(n)
}
models_hs = {
    i: load_params(str(models_hs_paths[i]), device=device_accel, tokens=overfit_tokens, dtype=dtype)
    for i in range(n)
}

ene_seqs_t1o = []
ene_seqs_hs = []
ene_msa_t1o = []
ene_msa_hs = []

for key, seq in seq_1to.items():
    # Current adabmDCA expects one-hot tensors of shape (N, L, q): the single excluded
    # sequence is therefore kept as a batch of one instead of being flattened
    seq_oh = one_hot(torch.tensor(seq).unsqueeze(0), num_classes=len(overfit_tokens)).to(device_accel)
    msa_oh = one_hot(torch.tensor(msa_t1o[key]), num_classes=len(overfit_tokens)).to(device_accel)

    ene_seq_t1o = compute_energy(seq_oh, models_t1o[key])
    ene_seq_hs = compute_energy(seq_oh, models_hs[key])
    ene_msa__t1o = compute_energy(msa_oh, models_t1o[key])
    ene_msa__hs = compute_energy(msa_oh, models_hs[key])

    ene_seqs_t1o.append(ene_seq_t1o.item())
    ene_seqs_hs.append(ene_seq_hs.item())
    ene_msa_t1o.append(ene_msa__t1o.detach().cpu())
    ene_msa_hs.append(ene_msa__hs.detach().cpu())

data_t1o = [values.numpy() - values.numpy().mean() for values in ene_msa_t1o]
data_hs = [values.numpy() - values.numpy().mean() for values in ene_msa_hs]
excluded_t1o = [ene_seqs_t1o[i] - ene_msa_t1o[i].numpy().mean() for i in range(n)]
excluded_hs = [ene_seqs_hs[i] - ene_msa_hs[i].numpy().mean() for i in range(n)]

# %% [markdown]
# ## Plotting Helpers and Figure Export
#
# Define the two violin-plot panels, assemble the final layout, and save the combined figure in the `images/` directory.

# %%

# ============================================================================
# PLOTTING HELPERS
# ============================================================================

def plot_mutability_violin(ax):
    sns.violinplot(
        data=sampled_cde_means,
        inner="box",
        palette=[colors[name] for name in mutability_order],
        ax=ax,
    )
    ax.set_xticks(range(len(mutability_order)))
    ax.set_xticklabels(mutability_order, fontsize=26)
    ax.set_xlabel("")
    ax.set_ylabel("Mutability (CDE)", fontsize=32)
    ax.tick_params(axis="y", labelsize=26)
    ax.tick_params(axis="x", labelsize=32)
    ax.set_title("")
    ax.grid(True, alpha=0.4)


def plot_take1out_violin(ax, legend_fontsize=18):
    positions = np.arange(1, n + 1)

    vp1 = ax.violinplot(data_t1o, positions=positions, showmeans=False, widths=0.35)
    vp2 = ax.violinplot(data_hs, positions=positions + 0.2, showmeans=False, widths=0.35)

    for pc in vp1["bodies"]:
        pc.set_facecolor(colors["bmDCA"])
        pc.set_alpha(0.6)
        pc.set_edgecolor("black")
    for partname in ("cbars", "cmins", "cmaxes"):
        if partname in vp1:
            vp1[partname].set_edgecolor("black")
            vp1[partname].set_linewidth(1)

    for pc in vp2["bodies"]:
        pc.set_facecolor(colors["meDCA"])
        pc.set_alpha(0.6)
        pc.set_edgecolor("black")
    for partname in ("cbars", "cmins", "cmaxes"):
        if partname in vp2:
            vp2[partname].set_edgecolor("black")
            vp2[partname].set_linewidth(1)

    ax.plot(
        positions,
        excluded_t1o,
        c=colors["bmDCA"],
        marker="o",
        linestyle="None",
        markersize=10,
        markeredgecolor="black",
        label="Excluded seq (bmDCA)",
    )
    ax.plot(
        positions + 0.2,
        excluded_hs,
        c=colors["meDCA"],
        marker="o",
        linestyle="None",
        markersize=10,
        markeredgecolor="black",
        label="Excluded seq (meDCA)",
    )

    legend_elements = [
        Patch(facecolor=colors["bmDCA"], alpha=1, label="MSA distribution (bmDCA)"),
        Patch(facecolor=colors["meDCA"], alpha=1, label="MSA distribution (meDCA)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["bmDCA"], markersize=10, markeredgecolor="black", label="Excluded seq (bmDCA)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["meDCA"], markersize=10, markeredgecolor="black", label="Excluded seq (meDCA)"),
    ]

    ax.legend(handles=legend_elements, fontsize=legend_fontsize, frameon=False) #, framealpha=0.9
    ax.set_xticks(positions)
    ax.set_xticklabels(headers2, rotation=30, ha="right", fontsize=26)
    ax.tick_params(axis="y", labelsize=26)
    ax.set_xlabel("Excluded Sequence", fontsize=32)
    ax.set_ylabel(r"$E-\langle E\rangle_{\mathrm{MSA}}$", fontsize=32)
    ax.grid(True, alpha=0.4)


# ============================================================================
# SAVE FIGURE 2 WITH TWO SUBPLOTS
# ============================================================================

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(24, 8),
    gridspec_kw={"width_ratios": [1.3, 1.0]},
)
plot_take1out_violin(ax1, legend_fontsize=21.1)
plot_mutability_violin(ax2)

fig.tight_layout()

# Explicit layout knobs you can tune quickly.
SUBPLOT_WSPACE = 0.150
LEFT_XLABEL_Y_AXES = -0.3
RIGHT_AX_WIDTH_SCALE = 1.00
RIGHT_AX_HEIGHT_SCALE = 1.28
RIGHT_AX_X_SHIFT = 0.00
RIGHT_AX_Y_SHIFT = -0.14
RIGHT_AX_MATCH_TOP_WITH_LEFT = True

plot_take1out_violin(ax1, legend_fontsize=21.1)
plot_mutability_violin(ax2)

fig.tight_layout()

# Explicit horizontal gap between the two subplots.
fig.subplots_adjust(wspace=SUBPLOT_WSPACE)

pos1 = ax1.get_position()
pos2 = ax2.get_position()

new_x0 = pos2.x0 + RIGHT_AX_X_SHIFT
new_w = pos2.width * RIGHT_AX_WIDTH_SCALE
new_h = pos2.height * RIGHT_AX_HEIGHT_SCALE

if RIGHT_AX_MATCH_TOP_WITH_LEFT:
    new_y1 = pos1.y1
    new_y0 = new_y1 - new_h
else:
    new_y0 = pos2.y0 + RIGHT_AX_Y_SHIFT

ax2.set_position([new_x0, new_y0, new_w, new_h])

# Explicit vertical position for left x-label.
ax1.xaxis.set_label_coords(0.5, LEFT_XLABEL_Y_AXES)

fig.savefig(FIGURES_MAIN / "fig04_generalization_and_mutability.pdf", dpi=300, bbox_inches="tight")
print("Saved:", FIGURES_MAIN / "fig04_generalization_and_mutability.pdf")
plt.show()

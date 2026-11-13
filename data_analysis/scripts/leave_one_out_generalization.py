#!/usr/bin/env python3
"""
Leave-one-group-out generalization test and decimation entropy across families.

Figures
-------
Fig. S3   figures/supplementary/figS3_leave_one_out_pca.pdf
Fig. S4   figures/supplementary/figS4_excluded_sequence_divergence.pdf
Fig. S5   figures/supplementary/figS5_leave_one_out_energy_shift.png
Fig. S8   figures/supplementary/figS8_entropy_vs_density_all_families.pdf
Fig. 4a   figures/intermediate/leave_one_out_energy_scatter.pdf
          (the published Fig. 4 is assembled by generalization_mutability_figure.py)

Inputs
------
data/natural_msa/                     natural MSA
data/leave_one_out/filtered_msa/      the 10 excluded sequences and their training MSAs
data/leave_one_out/take_1out_models/  the models retrained on each filtered MSA

Runtime
-------
Tens of minutes: 10 leave-one-out models are scored twice (bmDCA and meDCA), plus the
full-data references. A GPU is used when available.

# Leave-One-Out Overfitting Analysis

This script evaluates whether DCA models assign unusually favorable energies to sequences that were present during training. It compares full-data models with leave-one-out models trained after removing selected, mutually distant Chorismate Mutase sequences.

The main outputs are energy-difference summaries, violin plots of MSA energy distributions versus excluded-sequence energies, scatter plots comparing full and leave-one-out scores, and an entropy-decimation comparison across protein families.

Converted from the notebook data_analysis/overfitting_analysis/evaluate_dE_take1out.ipynb; the analysis code is unchanged.
Cell boundaries are kept as `# %%`, so the script can also be run cell by cell.
"""

# %% [markdown]
# ## Imports and Dependencies
#
# Load adabmDCA utilities, numerical libraries, plotting tools, and model I/O functions used throughout the overfitting analysis.

# %%
import adabmDCA
import copy
import time
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# ===============================================================================
# Import adabmDCA utility functions
# ===============================================================================
# These functions provide core functionality for:
# - FASTA file I/O and sequence handling
# - Statistical analysis (frequencies, correlations)
# - One-hot encoding of sequences
# - Parameter and chain initialization
# - MCMC sampling
# - Energy computations
# ===============================================================================

from adabmDCA.fasta import get_tokens, import_from_fasta, compute_weights
from adabmDCA.stats import get_freq_single_point, get_freq_two_points
from adabmDCA.functional import one_hot
from adabmDCA.utils import init_parameters, init_chains, get_device, get_dtype
from adabmDCA.sampling import get_sampler

from adabmDCA.io import load_params, load_chains
# from adabmDCA.resampling import compute_mixing_time, compute_seqID, get_mean_seqid
from adabmDCA.utils import resample_sequences
from adabmDCA.statmech import compute_energy

# ===============================================================================
# Force default black text/axes colors for all plots in this notebook
# ===============================================================================
plt.rcParams.update({
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.titlecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "legend.labelcolor": "black",
})

# ===============================================================================
# Configure computational device and data type
# ===============================================================================
device = get_device("cuda")      # Use GPU for accelerated computation
dtype = get_dtype("float32")     # Use 32-bit floating point precision

# %% [markdown]
# ## Load Chorismate Mutase MSA
#
# Read the complete Chorismate Mutase MSA and prepare the output directory for generated figures.

# %%
# ===============================================================================
# LOAD CHORISMATE MUTASE MSA DATA
# ===============================================================================
# Load the complete Multiple Sequence Alignment (MSA) for Chorismate Mutase.
# - get_tokens: Define the amino acid alphabet (20 amino acids + gap)
# - import_from_fasta: Load sequences from FASTA file
# - filter_sequences: Remove sequences with too many gaps or unusual characters
# ===============================================================================

tokens = get_tokens("protein")  # Standard protein alphabet (20 amino acids + gap)
headers, msa_data = import_from_fasta(
    str(NATURAL_MSA),
    tokens=tokens, 
    filter_sequences=True  # Remove problematic sequences
)

from paths import (FIGURES_INTERMEDIATE, FIGURES_SUPPLEMENTARY, FILTERED_MSA_DIR,
                   NATURAL_MSA, TAKE_1OUT_MODELS_DIR)

# %% [markdown]
# ## Configure Leave-One-Out Paths
#
# Define the ten filtered MSA files, the excluded-sequence FASTA file, the leave-one-out model parameter files, and the corresponding sampled chains.

# %%
# ===============================================================================
# CONFIGURE LEAVE-ONE-OUT CROSS-VALIDATION PATHS
# ===============================================================================
# For overfitting analysis, we train 10 models, each excluding one sequence:
# - MSA paths: FASTA files with one sequence excluded
# - Model paths: Trained model parameters (bmDCA - fully connected)
# - Models HS paths: High-entropy sparse models (meDCA - maximum entropy)
# - Chains paths: MCMC samples from each model
# ===============================================================================

n = 10  # Number of leave-one-out models

# MSA files, each excluding one sequence
msa_paths = [FILTERED_MSA_DIR / f"msa_exclude_{i}.fasta" for i in range(1, n+1)]

# FASTA file containing the 10 excluded sequences
seq_1to_path = FILTERED_MSA_DIR / "selected_sequences.fasta"

# Trained model parameters (bmDCA - fully connected Boltzmann machine)
model_paths = [TAKE_1OUT_MODELS_DIR / f"model_{i}" / "params.dat" for i in range(1, n+1)]

# High-entropy sparse models (meDCA with density=0.125, i.e., 12.5% active parameters)
models_hs_paths = [TAKE_1OUT_MODELS_DIR / f"model_{i}" / "params_density_0.125.dat" for i in range(1, n+1)]

# MCMC chain samples from bmDCA models
chains_1to_paths = [TAKE_1OUT_MODELS_DIR / f"model_{i}" / "chains.fasta" for i in range(1, n+1)]

# MCMC chain samples from meDCA models
chains_hs_paths = [TAKE_1OUT_MODELS_DIR / f"model_{i}" / "chains_density_0.125.fasta" for i in range(1, n+1)]

# %% [markdown]
# ## Load Leave-One-Out Data and Models
#
# Load the filtered MSAs, the ten excluded sequences, and both bmDCA and meDCA leave-one-out parameter sets.

# %%
# ===============================================================================
# LOAD LEAVE-ONE-OUT DATA AND MODELS
# ===============================================================================
# Load:
# 1. MSA datasets (each excluding one sequence)
# 2. The 10 excluded sequences
# 3. Trained models (both bmDCA and meDCA variants)
# ===============================================================================

# Load MSA data for each leave-one-out scenario (dictionary indexed 0-9)
msa_t1o = {i: import_from_fasta(msa_paths[i], tokens=tokens, filter_sequences=True)[1] 
           for i in range(n)}

# Load the excluded sequences
headers, seqs = import_from_fasta(seq_1to_path, tokens=tokens, filter_sequences=True)
seq_1to = {i: seqs[i] for i in range(n)}  # Dictionary of excluded sequences

# Load leave-one-out models (bmDCA - fully connected)
models_t1o = {i: load_params(model_paths[i], device=device, tokens=tokens, dtype=dtype) 
              for i in range(n)}

# Load leave-one-out high-entropy sparse models (meDCA - maximum entropy)
models_hs = {i: load_params(models_hs_paths[i], device=device, tokens=tokens, dtype=dtype) 
             for i in range(n)}

# %% [markdown]
# ## Load Full-Data Reference Models
#
# Load the models trained on the complete MSA. These references are compared against the leave-one-out models to quantify overfitting.

# %%
# ===============================================================================
# LOAD FULL MODELS (TRAINED ON ALL SEQUENCES)
# ===============================================================================
# These are the reference models trained on the complete MSA (all sequences),
# used for comparison with leave-one-out models.
# ===============================================================================

# Full bmDCA model (trained on all sequences)
original_model_path = TAKE_1OUT_MODELS_DIR / "model_all" / "params.dat"
original_model = load_params(original_model_path, device=device, tokens=tokens, dtype=dtype)

# Full meDCA model (high-entropy sparse model with density=0.125)
hs_model_path = TAKE_1OUT_MODELS_DIR / "model_all" / "params_density_0.125.dat"
original_model_hs = load_params(hs_model_path, device=device, tokens=tokens, dtype=dtype)

# %% [markdown]
# ## Sequence Labels
#
# Assign human-readable identifiers to the ten excluded sequences so plots can be interpreted directly.

# %%
# ===============================================================================
# DEFINE SEQUENCE IDENTIFIERS
# ===============================================================================
# Human-readable labels for the 10 excluded sequences (protein database IDs)
# These correspond to the sequences in filtered_msa/selected_sequences.fasta
# ===============================================================================

headers2 = [
    '1ECM|A',              # PDB structure identifier
    'YP_788889.1',         # NCBI protein accession
    'ZP_03546198.1',
    'NP_378270.1',
    'ZP_03716454.1',
    'ZP_07298571.1',
    'YP_872130.1',
    'ZP_03290748.1',
    'YP_001612384.1',
    'YP_003610308.1'
]

# %% [markdown]
# ## Compute Energy Differences
#
# For each excluded sequence, compute its energy relative to the mean energy of the corresponding filtered MSA under leave-one-out and full-data models.

# %%
# ===============================================================================
# COMPUTE ENERGY DIFFERENCES FOR OVERFITTING ANALYSIS
# ===============================================================================
# For each excluded sequence, compute:
# 1. Energy of the excluded sequence under different models
# 2. Mean energy of MSA sequences under those models
# 3. Energy difference ΔE = E(excluded seq) - <E(MSA)>
#
# A large positive ΔE suggests the excluded sequence is an outlier,
# indicating potential overfitting to remaining sequences.
# ===============================================================================

# Initialize lists to store results
delta_es_t1o = []        # ΔE for leave-one-out bmDCA models
delta_es_hs = []         # ΔE for leave-one-out meDCA models
delta_es_orig = []       # ΔE for full bmDCA model
delta_es_orig_hs = []    # ΔE for full meDCA model

ene_seqs_t1o = []        # Energy of excluded sequence (bmDCA leave-one-out)
ene_seqs_hs = []         # Energy of excluded sequence (meDCA leave-one-out)
ene_seqs_orig = []       # Energy of excluded sequence (full bmDCA)
ene_seqs_orig_hs = []    # Energy of excluded sequence (full meDCA)

ene_msa_t1o = []         # MSA energies (bmDCA leave-one-out)
ene_msa_hs = []          # MSA energies (meDCA leave-one-out)
ene_msa_orig = []        # MSA energies (full bmDCA)
ene_msa_orig_hs = []     # MSA energies (full meDCA)

# Loop over each excluded sequence
for key, seq in seq_1to.items():
    # Convert sequences to one-hot encoding
    # Current adabmDCA expects one-hot tensors of shape (N, L, q): the single excluded
    # sequence is therefore kept as a batch of one instead of being flattened
    seq_oh = one_hot(torch.tensor(seq).unsqueeze(0), num_classes=len(tokens)).to(device)
    msa_oh = one_hot(torch.tensor(msa_t1o[key]), num_classes=len(tokens)).to(device)
    original_msa_oh = one_hot(torch.tensor(msa_data), num_classes=len(tokens)).to(device)

    # Compute energy of excluded sequence under different models
    ene_seq_t1o = compute_energy(seq_oh, models_t1o[key])      # Leave-one-out bmDCA
    ene_seq_hs = compute_energy(seq_oh, models_hs[key])        # Leave-one-out meDCA
    ene_seq_orig = compute_energy(seq_oh, original_model)      # Full bmDCA
    ene_seq_orig_hs = compute_energy(seq_oh, original_model_hs) # Full meDCA

    # Store excluded sequence energies
    ene_seqs_t1o.append(ene_seq_t1o.item())
    ene_seqs_hs.append(ene_seq_hs.item())
    ene_seqs_orig.append(ene_seq_orig.item())
    ene_seqs_orig_hs.append(ene_seq_orig_hs.item())

    # Compute MSA energies under different models
    ene_msa__hs = compute_energy(msa_oh, models_hs[key])       # Leave-one-out meDCA
    ene_msa__t1o = compute_energy(msa_oh, models_t1o[key])     # Leave-one-out bmDCA
    ene_msa__orig = compute_energy(msa_oh, original_model)     # Full bmDCA
    ene_msa__orig_hs = compute_energy(msa_oh, original_model_hs) # Full meDCA

    # Store MSA energy distributions
    ene_msa_t1o.append(ene_msa__t1o)
    ene_msa_hs.append(ene_msa__hs)
    ene_msa_orig.append(ene_msa__orig)
    ene_msa_orig_hs.append(ene_msa__orig_hs)

    # Compute energy differences: ΔE = E(excluded) - mean(E(MSA))
    dE_t1o = ene_seq_t1o - ene_msa__t1o.mean().item()
    dE_hs = ene_seq_hs - ene_msa__hs.mean().item()
    dE_orig = ene_seq_orig - ene_msa__orig.mean().item()
    
    # Store energy differences
    delta_es_t1o.append(dE_t1o.item())
    delta_es_hs.append(dE_hs.item())
    delta_es_orig.append(dE_orig.item())
    
    # Print results for each sequence
    print(f"Sequence {key+1}: dE_take1out = {dE_t1o.item():.3f}, dE_hs = {dE_hs.item():.3f}, dE_original = {dE_orig.item():.3f}")

# %% [markdown]
# ## Plot Style
#
# Configure LaTeX-style plotting parameters used by the overfitting figures.

# %%
import seaborn as sns

# ===============================================================================
# CONFIGURE LATEX STYLE FOR PUBLICATION-QUALITY PLOTS
# ===============================================================================
plt.rcParams.update({
    "text.usetex": True,                         # Use LaTeX for text rendering
    # "text.color": "0.35",                      # Softer gray shared by all plot text
    # "axes.labelcolor": "0.35",
    # "axes.titlecolor": "0.35",
    # "xtick.color": "0.35",
    # "ytick.color": "0.35",
    # "legend.labelcolor": "0.35",
    # "font.family": "serif",
    # "font.serif": ["Computer Modern Roman"],     # LaTeX default font
    # "axes.linewidth": 1.2,
    # "xtick.direction": "in",                     # Ticks point inward
    # "ytick.direction": "in",
    # "xtick.top": True,                           # Show ticks on all sides
    # "ytick.right": True,
    # "xtick.major.size": 3.5,
    # "ytick.major.size": 3.5,
    # "xtick.minor.size": 2,
    # "ytick.minor.size": 2,
    # "axes.labelsize": 14,
    # "xtick.labelsize": 14,
    # "ytick.labelsize": 14,
    # "legend.frameon": False
})

# ===============================================================================
# DEFINE CONSISTENT COLOR PALETTE
# ===============================================================================
# Seaborn "deep" palette provides visually distinct, publication-ready colors
# These colors match those used in images_entropy_final.ipynb for consistency
# ===============================================================================
palette = sns.color_palette("deep", 4)
colors = {
    'meDCA': palette[0],    # Cool blue - Maximum Entropy DCA
    'edDCA': palette[1],    # Violet-pink - Edge-Decimated DCA
    'bmDCA': palette[2],    # Pink-orange - Boltzmann Machine DCA
    'eaDCA': palette[3]     # Warm red - Edge Activation DCA
}

# %% [markdown]
# ## PCA Projection of Excluded Sequences
#
# Project the MSA and selected excluded sequences into the first two principal components to verify that the held-out set covers diverse regions of sequence space.

# %%
from sklearn.decomposition import PCA
import seaborn as sns

# ===============================================================================
# PCA CONFIGURATION
# ===============================================================================
plt.rcParams.update({
    "text.usetex": True,
    # "text.color": "0.35",                      # Match labels/titles to tick color
    # "axes.labelcolor": "0.35",
    # "axes.titlecolor": "0.35",
    # "xtick.color": "0.35",
    # "ytick.color": "0.35",
    # "legend.labelcolor": "0.35",
    # "font.family": "serif",
    # "font.serif": ["Computer Modern Roman"],
    # "axes.linewidth": 1.2,
    # "xtick.direction": "in",
    # "ytick.direction": "in",
    # "xtick.top": True,
    # "ytick.right": True,
    # "xtick.major.size": 3.5,
    # "ytick.major.size": 3.5,
    # "xtick.minor.size": 2,
    # "ytick.minor.size": 2,
    # "axes.labelsize": 14,
    # "xtick.labelsize": 14,
    # "ytick.labelsize": 14,
    # "legend.frameon": False
})

# ===============================================================================
# ONE-HOT ENCODE SEQUENCES FOR PCA
# ===============================================================================
# PCA requires numerical input. One-hot encoding converts sequences to binary
# vectors: each position has 21 elements (20 amino acids + gap), with a 1
# indicating the present amino acid and 0s elsewhere.
# ===============================================================================

q = 21              # Number of classes: 20 amino acids + gap
L = msa_data.shape[1]  # Sequence length

# One-hot encode MSA data
msa_oh = one_hot(torch.tensor(msa_data), num_classes=q).cpu().numpy()
msa_oh_flat = msa_oh.reshape(msa_oh.shape[0], -1)  # Flatten to (N_sequences, L*q)

# One-hot encode excluded sequences
seqs_list = list(seq_1to.values())
seqs_array = np.array(seqs_list)
seqs_oh = one_hot(torch.tensor(seqs_array), num_classes=q).cpu().numpy()
seqs_oh_flat = seqs_oh.reshape(seqs_oh.shape[0], -1)  # Flatten to (n, L*q)

# ===============================================================================
# PERFORM PCA ON MSA DATA
# ===============================================================================
# Principal Component Analysis reduces the high-dimensional sequence space
# (L*q dimensions) to 2D for visualization. The first two principal components
# capture the directions of maximum variance in the sequence alignment.
# ===============================================================================

pca = PCA(n_components=2)
msa_pca = pca.fit_transform(msa_oh_flat)  # Project MSA onto first 2 PCs

# Transform excluded sequences using the same PCA (ensures consistent projection)
seqs_pca = pca.transform(seqs_oh_flat)

# ===============================================================================
# VISUALIZE PCA RESULTS
# ===============================================================================
plt.figure(figsize=(14, 10))

# Plot MSA data as gray background (training sequences)
plt.scatter(msa_pca[:, 0], msa_pca[:, 1], c='gray', alpha=0.3, s=80, 
           label='MSA sequences', edgecolors='none')

# Plot excluded sequences with distinct colors and star markers
cm = plt.get_cmap('tab10')  # Colormap with 10 distinct colors
for i in range(n):
    plt.scatter(seqs_pca[i, 0], seqs_pca[i, 1], 
                c=[cm(i)], s=400, marker='*',     # Large star markers
                edgecolors='black', linewidths=2,
                label=headers2[i], zorder=10)     # Draw on top

# Axis labels with variance explained
plt.xlabel('PC1', fontsize=32)
plt.ylabel('PC2', fontsize=32)
plt.title('PCA of MSA and Excluded Sequences', fontsize=36, pad=20)
plt.legend(fontsize=24, loc='best', framealpha=0.9, markerscale=1.5)
plt.grid(True, alpha=0.4)
plt.tick_params(axis='both', which='major', labelsize=28)
plt.tight_layout()

# Save high-resolution figure
plt.savefig(FIGURES_SUPPLEMENTARY / 'figS3_leave_one_out_pca.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Print variance explained by each principal component
print(f"PCA completed. PC1 explains {pca.explained_variance_ratio_[0]*100:.2f}% of variance.")
print(f"PC2 explains {pca.explained_variance_ratio_[1]*100:.2f}% of variance.")

# %% [markdown]
# ## Pairwise Distances Between Excluded Sequences
#
# Compute pairwise normalized Hamming distances among the ten excluded sequences as an additional diversity check.

# %%
# ===============================================================================
# COMPUTE PAIRWISE HAMMING DISTANCES BETWEEN EXCLUDED SEQUENCES
# ===============================================================================
# For each excluded sequence, compute its Hamming distance (fraction of differing
# positions) to all other excluded sequences. This reveals how diverse the
# excluded sequences are from each other.
# ===============================================================================

# Convert seq_1to dictionary to list for easier indexing
seqs_list_calc = [seq_1to[i] for i in range(n)]

# Calculate pairwise Hamming distances
hamming_distributions = []

for i in range(n):
    distances = []
    seq_i = seqs_list_calc[i]
    for j in range(n):
        if i != j:  # Skip self-comparison
            seq_j = seqs_list_calc[j]
            # Calculate Hamming distance (fraction of differing positions)
            hamming_dist = np.sum(seq_i != seq_j) / len(seq_i)  # Normalized by length
            distances.append(hamming_dist)
    hamming_distributions.append(distances)

# ===============================================================================
# VISUALIZE HAMMING DISTANCE DISTRIBUTIONS WITH VIOLIN PLOT
# ===============================================================================
plt.figure(figsize=(12, 8))

# Create violin plot showing distribution of distances for each sequence
vp = plt.violinplot(hamming_distributions, positions=range(1, n+1), 
                    showmeans=True, widths=0.7)

# Color the violin plots
for pc in vp['bodies']:
    pc.set_facecolor(colors['bmDCA'])
    pc.set_alpha(0.7)
    pc.set_edgecolor('black')

# Style the means, mins, maxes, bars
for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
    if partname in vp:
        vp[partname].set_edgecolor('black')
        vp[partname].set_linewidth(1.5)

# Configure axes
plt.xticks(range(1, n+1), [headers2[i] for i in range(n)], 
          rotation=45, ha='right', fontsize=26)
plt.yticks(fontsize=26)
plt.xlabel("Excluded Sequences", fontsize=32)
plt.ylabel("Sequence Divergence", fontsize=32)
plt.ylim(0.75, 0.95)  # Hamming distance ranges from 0 (identical) to 1 (completely different)
plt.grid(True, alpha=0.4, axis='y')
plt.tight_layout()

# Save figure
plt.savefig(FIGURES_SUPPLEMENTARY / "figS4_excluded_sequence_divergence.pdf", dpi=300, bbox_inches='tight')
plt.show()

# ===============================================================================
# PRINT SUMMARY STATISTICS
# ===============================================================================
print("\nHamming Distance Statistics:")
for i in range(n):
    mean_dist = np.mean(hamming_distributions[i])
    std_dist = np.std(hamming_distributions[i])
    print(f"{headers2[i]}: mean = {mean_dist:.3f}, std = {std_dist:.3f}")

# %% [markdown]
# ## Energy Distribution Violin Plot
#
# Compare each filtered MSA energy distribution with the corresponding excluded-sequence energy for bmDCA and meDCA models.

# %%
# ===============================================================================
# VIOLIN PLOT: MSA ENERGY DISTRIBUTIONS vs EXCLUDED SEQUENCE ENERGIES
# ===============================================================================
# This plot shows:
# - Violin plots: Energy distribution of MSA sequences (excluding one)
# - Dots: Energy of the excluded sequence
# For both bmDCA (fully connected) and meDCA (sparse high-entropy) models.
# ===============================================================================

plt.figure(figsize=(10, 7))

# Normalize energies by subtracting the mean (shows relative position in distribution)
data_t1o = [ene_msa_t1o[i].cpu() - ene_msa_t1o[i].cpu().mean() for i in range(n)]
data_hs = [ene_msa_hs[i].cpu() - ene_msa_hs[i].cpu().mean() for i in range(n)]

# ===============================================================================
# CREATE VIOLIN PLOTS FOR MSA ENERGY DISTRIBUTIONS
# ===============================================================================

# bmDCA violin plots (fully connected model)
vp1 = plt.violinplot(data_t1o, showmeans=False, widths=0.35)

# meDCA violin plots (sparse model - offset slightly for visibility)
vp2 = plt.violinplot(data_hs, positions=np.array(range(1, n+1))+0.2, 
                     showmeans=False, widths=0.35)

# ===============================================================================
# STYLE VIOLIN PLOTS
# ===============================================================================

# Color bmDCA violin plots
for pc in vp1['bodies']:
    pc.set_facecolor(colors['bmDCA'])
    pc.set_alpha(0.6)
    pc.set_edgecolor('black')
for partname in ('cbars', 'cmins', 'cmaxes'):
    if partname in vp1:
        vp1[partname].set_edgecolor('black')
        vp1[partname].set_linewidth(1)

# Color meDCA violin plots
for pc in vp2['bodies']:
    pc.set_facecolor(colors['meDCA'])
    pc.set_alpha(0.6)
    pc.set_edgecolor('black')
for partname in ('cbars', 'cmins', 'cmaxes'):
    if partname in vp2:
        vp2[partname].set_edgecolor('black')
        vp2[partname].set_linewidth(1)

# ===============================================================================
# PLOT EXCLUDED SEQUENCE ENERGIES AS DOTS
# ===============================================================================

# bmDCA: Energy of excluded sequence
plt.plot(np.array(range(1, n+1)), 
        [ene_seqs_t1o[i] - ene_msa_t1o[i].cpu().mean() for i in range(n)], 
        c=colors['bmDCA'], marker='o', linestyle='None', markersize=10, 
        markeredgecolor="black", label='Excluded seq (bmDCA)')

# meDCA: Energy of excluded sequence (offset for visibility)
plt.plot(np.array(range(1, n+1))+0.2, 
        [ene_seqs_hs[i] - ene_msa_hs[i].cpu().mean() for i in range(n)], 
        c=colors['meDCA'], marker='o', linestyle='None', markersize=10, 
        markeredgecolor="black", label='Excluded seq (meDCA)')

# ===============================================================================
# CONFIGURE LEGEND AND LAYOUT
# ===============================================================================

# Create custom legend handles for violin plots and dots
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors['bmDCA'], alpha=1, label='MSA distribution (bmDCA)'),
    Patch(facecolor=colors['meDCA'], alpha=1, label='MSA distribution (meDCA)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['bmDCA'], 
               markersize=10, markeredgecolor="black", label='Excluded seq (bmDCA)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['meDCA'], 
               markersize=10, markeredgecolor="black", label='Excluded seq (meDCA)')
]

plt.legend(handles=legend_elements, fontsize=18)
plt.xticks(range(1, n+1), [headers2[i] for i in range(n)], rotation=45, ha='right', fontsize=18)
plt.yticks(fontsize=20)
plt.xlabel("Excluded Sequence", fontsize=24)
plt.ylabel("Energy (normalized by MSA mean)", fontsize=24)
plt.tight_layout()
plt.grid(True, alpha=0.4)

# Save figure
plt.savefig(FIGURES_SUPPLEMENTARY / "figS5_leave_one_out_energy_shift.png", dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Full vs Leave-One-Out Energy Scatter
#
# Plot full-model energies against leave-one-out energies for each filtered MSA, highlighting where the excluded sequence falls relative to the bisector.

# %%
# ===============================================================================
# SCATTER PLOT COMPARISON: FULL MODEL vs LEAVE-ONE-OUT MODEL ENERGIES
# ===============================================================================
# Creates a 2x5 grid of scatter plots showing:
# - X-axis: Energy from full model (trained on all sequences)
# - Y-axis: Energy from leave-one-out model (trained without one sequence)
# - Bisector line: y=x indicates perfect agreement
# - Stars: Excluded sequences (red border = better agreement with bisector)
# ===============================================================================

import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Create 2x5 subplot grid
fig, axes = plt.subplots(2, 5, figsize=(30, 12), sharey=True, sharex=True, 
                         gridspec_kw={'hspace': 0.15, 'wspace': 0.05})
axes = axes.flatten()

for i in range(10):
    ax = axes[i]
    
    # ===============================================================================
    # PLOT MSA SEQUENCES (SCATTER POINTS)
    # ===============================================================================
    
    # meDCA: Scatter plot of MSA energies
    ax.scatter(ene_msa_orig_hs[i].cpu() - ene_msa_orig_hs[i].cpu().mean(),  # Full model
               ene_msa_hs[i].cpu() - ene_msa_hs[i].cpu().mean(),             # Leave-one-out
               color=colors['meDCA'], label='MSA (meDCA)' if i == 0 else '', 
               s=80, alpha=0.5)
    
    # bmDCA: Scatter plot of MSA energies
    ax.scatter(ene_msa_orig[i].cpu() - ene_msa_orig[i].cpu().mean(),        # Full model
               ene_msa_t1o[i].cpu() - ene_msa_t1o[i].cpu().mean(),           # Leave-one-out
               color=colors['bmDCA'], label='MSA (bmDCA)' if i == 0 else '', 
               s=80, alpha=0.5)
    
    # ===============================================================================
    # COMPUTE EXCLUDED SEQUENCE POSITIONS
    # ===============================================================================
    
    # meDCA excluded sequence coordinates
    star_hs_x = ene_seqs_orig_hs[i] - ene_msa_orig_hs[i].cpu().mean()    # Full model
    star_hs_y = ene_seqs_hs[i] - ene_msa_hs[i].cpu().mean()              # Leave-one-out
    
    # bmDCA excluded sequence coordinates
    star_dense_x = ene_seqs_orig[i] - ene_msa_orig[i].cpu().mean()       # Full model
    star_dense_y = ene_seqs_t1o[i] - ene_msa_t1o[i].cpu().mean()         # Leave-one-out
    
    # ===============================================================================
    # HIGHLIGHT BETTER-PERFORMING MODEL
    # ===============================================================================
    # Calculate absolute difference from bisector |x - y|
    # Smaller difference means better agreement between full and leave-one-out
    
    diff_hs = abs(star_hs_x - star_hs_y)          # meDCA difference
    diff_dense = abs(star_dense_x - star_dense_y)  # bmDCA difference
    
    # Plot star with red border for model with smaller difference (better performance)
    if diff_hs < diff_dense:
        # meDCA has smaller difference - highlight with red border
        ax.scatter(star_hs_x, star_hs_y, color=colors['meDCA'], marker='*', 
                  s=700, edgecolors='red', linewidths=3, zorder=10)
        ax.scatter(star_dense_x, star_dense_y, color=colors['bmDCA'], marker='*', 
                  s=700, edgecolors='black', zorder=9)
    else:
        # bmDCA has smaller difference - highlight with red border
        ax.scatter(star_hs_x, star_hs_y, color=colors['meDCA'], marker='*', 
                  s=700, edgecolors='black', zorder=9)
        ax.scatter(star_dense_x, star_dense_y, color=colors['bmDCA'], marker='*', 
                  s=700, edgecolors='red', linewidths=3, zorder=10)
    
    # ===============================================================================
    # PLOT BISECTOR LINE (y = x)
    # ===============================================================================
    # Points on this line have identical energy in full and leave-one-out models
    
    data_x = ene_msa_hs[i].cpu().numpy() - ene_msa_hs[i].cpu().numpy().mean()
    data_y = ene_msa_t1o[i].cpu().numpy() - ene_msa_t1o[i].cpu().numpy().mean()
    min_val = min(data_x.min(), data_y.min())
    max_val = max(data_x.max(), data_y.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', 
           label='Bisector' if i == 0 else '')
    
    # ===============================================================================
    # SUBPLOT STYLING
    # ===============================================================================
    
    ax.grid(True, alpha=0.5)
    ax.set_title(f'{headers2[i]}', fontsize=36)
    ax.tick_params(axis='both', which='major', labelsize=36)
    
    # Create legend only for the first subplot
    if i == 0:
        # Custom legend with normal-sized stars (not giant like in plot)
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['meDCA'], 
                      markersize=25, alpha=0.5, label='Leave-one-out MSA (meDCA)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['bmDCA'], 
                      markersize=25, alpha=0.5, label='Leave-one-out MSA (bmDCA)'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=colors['meDCA'], 
                      markersize=25, markeredgecolor='black', label='Excluded seq (meDCA)'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=colors['bmDCA'], 
                      markersize=25, markeredgecolor='black', label='Excluded seq (bmDCA)'),
            plt.Line2D([0], [0], color='k', linestyle='--', label='Bisector')
        ]
        ax.legend(handles=legend_elements, fontsize=32, loc='upper center', 
                 bbox_to_anchor=(2.4, 1.6), ncol=3)#, framealpha=0.9)

# ===============================================================================
# GLOBAL LABELS AND TITLE
# ===============================================================================

# Centered axis labels for entire figure
fig.text(0.5, 0.0, 'Energy (Model trained on the full MSA)', ha='center', fontsize=40)
fig.text(0.07, 0.5, 'Energy (Model trained on the leave-one-out MSA)', va='center', 
        rotation='vertical', fontsize=40)

# plt.suptitle('Energy Comparison: Full Model (x-axis) vs Leave-One-Out Model (y-axis)', 
            # fontsize=42, y=0.98)

# Save high-resolution figure
plt.savefig(FIGURES_INTERMEDIATE / "leave_one_out_energy_scatter.pdf", dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# # Entropy Dynamics Across Protein Families
#
# Compare entropy trajectories along decimation paths for multiple protein families, including Chorismate Mutase.

# %%
# ===============================================================================
# ENTROPY DYNAMICS ACROSS MULTIPLE PROTEIN FAMILIES
# ===============================================================================
# This analysis compares how entropy evolves along the decimation path (from
# fully-connected bmDCA to sparse meDCA) across different protein families:
# - PF00014: Kunitz/Bovine pancreatic trypsin inhibitor domain
# - PF00076: RNA recognition motif (RRM)
# - PF00595: PDZ domain
# - CM_130530_MC: Chorismate Mutase
# ===============================================================================

# ===============================================================================
# LOAD ENTROPY DATA FOR EACH PROTEIN FAMILY
# ===============================================================================

# === PF00014: Kunitz/BPTI domain ===
pf00014_density = [1.019231, 0.989119, 0.809885, 0.659855, 0.529649, 0.439856, 
                   0.349954, 0.289803, 0.239773, 0.189901, 0.159848, 0.129968, 
                   0.099991, 0.079925, 0.069932, 0.059945, 0.049957, 0.039975]
pf00014_entropy = [78.436981, 78.752670, 78.864395, 79.083534, 79.123726, 78.901810, 
                   79.301865, 79.506935, 79.767586, 79.804550, 79.880264, 79.609695, 
                   79.712456, 79.495506, 79.367867, 78.699272, 77.568344, 75.695862]

# === PF00076: RNA Recognition Motif ===
pf00076_density = [1.014493, 0.989444, 0.809283, 0.659959, 0.529666, 0.439811, 
                   0.349850, 0.289945, 0.239833, 0.189871, 0.159924, 0.129966, 
                   0.099906, 0.079941, 0.069969, 0.059983, 0.049977, 0.039997, 0.029981]
pf00076_entropy = [123.996170, 123.725945, 124.601494, 124.940735, 124.858337, 124.867447, 
                   124.764877, 124.390854, 124.250275, 124.845901, 125.741234, 126.009445, 
                   125.935562, 125.791420, 125.690948, 125.343307, 124.471283, 123.930527, 
                   122.453560]

# === PF00595: PDZ Domain ===
pf00595_density = [1.012346, 0.989324, 0.809970, 0.659839, 0.529544, 0.439685, 
                   0.349722, 0.289812, 0.239939, 0.189929, 0.159957, 0.129970, 
                   0.099978, 0.079974, 0.069982, 0.059978, 0.049951, 0.039992, 
                   0.029976, 0.019986]
pf00595_entropy = [131.952621, 132.116653, 133.266281, 133.847412, 134.444458, 134.412476, 
                   133.929047, 133.383606, 133.111847, 133.840103, 134.533340, 136.119202, 
                   136.329620, 135.678848, 135.458710, 135.235992, 134.456848, 133.567810, 
                   131.706161, 128.206009]

# === CM_130530_MC: Chorismate Mutase ===
cm_density = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.125, 
             0.1, 0.075, 0.05, 0.03]
cm_entropy = [139.41, 137.49, 138.62, 138.19, 139.17, 140.61, 141.19, 142.15, 
             142.24, 146.43, 148.80, 150.13, 148.89, 145.96, 142.48, 137.71]

# ===============================================================================
# NORMALIZE ENTROPY VALUES: ΔS = S - S₀
# ===============================================================================
# Subtract the initial entropy (at density=1, fully-connected model) from each
# series. This shows the absolute change in entropy along the decimation path.
# Positive ΔS indicates entropy increase relative to bmDCA.
# ===============================================================================

pf00014_entropy_norm = np.array(pf00014_entropy) - pf00014_entropy[0]
pf00076_entropy_norm = np.array(pf00076_entropy) - pf00076_entropy[0]
pf00595_entropy_norm = np.array(pf00595_entropy) - pf00595_entropy[0]
cm_entropy_norm = np.array(cm_entropy) - cm_entropy[0]

# ===============================================================================
# PLOT NORMALIZED ENTROPY CURVES
# ===============================================================================

plt.figure(figsize=(10, 7))

# Plot entropy trajectories for each protein family
plt.plot(pf00014_density, pf00014_entropy_norm, marker='o', linestyle='-', 
        linewidth=2, markersize=10, label='PF00014', color=colors['meDCA'])
plt.plot(pf00076_density, pf00076_entropy_norm, marker='s', linestyle='-', 
        linewidth=2, markersize=10, label='PF00076', color=colors['bmDCA'])
plt.plot(pf00595_density, pf00595_entropy_norm, marker='^', linestyle='-', 
        linewidth=2, markersize=10, label='PF00595', color=colors['edDCA'])
plt.plot(cm_density, cm_entropy_norm, marker='d', linestyle='-', 
        linewidth=2, markersize=10, label='CM_130530_MC', color=colors['eaDCA'])

# ===============================================================================
# HIGHLIGHT ENTROPY MAXIMA
# ===============================================================================
# Find and mark the maximum entropy point for each family (optimal sparsity)

max_idx_pf00014 = np.argmax(pf00014_entropy_norm)
max_idx_pf00076 = np.argmax(pf00076_entropy_norm)
max_idx_pf00595 = np.argmax(pf00595_entropy_norm)
max_idx_cm = np.argmax(cm_entropy_norm)

# Plot maximum points with black edge highlighting
plt.plot(pf00014_density[max_idx_pf00014], pf00014_entropy_norm[max_idx_pf00014], 
        marker='o', color=colors['meDCA'], markersize=8, markeredgewidth=2, 
        markeredgecolor='black', linestyle='None', zorder=10)
plt.plot(pf00076_density[max_idx_pf00076], pf00076_entropy_norm[max_idx_pf00076], 
        marker='s', color=colors['bmDCA'], markersize=8, markeredgewidth=2, 
        markeredgecolor='black', linestyle='None', zorder=10)
plt.plot(pf00595_density[max_idx_pf00595], pf00595_entropy_norm[max_idx_pf00595], 
        marker='^', color=colors['edDCA'], markersize=8, markeredgewidth=2, 
        markeredgecolor='black', linestyle='None', zorder=10)
plt.plot(cm_density[max_idx_cm], cm_entropy_norm[max_idx_cm], 
        marker='d', color=colors['eaDCA'], markersize=8, markeredgewidth=2, 
        markeredgecolor='black', linestyle='None', zorder=10)

# ===============================================================================
# CONFIGURE PLOT APPEARANCE
# ===============================================================================

plt.xlabel(r'Fraction of Active Parameters', fontsize=24)
plt.ylabel(r'$\Delta S = S - S_0$', fontsize=24)  # LaTeX formatted label
plt.legend(fontsize=22, loc='best')
plt.grid(True, alpha=0.4)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()

# Save high-resolution figure
plt.savefig(FIGURES_SUPPLEMENTARY / "figS8_entropy_vs_density_all_families.pdf", dpi=300, bbox_inches='tight')
plt.show()

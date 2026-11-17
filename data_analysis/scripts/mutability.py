#!/usr/bin/env python3
"""
Context-dependent entropy (mutability) of the natural sequences.

Figures
-------
Fig. S6   figures/supplementary/figS6_mutability_all_models.pdf
Fig. 4b   figures/intermediate/mutability_violin_two_models.pdf
          (the published Fig. 4 is assembled by generalization_mutability_figure.py)

Inputs
------
data/natural_msa/   natural MSA and sequence weights
../models/tested_models/   the four released DCA models

Runtime
-------
Tens of minutes: the context-dependent entropy is evaluated for every sequence of the
MSA under four models. A GPU is used when available.

# Mutability Analysis

This script computes Conservation Diversity Entropy (CDE) for natural Chorismate Mutase MSA sequences under four DCA model variants. CDE is used here as a model-derived mutability score: higher values indicate broader tolerated amino-acid substitutions at a position.

The output figures compare the CDE distributions across models and provide the mutability panel used in the manuscript.

Converted from the notebook data_analysis/mutability.ipynb; the analysis code is unchanged.
Cell boundaries are kept as `# %%`, so the script can also be run cell by cell.
"""

# %% [markdown]
# ## Imports and Output Directory
#
# Load the scientific Python stack, adabmDCA helpers, and create the directory where generated figures are saved.

# %%
# ==============================================================================
# IMPORTS AND SETUP
# ==============================================================================
# This script computes Conservation Diversity Entropy (CDE) for natural MSA 
# sequences using multiple DCA models and generates violin plots for comparison
# ==============================================================================

import sys
import os
from pathlib import Path
import gc

# Third-party libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# adabmDCA package imports
from adabmDCA.fasta import get_tokens, import_from_fasta
from adabmDCA.utils import get_device, get_dtype
from adabmDCA.functional import one_hot
from adabmDCA.io import load_params
from adabmDCA.statmech import compute_energy

print("All libraries imported successfully!")
from paths import FIGURES_INTERMEDIATE, FIGURES_SUPPLEMENTARY, MODEL_FILES, NATURAL_MSA, NATURAL_WEIGHTS

# %% [markdown]
# ## Plot Style
#
# Define a consistent model color palette and Matplotlib options for publication-quality violin plots.

# %%
# ==============================================================================
# PLOT STYLING CONFIGURATION
# ==============================================================================
# Configure color palette and matplotlib style for publication-quality figures
# ==============================================================================

# Define color palette for each DCA model variant
palette = sns.color_palette("deep", 4)
colors = {
    'meDCA': palette[0],    # Medium entropy DCA - Blue
    'edDCA': palette[1],    # Extreme density DCA - Purple-Pink
    'bmDCA': palette[2],    # Boltzmann machine DCA - Pink-Orange
    'eaDCA': palette[3]     # Extreme attributes DCA - Red
}

# Configure LaTeX-style plotting aesthetics
plt.rcParams.update({
    "text.usetex": True,
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

print("Plot styling configured!")

# %% [markdown]
# ## Runtime Configuration
#
# Select compute devices, data files, model parameter files, and the amino-acid alphabet used to encode the MSA and model parameters.

# %%
# ==============================================================================
# DEVICE, PATHS, AND TOKENS CONFIGURATION
# ==============================================================================

# Configure computational devices
device = get_device("cpu")          # CPU device for general operations
device2 = get_device("cuda")        # GPU device for intensive CDE computation
dtype = get_dtype("float32")         # Data type for tensor operations

# Data paths
data_path = NATURAL_MSA
weights_path = NATURAL_WEIGHTS

# Model paths - different DCA model variants trained on the same MSA
HAmodel_path = MODEL_FILES["meDCA"]
decmodel_path = MODEL_FILES["edDCA"]
eamodel_path = MODEL_FILES["eaDCA"]
densemodel_path = MODEL_FILES["bmDCA"]

# Define amino acid tokens (20 amino acids + gap character)
tokens = get_tokens("ACDEFGHIKLMNPQRSTVWY-")

print("Configuration loaded successfully!")
print(f"Using devices: {device} and {device2}")
print(f"Number of tokens: {len(tokens)}")

# %% [markdown]
# ## Load the Natural MSA
#
# Read the Chorismate Mutase multiple sequence alignment, filter invalid sequences, convert integer encodings to tensors, and build the one-hot representation required by the energy function.

# %%
# ==============================================================================
# LOAD NATURAL MULTIPLE SEQUENCE ALIGNMENT (MSA)
# ==============================================================================
# Import sequences from FASTA file and convert to one-hot encoding
# ==============================================================================

print("Loading natural MSA...")
headers, msa_enc_nat = import_from_fasta(
    data_path, 
    tokens=tokens, 
    filter_sequences=True,  # Remove sequences with invalid characters
    remove_duplicates=False,
)

print(f"MSA shape: {msa_enc_nat.shape}")
print(f"Number of sequences: {len(headers)}")

# Convert to PyTorch tensor
msa_enc_nat = torch.tensor(msa_enc_nat, device=device, dtype=torch.long)

# Convert to one-hot encoding: (num_sequences, L, q)
# where L = sequence length, q = number of tokens
msa_oh_nat = one_hot(msa_enc_nat, num_classes=len(tokens))
print(f"One-hot MSA shape: {msa_oh_nat.shape}")

print("Natural MSA loaded successfully!")

# %% [markdown]
# ## Sequence Weights and Sampling
#
# Load precomputed sequence weights and sample representative MSA indices according to those weights. The sampled indices are used only for plotting, reducing visual clutter while preserving the weighted MSA distribution.

# %%
# ==============================================================================
# LOAD SEQUENCE WEIGHTS AND SAMPLE INDICES
# ==============================================================================
# Load precomputed sequence weights and sample indices for analysis
# Sampling reduces computational burden while maintaining statistical properties
# ==============================================================================

# Load sequence weights (typically computed based on sequence identity)
weights = torch.from_numpy(np.loadtxt(weights_path)).float()
print(f"Weights shape: {weights.shape}")

# Sample indices according to the weight distribution
# This ensures that representative sequences are selected for CDE analysis
num_samples = 1000
probs = weights / weights.sum()  # Normalize to probability distribution
sampled_indices = torch.multinomial(probs, num_samples=num_samples, replacement=True)

print(f"Sampled {num_samples} indices according to sequence weights")
print(f"Sampled indices shape: {sampled_indices.shape}")

# %% [markdown]
# ## Load DCA Models
#
# Load the four trained DCA parameter sets evaluated in the mutability comparison: meDCA, edDCA, eaDCA, and bmDCA.

# %%
# ==============================================================================
# LOAD DCA MODELS
# ==============================================================================
# Load four different DCA model variants for comparison:
# - meDCA: Medium entropy DCA model
# - edDCA: Extreme density DCA model  
# - bmDCA: Boltzmann machine DCA model
# - eaDCA: Extreme attributes DCA model
# ==============================================================================

print("Loading DCA models...")

original_tokens = get_tokens("ACDEFGHIKLMNPQRSTVWY-")

# Load all four model variants
HA_model = load_params(HAmodel_path, tokens=original_tokens, device=device, dtype=dtype)
dec_model = load_params(decmodel_path, tokens=original_tokens, device=device, dtype=dtype)
ea_model = load_params(eamodel_path, tokens=original_tokens, device=device, dtype=dtype)
dense_model = load_params(densemodel_path, tokens=original_tokens, device=device, dtype=dtype)

print("All models loaded successfully!")
print(f"  - meDCA (HA) model")
print(f"  - edDCA (dec) model")
print(f"  - eaDCA model")
print(f"  - bmDCA (dense) model")

# %% [markdown]
# ## CDE Computation
#
# Define the batched CDE routine. For each input sequence, the function enumerates all single-site substitutions, scores them with a DCA model, converts energies into a Boltzmann distribution, and reports the entropy at each position.

# %%
# ==============================================================================
# CDE COMPUTATION FUNCTION
# ==============================================================================
# Conservation Diversity Entropy (CDE) measures the mutability at each position
# by computing the entropy of the probability distribution over all amino acids
# when considering single mutations at that position.
#
# Algorithm:
# 1. For each sequence, generate all possible single-point mutations
# 2. Compute energy for each mutant using the DCA model
# 3. Calculate probability distribution: p_i = exp(-E_i) / Z
# 4. Compute entropy: CDE = -sum(p_i * log2(p_i))
#
# Returns CDE values for each position in each sequence
# ==============================================================================

def compute_CDE_full_batch_fast(sequences, model, device=device, dtype=dtype):
    """
    Compute Conservation Diversity Entropy for all sequences in batch.
    
    Parameters:
    -----------
    sequences : torch.Tensor
        One-hot encoded sequences of shape (batch_size, L, q)
    model : dict
        DCA model parameters containing 'bias' and 'coupling_matrix'
    device : torch.device
        Device to use for computation (CPU or CUDA)
    dtype : torch.dtype
        Data type for tensor operations
        
    Returns:
    --------
    CDE_full_batch : torch.Tensor
        CDE values of shape (batch_size, L) on CPU
    """
    
    with torch.no_grad():  # Memory efficient - no gradient computation needed
        batch_size, L, q = sequences.shape
        CDE_full_batch = torch.zeros((batch_size, L), device=device, dtype=dtype)
 
        # Process in chunks to manage GPU memory
        chunk_size = 150
        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            sequences_chunk = sequences[start:end]
            chunk_batch_size = sequences_chunk.shape[0]
            
            # Move model parameters to the target device
            model_device = {}
            model_device["bias"] = model["bias"].to(device)
            model_device["coupling_matrix"] = model["coupling_matrix"].to(device)
            
            # Get amino acid indices from one-hot encoding
            sequence_indices = sequences_chunk.argmax(-1)
            
            # Create base tensor for all possible single mutations
            # Shape: (chunk_batch_size * L * q, L)
            base = sequence_indices.unsqueeze(1).expand(-1, L * q, -1).reshape(chunk_batch_size * L * q, L).clone().to(device)
            
            # Generate position and value arrays for mutations
            positions = torch.arange(L, device=device).repeat_interleave(q).repeat(chunk_batch_size)
            values = torch.arange(q, device=device, dtype=torch.int64).repeat(L).repeat(chunk_batch_size)
            
            # Apply mutations to create Deep Mutational Scan (DMS) sequences
            base[torch.arange(chunk_batch_size * L * q, device=device), positions] = values
            dms = base
            
            # Convert to one-hot encoding for energy computation
            dms_oh = one_hot(dms, num_classes=q).to(dtype)
            
            # Compute energy for all mutants
            energy_dms = compute_energy(dms_oh, model_device)
            energy_dms = energy_dms.view(chunk_batch_size, L, q)
            
            # Normalize energies (shift minimum to zero for numerical stability)
            energy_dms -= energy_dms.min(dim=2, keepdim=True)[0]
            
            # Compute Boltzmann probabilities: p_i = exp(-E_i) / Z
            exp_energy_dms = torch.exp(-energy_dms)
            sums = exp_energy_dms.sum(dim=2, keepdim=True)
            p_i = exp_energy_dms / sums
            
            # Compute Shannon entropy as CDE measure
            CDE_chunk = -torch.sum(p_i * torch.log2(p_i + 1e-10), dim=2)  # Add small epsilon to avoid log(0)
            CDE_full_batch[start:end] = CDE_chunk.cpu()
            
            # Free GPU memory explicitly
            del base, positions, values, dms, dms_oh, energy_dms, exp_energy_dms, sums, p_i
            del model_device["bias"], model_device["coupling_matrix"], CDE_chunk
            
        # Clean up
        del sequences
        gc.collect()
        torch.cuda.empty_cache()
        
    return CDE_full_batch.cpu()

print("CDE computation function defined!")

# %% [markdown]
# ## Compute Model-Specific CDE Scores
#
# Run the CDE calculation for each model and summarize each sequence by its mean CDE across positions.

# %%
# ==============================================================================
# COMPUTE CDE FOR ALL MODELS
# ==============================================================================
# Compute CDE values for natural MSA sequences using each DCA model
# CDE quantifies position-specific mutability as predicted by each model
# ==============================================================================

print("Computing CDE for all models...")
print("This may take several minutes depending on MSA size and GPU availability...")

# Move data to GPU for faster computation
msa_oh_nat_gpu = msa_oh_nat.to(device2)

# Initialize storage for results
CDE_results = {}
CDE_mean_results = {}

# Create model dictionary with descriptive names
models = {
    'meDCA': HA_model,      # Medium entropy model
    'edDCA': dec_model,     # Extreme density model
    'eaDCA': ea_model,      # Extreme attributes model
    'bmDCA': dense_model    # Boltzmann machine model
}

# Compute CDE for each model
for model_name, model in models.items():
    print(f"\n{model_name} model:")
    
    # Compute CDE for all positions and sequences
    CDE = compute_CDE_full_batch_fast(msa_oh_nat_gpu, model, device=device2)
    print(f"  CDE shape: {CDE.shape}")
    
    # Compute mean CDE across all positions for each sequence
    # This gives a single mutability score per sequence
    CDE_mean = CDE.mean(-1)
    print(f"  CDE mean shape: {CDE_mean.shape}")
    print(f"  Average CDE value: {CDE_mean.mean().item():.4f}")
    
    # Store results
    CDE_results[model_name] = CDE
    CDE_mean_results[model_name] = CDE_mean

print("\n" + "="*60)
print("All CDE computations completed successfully!")
print("="*60)

# %% [markdown]
# ## Four-Model Mutability Comparison
#
# Plot the weighted distribution of mean CDE values for all four models and save the full comparison figure.

# %%
# ==============================================================================
# VIOLIN PLOT - ALL FOUR MODELS
# ==============================================================================
# Create violin plot comparing CDE distributions across all four DCA models
# Using weighted sampling to ensure representative sequences
# ==============================================================================

# Define model order for plot (low to high entropy)
order = ['eaDCA', 'bmDCA', 'edDCA', 'meDCA']

# Extract sampled CDE mean values for each model
sampled_cde_means = [CDE_mean_results[name].cpu().numpy()[sampled_indices] for name in order]
model_names = order

# Create violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(
    data=sampled_cde_means, 
    inner="box",  # Show box plot inside violin
    palette=[colors[name] for name in model_names]
)

# Configure plot aesthetics
plt.xticks(ticks=range(len(model_names)), labels=model_names, fontsize=24)
plt.ylabel('Mutability (CDE)', fontsize=26)
plt.yticks(fontsize=24)
plt.title('')
plt.grid(True, alpha=0.4)
plt.tight_layout()

# Save figure
plt.savefig(FIGURES_SUPPLEMENTARY / "figS6_mutability_all_models.pdf", dpi=300, bbox_inches='tight')
print("Saved: violin_plot_CDE_mean_ALL.pdf")

plt.show()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS - ALL MODELS")
print("="*60)
for name in order:
    values = CDE_mean_results[name].cpu().numpy()[sampled_indices]
    print(f"\n{name}:")
    print(f"  Mean ± Std: {values.mean():.4f} ± {values.std():.4f}")
    print(f"  Range: [{values.min():.4f}, {values.max():.4f}]")

# %% [markdown]
# ## Main Mutability Contrast
#
# Plot the reduced eaDCA-versus-meDCA comparison used to emphasize the difference between the lowest- and highest-mutability regimes.

# %%
# ==============================================================================
# VIOLIN PLOT - eaDCA vs meDCA COMPARISON
# ==============================================================================
# Create violin plot comparing only the extreme cases:
# - eaDCA: Extreme attributes model (typically lower CDE)
# - meDCA: Medium entropy model (typically higher CDE)
# ==============================================================================

# Select only eaDCA and meDCA models
order = ['eaDCA', 'meDCA']

# Extract sampled CDE mean values for selected models
sampled_cde_means = [CDE_mean_results[name].cpu().numpy()[sampled_indices] for name in order]
model_names = order

# Create violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(
    data=sampled_cde_means, 
    inner="box",  # Show box plot inside violin
    palette=[colors[name] for name in model_names]
)

# Configure plot aesthetics
plt.xticks(ticks=range(len(model_names)), labels=model_names, fontsize=24)
plt.ylabel('Mutability (CDE)', fontsize=26)
plt.yticks(fontsize=24)
plt.title('')
plt.grid(True, alpha=0.4)
plt.tight_layout()

# Save figure
plt.savefig(FIGURES_INTERMEDIATE / "mutability_violin_two_models.pdf", dpi=300, bbox_inches='tight')
print("Saved: violin_plot_CDE_mean.pdf")

plt.show()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS - eaDCA vs meDCA")
print("="*60)
for name in order:
    values = CDE_mean_results[name].cpu().numpy()[sampled_indices]
    print(f"\n{name}:")
    print(f"  Mean ± Std: {values.mean():.4f} ± {values.std():.4f}")
    print(f"  Range: [{values.min():.4f}, {values.max():.4f}]")

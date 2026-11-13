#!/usr/bin/env python3
"""
Construction of the leave-one-group-out training sets.

Data preparation step, run once before leave_one_out_generalization.py: it selects the
10 mutually distant natural sequences and writes the filtered MSAs that exclude each of
them together with its close homologs.

Figures
-------
None.

Inputs
------
data/natural_msa/   natural MSA

Outputs
-------
data/leave_one_out/filtered_msa/   selected_sequences.fasta and msa_exclude_<i>.fasta

Note
----
The writing cells are deliberately commented out, so running this script cannot
overwrite the filtered MSAs that the published analysis used. Uncomment them only to
regenerate the data from scratch.

Runtime
-------
Seconds, CPU only.

# MSA Filtering Protocol for Leave-One-Out Analysis

This script documents how the filtered MSAs used in the overfitting analysis were generated. The code cells are intentionally commented out so opening this file from GitHub does not accidentally overwrite the committed FASTA files.

To regenerate the data, uncomment and run the cells sequentially after verifying the input and output paths.

Converted from the notebook data_analysis/overfitting_analysis/filter_msa.ipynb; the analysis code is unchanged.
Cell boundaries are kept as `# %%`, so the script can also be run cell by cell.
"""

# %% [markdown]
# ## Dependencies and Runtime Settings
#
# Imports and device configuration required to rerun the filtering procedure. The code is commented out intentionally because the filtered FASTA files are already included in the repository.

# %%
# import adabmDCA
# import copy
# import time
# from pathlib import Path
# import numpy as np
# import torch
# import matplotlib.pyplot as plt

# # Import some useful functions from adabmDCA
# from adabmDCA.fasta import get_tokens, import_from_fasta, compute_weights
# from adabmDCA.stats import get_freq_single_point, get_freq_two_points
# from adabmDCA.functional import one_hot
# from adabmDCA.utils import init_parameters, init_chains, get_device, get_dtype
# from adabmDCA.sampling import get_sampler


# from adabmDCA.io import load_params, load_chains
# # from adabmDCA.resampling import compute_mixing_time, compute_seqID, get_mean_seqid
# from adabmDCA.utils import resample_sequences
# from adabmDCA.statmech import compute_energy

# # Set the device
# device = get_device("cuda")
# dtype = get_dtype("float32")

# %% [markdown]
# ## Load the Source MSA
#
# Read the Chorismate Mutase MSA used as the starting point for selecting leave-one-out sequences.

# %%
# tokens = get_tokens("protein")
# headers, msa_data = import_from_fasta("../data/CM_130530_MC.fasta", tokens=tokens, filter_sequences=True)

# %% [markdown]
# ## Pairwise Hamming Distance Function
#
# Define the pairwise Hamming-distance matrix used to quantify sequence diversity across the full MSA.

# %%
# def compute_hamming_distance_matrix(msa_data):
#     """
#     Compute the Hamming distance matrix for all sequences in the MSA.
    
#     Parameters:
#     -----------
#     msa_data : torch.Tensor or np.ndarray
#         Multiple sequence alignment data of shape (N, L) where N is the number
#         of sequences and L is the sequence length.
    
#     Returns:
#     --------
#     distance_matrix : torch.Tensor
#         NxN matrix where element [i,j] contains the Hamming distance between
#         sequence i and sequence j.
#     """
#     # Convert to tensor if needed
#     if isinstance(msa_data, np.ndarray):
#         msa_data = torch.from_numpy(msa_data)
    
#     N, L = msa_data.shape
    
#     # Expand dimensions to compute pairwise differences
#     # msa_data[i, :] != msa_data[j, :] gives a boolean array
#     # We sum over the sequence length dimension to get distances
    
#     # Reshape for broadcasting: (N, 1, L) and (1, N, L)
#     msa_expanded_i = msa_data.unsqueeze(1)  # (N, 1, L)
#     msa_expanded_j = msa_data.unsqueeze(0)  # (1, N, L)
    
#     # Compute element-wise inequality and sum over sequence length
#     distance_matrix = (msa_expanded_i != msa_expanded_j).sum(dim=2).float()
    
#     return distance_matrix

# %% [markdown]
# ## Compute Full Distance Matrix
#
# Evaluate the all-by-all Hamming distances for the source MSA.

# %%
# distance_matrix = compute_hamming_distance_matrix(msa_data)

# %% [markdown]
# ## Select Distant Representative Sequences
#
# Greedily select ten sequences that are mutually distant enough to serve as challenging leave-one-out cases.

# %%
# # Compute normalized distance (as percentage)
# N, L = msa_data.shape
# normalized_distance = distance_matrix / L

# # Find 10 sequences with at least 80% distance from each other
# selected_indices = []
# threshold = 0.80

# # Start with the first sequence
# selected_indices.append(0)

# # Greedily select sequences that are far from all previously selected ones
# for i in range(1, N):
#     if len(selected_indices) >= 10:
#         break
    
#     # Check if sequence i is far enough from all selected sequences
#     is_far_enough = True
#     for selected_idx in selected_indices:
#         if normalized_distance[i, selected_idx] < threshold:
#             is_far_enough = False
#             break
    
#     if is_far_enough:
#         selected_indices.append(i)

# print(f"Found {len(selected_indices)} sequences with at least 80% distance from each other")
# print(f"Selected indices: {selected_indices}")

# # If we found fewer than 10, lower the threshold and try again
# if len(selected_indices) < 10:
#     print(f"Only found {len(selected_indices)} sequences at 80% threshold")
#     print("Trying with lower thresholds...")
    
#     for new_threshold in [0.75, 0.70, 0.65, 0.60]:
#         selected_indices = [0]
#         for i in range(1, N):
#             if len(selected_indices) >= 10:
#                 break
#             is_far_enough = True
#             for selected_idx in selected_indices:
#                 if normalized_distance[i, selected_idx] < new_threshold:
#                     is_far_enough = False
#                     break
#             if is_far_enough:
#                 selected_indices.append(i)
        
#         print(f"At {new_threshold*100}% threshold: found {len(selected_indices)} sequences")
#         if len(selected_indices) >= 10:
#             break

# selected_indices = selected_indices[:10]  # Take only first 10
# print(f"\nFinal selection: {len(selected_indices)} sequences")
# print(f"Indices: {selected_indices}")

# %% [markdown]
# ## Save Selected Sequences
#
# Write the ten selected sequences to FASTA so the downstream analysis can refer to a fixed held-out set.

# %%
# # Save the 10 selected sequences
# from adabmDCA.fasta import write_fasta

# selected_headers = headers[selected_indices]
# selected_sequences = msa_data[selected_indices]

# write_fasta(
#     fname="filtered_msa/selected_sequences.fasta",
#     headers=selected_headers,
#     sequences=selected_sequences,
#     tokens=tokens
# )

# print(f"Saved {len(selected_indices)} selected sequences to 'selected_sequences.fasta'")

# %% [markdown]
# ## Build Filtered Leave-One-Out MSAs
#
# Create one filtered MSA per selected sequence by removing the selected sequence and close homologs around it.

# %%
# # Create 10 new MSAs, each excluding one selected sequence and all sequences <20% distance from it
# import os

# output_dir = "filtered_msa"
# os.makedirs(output_dir, exist_ok=True)

# for i, exclude_idx in enumerate(selected_indices):
#     print(f"\nProcessing MSA {i+1}/10 (excluding sequence at index {exclude_idx})...")
    
#     # Find all sequences that are <20% distance from the excluded sequence
#     close_sequences_mask = normalized_distance[exclude_idx] < 0.20
    
#     # Create a mask for sequences to keep (NOT close to the excluded sequence)
#     keep_mask = ~close_sequences_mask
    
#     # Get indices of sequences to keep
#     keep_indices = torch.nonzero(keep_mask).squeeze().numpy()
#     if keep_indices.ndim == 0:
#         keep_indices = keep_indices.reshape(1)
    
#     # Extract the filtered MSA
#     filtered_headers = headers[keep_indices]
#     filtered_sequences = msa_data[keep_indices]
    
#     # Save to file
#     output_filename = f"{output_dir}/msa_exclude_{i+1}.fasta"
#     write_fasta(
#         fname=output_filename,
#         headers=filtered_headers,
#         sequences=filtered_sequences,
#         tokens=tokens
#     )
    
#     num_excluded = close_sequences_mask.sum().item()
#     num_kept = len(keep_indices)
#     print(f"  Excluded: {num_excluded} sequences (including the selected one)")
#     print(f"  Kept: {num_kept} sequences")
#     print(f"  Saved to: {output_filename}")

# print(f"\n✓ All 10 filtered MSAs saved to '{output_dir}/' directory")

# %% [markdown]
# ## One-Hot Encoding for PCA
#
# Convert the MSA to one-hot features and flatten the representation for principal component analysis.

# %%
# # One-hot encode the MSA data
# from adabmDCA.functional import one_hot

# # Convert to tensor first
# msa_data_tensor = torch.from_numpy(msa_data) if isinstance(msa_data, np.ndarray) else msa_data

# # One-hot encode: shape (N, L) -> (N, L, q) where q is the number of tokens
# msa_one_hot = one_hot(msa_data_tensor, num_classes=len(tokens))
# print(f"One-hot encoded shape: {msa_one_hot.shape}")

# # Flatten to (N, L*q) for PCA
# N, L, q = msa_one_hot.shape
# msa_flattened = msa_one_hot.reshape(N, L * q).cpu().numpy()
# print(f"Flattened shape for PCA: {msa_flattened.shape}")

# %% [markdown]
# ## Fit PCA and Project Selected Sequences
#
# Fit a two-component PCA model to the full MSA and project the selected held-out sequences into that space.

# %%
# # Train PCA on the full MSA data
# from sklearn.decomposition import PCA

# # Initialize and fit PCA
# pca = PCA(n_components=2)
# msa_pca = pca.fit_transform(msa_flattened)

# print(f"PCA trained on {N} sequences")
# print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
# print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")

# # Transform selected sequences
# selected_sequences_tensor = torch.from_numpy(selected_sequences) if isinstance(selected_sequences, np.ndarray) else selected_sequences
# selected_one_hot = one_hot(selected_sequences_tensor, num_classes=len(tokens))
# selected_flattened = selected_one_hot.reshape(len(selected_indices), L * q).cpu().numpy()
# selected_pca = pca.transform(selected_flattened)

# print(f"\nProjected all {N} sequences and {len(selected_indices)} selected sequences")

# %% [markdown]
# ## Visualize the Selected Sequences
#
# Plot the full MSA and the selected leave-one-out sequences in PCA space to document the coverage of the held-out set.

# %%
# # Plot PCA projection
# fig, ax = plt.subplots(figsize=(10, 8))

# # Plot all MSA sequences
# ax.scatter(msa_pca[:, 0], msa_pca[:, 1], alpha=0.3, s=20, c='gray', label='All sequences')

# # Plot selected sequences with different colors
# ax.scatter(selected_pca[:, 0], selected_pca[:, 1], alpha=0.8, s=100, c='red', 
#            edgecolors='black', linewidths=1.5, label='Selected sequences', marker='o')

# # Annotate selected sequences
# for i, (x, y) in enumerate(selected_pca):
#     ax.annotate(f'{i+1}', (x, y), fontsize=8, ha='center', va='center', 
#                 color='white', weight='bold')

# ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
# ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
# ax.set_title('PCA Projection: All MSA sequences vs Selected sequences', fontsize=14)
# ax.legend(fontsize=10)
# ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

# print(f"✓ Plotted {N} total sequences with {len(selected_indices)} selected sequences highlighted")

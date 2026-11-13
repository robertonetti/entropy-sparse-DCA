#!/usr/bin/env python3
"""
GPT inference and entropy estimation on Chorismate Mutase.

Results
-------
Tab. S1   printed to stdout (GPT architectures, entropies and correlations)

Inputs
------
data/transformer/      training and validation FASTA files
checkpoints            NOT included in this package: the trained GPT checkpoints are
                       too large to ship, so the path has to be set by hand before use

Runtime
-------
Minutes per architecture on a GPU, once a checkpoint is available.

# Transformer Inference and Sampling

This script reproduces the inference workflow for the Chorismate Mutase autoregressive protein Transformer. It loads the same vocabulary and dataset configuration used during training, restores a trained checkpoint, generates new sequences, scores them with the model, and optionally loads external FASTA sequences for inspection.

The script is intended as a reproducible companion artifact for the manuscript: paths and hyperparameters are explicit, generated samples are evaluated against natural one- and two-point statistics, and FASTA output can be inspected directly.

Converted from the notebook data_analysis/transformer_train/Inference.ipynb; the analysis code is unchanged.
Cell boundaries are kept as `# %%`, so the script can also be run cell by cell.
"""

# %% [markdown]
# ## Imports and Project Modules
#
# Import PyTorch utilities, path/context helpers, and the local Transformer, dataset, loss, and Monte Carlo helper functions used throughout the notebook.

# %%
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # scripts/ holds paths.py
from paths import TRANSFORMER_DATA_DIR

import torch
import torch.nn.functional as F
import pandas as pd
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch import nn
from pathlib import Path
from contextlib import nullcontext

from adabmDCA.stats import get_correlation_two_points, get_freq_single_point, get_freq_two_points
from transformer import GPTTransformer
from Transformer_Reint import GPTmodel, load_dataset, get_batch, reint_loss, estimate_evaluation_losses, reint_ppo_loss, dkl_between_logits, set_dropout

# %% [markdown]
# ## Device, Vocabulary, and Dataset Setup
#
# Select CPU or GPU, define the model context length, read the training FASTA file, build the character-level protein vocabulary, add a padding token, define encode/decode helpers, and load the FASTA sequences into padded tensors for train/validation splits.

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_context = 98

TRAIN_DN_path = str(TRANSFORMER_DATA_DIR / "CM_130530_MC.fasta")
VALIDATION_DN_path = str(TRANSFORMER_DATA_DIR / "CM_validation.fasta")
reweighting = True
reweighting_th = 0.8

temp_sequences = []
for dataset_path in (TRAIN_DN_path, VALIDATION_DN_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        temp_sequences.extend(f.read().splitlines()[1::2])
temp = "\n".join(temp_sequences)


chars = sorted(list(set(temp)))
# Create the initial character-to-integer mapping from FASTA symbols.
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

PAD_TOKEN = "?"

# Add a padding symbol after the biological and special sequence tokens.
chars = chars + [PAD_TOKEN]

# Rebuild mappings after adding the padding token.
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Store the integer ID of the padding token.
pad_token = stoi[PAD_TOKEN]

vocab_size = len(stoi)

encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

decode_batch = lambda ls: ["".join([itos[i] for i in l]) for l in ls]
encode_batch = lambda ss: torch.tensor([[stoi[c] for c in s] for s in ss]).to(device)


DN = load_dataset(  
      
    seq_path=TRAIN_DN_path,
    validation_seq_path=VALIDATION_DN_path,
    label_path='.',
    block_size=max_context,
    train_size=None,
    encode_fn=encode,
    pad_token=pad_token,
    reweighting=reweighting,
    reweighting_th=reweighting_th,
)

# %% [markdown]
# ## Natural Cij Reference
#
# Warm up the Cij/statistics functions on a tiny subset, then convert the natural training sequences to aligned one-hot tensors, keep only the 96 biological alignment positions, and compute the natural one- and two-point frequencies used as the Cij reference.

# %%
sequence_length = 96
generation_sample_size = 10_000
cij_warmup_size = 8

valid_token_ids = [stoi[ch] for ch in chars if ch not in {PAD_TOKEN, "\n"}]
token_remap = {token_id: idx for idx, token_id in enumerate(valid_token_ids)}


def tokens_to_alignment_one_hot(sequences, valid_token_ids, token_remap, sequence_length, device):
    alignment_sequences = []
    valid_tokens = set(valid_token_ids)
    for seq in sequences:
        filtered = [token_remap[int(token)] for token in seq.tolist() if int(token) in valid_tokens]
        if len(filtered) >= sequence_length:
            alignment_sequences.append(filtered[:sequence_length])

    if not alignment_sequences:
        return None

    alignment_sequences = torch.tensor(alignment_sequences, dtype=torch.long, device=device)
    return F.one_hot(alignment_sequences, num_classes=len(valid_token_ids)).to(torch.float32)


train_weights = torch.tensor(DN["train"]["weights"], dtype=torch.float32, device=device)
if reweighting and torch.any(train_weights == 0):
    raise ValueError("Reweighting is enabled, but train_weights contains zero-valued entries.")

cij_warmup = tokens_to_alignment_one_hot(
    DN["train"]["data"][:cij_warmup_size],
    valid_token_ids,
    token_remap,
    sequence_length,
    device,
)
if cij_warmup is not None:
    cij_warmup_weights = train_weights[:cij_warmup.shape[0]] if reweighting else None
    fi_warmup = get_freq_single_point(data=cij_warmup, weights=cij_warmup_weights, pseudo_count=0.0)
    fij_warmup = get_freq_two_points(data=cij_warmup, weights=cij_warmup_weights, pseudo_count=0.0)
    _ = get_correlation_two_points(
        fi=fi_warmup,
        pi=fi_warmup,
        fij=fij_warmup,
        pij=fij_warmup,
    )

natural_train = tokens_to_alignment_one_hot(
    DN["train"]["data"],
    valid_token_ids,
    token_remap,
    sequence_length,
    device,
)
if natural_train is None:
    raise ValueError("No valid fixed-length natural training sequences found for Cij evaluation.")

natural_weights = train_weights if reweighting else None
fi_natural = get_freq_single_point(data=natural_train, weights=natural_weights, pseudo_count=0.0)
fij_natural = get_freq_two_points(data=natural_train, weights=natural_weights, pseudo_count=0.0)

print(
    f"Natural Cij reference: sequences={natural_train.shape[0]}, L={natural_train.shape[1]}, "
    f"q={natural_train.shape[2]}, reweighting={reweighting}, Meff={train_weights.sum().item():.3f}"
)

# %% [markdown]
# ## Model Configuration and Checkpoint Loading
#
# Define the Transformer architecture hyperparameters, instantiate the `GPTTransformer`, load a pre-trained chorismate mutase checkpoint, move the model to the selected device, and compile it for faster execution.

# %%
# model params
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1
loss_threshold = 1 / 16 
alpha = 1  
num_bits = 3

legacy_model = GPTTransformer(
    vocab_size=vocab_size,
    embed_dim=n_embd,
    num_layers=n_layer,
    num_heads=n_head,
    mlp_ratio=4,
    dropout_p=0.1,
    pad_id=pad_token,
)
ckpt_path_gpt = Path("./checkpoints/CM/CM_130530_MC_GPT_Rew_L4_H4_E128/CM_130530_MC_GPT_Rew_iter220.pt")
legacy_model.load_state_dict(torch.load(ckpt_path_gpt, map_location="cpu"))
legacy_model = legacy_model.to(device)
legacy_model = torch.compile(legacy_model)

# %% [markdown]
# ## Precision and Autocast Context
#
# Enable TensorFloat-32 on CUDA when available, choose the floating-point precision, and create the autocast context used by scoring or generation helper functions.

# %%
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    if "cuda" in device.type
    else nullcontext()
)

# %% [markdown]
# ## Monte Carlo Helper Imports
#
# Load generation, scoring, FASTA printing, and FASTA loading utilities from the local `mc.py` module.

# %%
from mc import *

# %% [markdown]
# ## Sequence Generation
#
# Switch the model to evaluation mode and sample a batch of new sequences autoregressively. Generation starts from the newline token used as both beginning/end-of-sequence marker, samples up to the requested token count, and pads sequences after an end marker is produced.

# %%
legacy_model.eval()

generated_sequences, per_token_legacy_log_probs, per_token_sampling_log_probs = generate_block(
    legacy_model,
    beta_sampling=1,
    N=generation_sample_size,
    num_tokens=sequence_length + 1,
    max_context=max_context,
    device=device,
    bos_eos_token=stoi["\n"],
    pad_token=pad_token,
)

# %% [markdown]
# ## Cij, Entropy, and Summary Table
#
# Convert the generated sequences to aligned one-hot tensors, compute their Cij statistics and Pearson correlation against the natural reference, estimate both the total generated-sequence entropy and the entropy restricted to the 96 aligned positions, and summarize the model size and metrics in a table.

# %%
generated_one_hot = tokens_to_alignment_one_hot(
    generated_sequences,
    valid_token_ids,
    token_remap,
    sequence_length,
    device,
)
if generated_one_hot is None:
    raise ValueError("No valid fixed-length generated sequences found for Cij evaluation.")

pi_generated = get_freq_single_point(data=generated_one_hot, weights=None, pseudo_count=0.0)
pij_generated = get_freq_two_points(data=generated_one_hot, weights=None, pseudo_count=0.0)
pearson_cij, slope_cij = get_correlation_two_points(
    fi=fi_natural,
    pi=pi_generated,
    fij=fij_natural,
    pij=pij_generated,
)

legacy_model.eval()
generated_log_probs = score_sequences(generated_sequences, legacy_model, pad_token=pad_token)
entropy_nats = -generated_log_probs.mean().item()

alignment_sequences = []
valid_tokens = set(valid_token_ids)
for seq in generated_sequences:
    filtered = [int(token) for token in seq.tolist() if int(token) in valid_tokens]
    if len(filtered) >= sequence_length:
        alignment_sequences.append([stoi["\n"]] + filtered[:sequence_length])

if not alignment_sequences:
    raise ValueError("No generated sequences with 96 valid alignment positions found for alignment entropy.")

alignment_sequences = torch.tensor(alignment_sequences, dtype=torch.long, device=device)
with torch.inference_mode():
    alignment_logits = legacy_model(alignment_sequences[:, :-1])["logits"]
    alignment_targets = alignment_sequences[:, 1:]
    alignment_loss_per_token = F.cross_entropy(
        alignment_logits.transpose(1, 2),
        alignment_targets,
        reduction="none",
    )
entropy_96_positions_nats = alignment_loss_per_token.sum(dim=1).mean().item()

model_for_params = legacy_model._orig_mod if hasattr(legacy_model, "_orig_mod") else legacy_model
num_parameters = sum(p.numel() for p in model_for_params.parameters())

summary_table = pd.DataFrame([
    {
        "num_parameters": num_parameters,
        "pearson_cij": pearson_cij,
        "entropy_nats": entropy_nats,
        "entropy_96_positions_nats": entropy_96_positions_nats,
    }
])
print(summary_table)

# %% [markdown]
# ## FASTA Output for Generated Sequences
#
# Decode the generated token tensors back into sequence strings, remove padding, and print them in FASTA-like format.

# %%
print_fasta(generated_sequences, pad_token, decode)

# %% [markdown]
# ## Loading External FASTA Sequences
#
# Load sequences from an external FASTA file, encode them with the same vocabulary, add beginning/end markers, pad them to a common length, and print the decoded result. This is useful for checking real or candidate sequences with the model vocabulary before scoring.

# %%
path = "./DATA/seq_toload.fasta"
seqs = load_fasta(path, encode, 0, pad_token, device, max_length=None)
print_fasta(seqs, pad_token, decode)

# %% [markdown]
# ## Vocabulary Inspection
#
# Display the string-to-index vocabulary mapping used by the model, including amino acids, gap or special symbols present in the training data, the newline marker, and the padding token.

# %%
print(stoi)

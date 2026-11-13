"""
Train the autoregressive Transformer on Chorismate Mutase protein sequences.

The script builds a character-level protein vocabulary from the training and
validation FASTA files, loads padded sequence tensors, trains a decoder-only
Transformer with optional sequence reweighting, and writes checkpoints plus a
plain-text training log under `checkpoints/CM/`.

Optional sampling metrics estimate how well generated sequences reproduce
natural one- and two-point statistics. They are disabled by default because
they require repeated large-batch generation during evaluation.
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # scripts/ holds paths.py
from paths import TRANSFORMER_DATA_DIR

import logging
import os
import sys
from contextlib import nullcontext, redirect_stdout
from io import StringIO

import torch
import torch.amp
import torch.nn.functional as F
from tqdm.auto import tqdm

from adabmDCA.stats import get_correlation_two_points, get_freq_single_point, get_freq_two_points

from mc import generate_block, score_sequences
from transformer import GPTTransformer
from Transformer_Reint import load_dataset, get_batch


# -----------------------------------------------------------------------------
# Dataset selection
# -----------------------------------------------------------------------------
# Choose the FASTA files used for training and validation.
# Explicit paths so they can be edited directly.
TRAIN_DN_path = str(TRANSFORMER_DATA_DIR / "CM_130530_MC.fasta")
VALIDATION_DN_path = str(TRANSFORMER_DATA_DIR / "CM_validation.fasta")


# -----------------------------------------------------------------------------
# Training and model hyperparameters
# -----------------------------------------------------------------------------
# These settings control numerical precision, attention backend selection,
# optimizer behavior, evaluation frequency, checkpoint naming, sequence context
# length, and Transformer size.
# Training parameters.
dtype = "bfloat16"  # 'float32', 'bfloat16', or 'float16'
use_flash_attn_if_available = True
use_mem_efficient_attn_if_available = (
    False  # by default, if both are true and available, flash attn is selected
)

batch_size = 512
max_iters = 300
eval_interval = 20
scheduler_step_size = 10_000
learning_rate = 1e-3
eval_iters = None  # None evaluates the full validation set once, split by batch_size.
log_steps = 20
compute_sampling_metrics = False

early_stopping_patience = 5
early_stopping_min_delta = 1e-4

reweighting = True
reweighting_th = 0.8
cij_sample_size = 10_000
cij_warmup_size = 8

# Model parameters.
model_name = "CM_130530_MC_GPT_Rew"  # string identifier for checkpoint naming
block_size = 98  # maximum context length for predictions (e.g. 1024 in GPT-3)
sequence_length = 96
dropout = 0.1


# Very small, fast baseline.
# n_embd = 32
# n_head = 4
# n_layer = 2

# Small, usually a reasonable first test.
# n_embd = 64
# n_head = 4
# n_layer = 4

# Medium-small, more capacity while remaining lightweight.
# n_embd = 96
# n_head = 4
# n_layer = 4

# Medium, useful if smaller models underfit.
# n_embd = 128
# n_head = 4
# n_layer = 4

# Deeper model; monitor overfitting carefully.
n_embd = 128
n_head = 8
n_layer = 6

# Large for this dataset; use only with strong regularization and monitoring.
# n_embd = 256
# n_head = 8
# n_layer = 6

# -----------------------------------------------------------------------------
# Device, precision, and autocast setup
# -----------------------------------------------------------------------------
# Select CUDA when available, fall back from bfloat16 when unsupported, enable
# TF32 kernels on NVIDIA GPUs, and build the autocast/scaler objects used during
# mixed-precision training.
# device and dtype settings
if not torch.cuda.is_available() or (not torch.cuda.is_bf16_supported() and dtype == "bfloat16"):
    dtype = "float32"

# For Matrix Multiplications (cuBLAS, affects nn.Linear, torch.matmul)

torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
# auto implement a GradScaler (to convert fp16 gradients to fp32, prevent underflow)
scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))


# -----------------------------------------------------------------------------
# Attention backend selection
# -----------------------------------------------------------------------------
# Check whether flash attention or memory-efficient attention can be used for the
# selected GPU/dtype combination, then enable the fastest available backend
# allowed by the configuration flags above.
# util to check if flash attention or efficient attention are being used
def set_attention_backend(model, dtype):
    """Enable the fastest CUDA scaled-dot-product attention backend available."""
    device = next(model.parameters()).device
    x = torch.rand((1, 8, 16, 8), dtype=dtype, device=device)  # dummy input [B, H, T, S]
    # all params must be set as implementation falls back to C class
    params = torch.backends.cuda.SDPAParams(
        x,     # query
        x,     # key
        x,     # value
        None,  # attn_mask
        0.0,   # dropout_p
        True,  # is_causal
        False, # enable_gqa
    )
    use_flash_attn = False
    use_mem_efficient_attn = False
    if torch.backends.cuda.can_use_flash_attention(params) and use_flash_attn_if_available:
        use_flash_attn = True
    else:
        # print("Flash attention disabled or not available")
        torch.backends.cuda.enable_flash_sdp(enabled=False)

    if (
        torch.backends.cuda.can_use_efficient_attention(params)
        and use_mem_efficient_attn_if_available
        and not use_flash_attn
    ):
        use_mem_efficient_attn = True
    else:
        # print("Efficient attention disabled or not available")
        torch.backends.cuda.enable_mem_efficient_sdp(enabled=False)
    

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
# Configure a file logger under the checkpoint directory and clear previous
# handlers so repeated runs do not duplicate log messages.
# set logger
def set_logger(log_path):
    """Configure stdout and file logging for a single training run."""
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    # Remove existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.FileHandler(log_path)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    with open(log_path, "w") as _:
        pass
    return logger


torch.manual_seed(1337)


# -----------------------------------------------------------------------------
# Vocabulary construction
# -----------------------------------------------------------------------------
# Read the FASTA sequences, collect every character present in the training data
# as a token, add a separate padding token, and define encode/decode helpers for
# character-level protein language modeling.
temp_sequences = []
for dataset_path in (TRAIN_DN_path, VALIDATION_DN_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        temp_sequences.extend(f.read().splitlines()[1::2])
temp = "\n".join(temp_sequences)


chars = sorted(list(set(temp)))
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

PAD_TOKEN = "?"

# Add the padding token after all biological and special sequence tokens.
chars = chars + [PAD_TOKEN]

# Rebuild the mappings after adding the padding token.
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Store the integer ID of the padding token.
pad_token = stoi[PAD_TOKEN]

vocab_size = len(stoi)

encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string


def tokens_to_alignment_one_hot(
    sequences,
    valid_token_ids,
    token_remap,
    sequence_length,
    device,
    return_sequences=False,
):
    """
    Convert generated token sequences into fixed-length alignment one-hot tensors.

    Special tokens such as padding and the newline BOS/EOS marker are removed.
    Sequences shorter than `sequence_length` after filtering are skipped because
    they cannot be compared position-by-position with the natural alignment.
    """
    alignment_sequences = []
    selected_sequences = []
    valid_tokens = set(valid_token_ids)
    for seq in sequences:
        filtered = [token_remap[int(token)] for token in seq.tolist() if int(token) in valid_tokens]
        if len(filtered) >= sequence_length:
            alignment_sequences.append(filtered[:sequence_length])
            selected_sequences.append(seq)

    if not alignment_sequences:
        return (None, None) if return_sequences else None

    alignment_sequences = torch.tensor(alignment_sequences, dtype=torch.long, device=device)
    one_hot = F.one_hot(alignment_sequences, num_classes=len(valid_token_ids)).to(torch.float32)
    if return_sequences:
        return one_hot, torch.stack(selected_sequences).to(device)
    return one_hot


def sample_alignment_one_hot(model, valid_token_ids, token_remap):
    """
    Generate enough fixed-length samples for Cij evaluation.

    Generation can produce early EOS tokens; therefore the function retries up
    to `max_attempts` times and keeps only samples that can be mapped to the
    biological alignment length.
    """
    sampled_batches = []
    sampled_sequence_batches = []
    sampled_count = 0
    attempts = 0
    max_attempts = 20

    while sampled_count < cij_sample_size and attempts < max_attempts:
        attempts += 1
        sequences, _, _ = generate_block(
            model,
            beta_sampling=1,
            N=cij_sample_size - sampled_count,
            num_tokens=sequence_length + 1,
            max_context=block_size,
            device=device,
            bos_eos_token=stoi["\n"],
            pad_token=pad_token,
        )
        one_hot = tokens_to_alignment_one_hot(
            sequences,
            valid_token_ids,
            token_remap,
            sequence_length,
            device,
            return_sequences=True,
        )
        one_hot, selected_sequences = one_hot
        if one_hot is not None:
            sampled_batches.append(one_hot)
            sampled_sequence_batches.append(selected_sequences)
            sampled_count += one_hot.shape[0]

    if not sampled_batches:
        return None, None

    return (
        torch.cat(sampled_batches, dim=0)[:cij_sample_size],
        torch.cat(sampled_sequence_batches, dim=0)[:cij_sample_size],
    )


def evaluate_validation_loss(model, datasets, batch_size, block_size, pad_token, device):
    """Compute token-weighted validation loss over the full validation split."""
    val_data = datasets["val"]["data"]
    total_loss = 0.0
    total_tokens = 0

    for start in range(0, len(val_data), batch_size):
        batch = torch.stack(val_data[start : start + batch_size], dim=0)
        x = batch[:, :block_size].to(device)
        y = batch[:, 1 : block_size + 1]
        if y.size(1) < x.size(1):
            pad_cols = torch.full(
                (y.size(0), x.size(1) - y.size(1)),
                pad_token,
                dtype=y.dtype,
            )
            y = torch.cat([y, pad_cols], dim=1)
        y = y[:, : x.size(1)].to(device)

        loss = model(x, targets=y)["loss"]
        token_count = (y != pad_token).sum().item()
        total_loss += loss.item() * token_count
        total_tokens += token_count

    if total_tokens == 0:
        raise ValueError("Validation set has no non-padding target tokens.")
    return total_loss / total_tokens


# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------
# Encode each FASTA sequence, add beginning/end markers inside load_dataset,
# pad or truncate to block_size, and split the data into train/validation sets.
with redirect_stdout(StringIO()):
    DN = load_dataset(
        seq_path=TRAIN_DN_path,
        validation_seq_path=VALIDATION_DN_path,
        label_path=".",
        block_size=block_size,
        train_size=None,
        encode_fn=encode,
        pad_token=pad_token,
        shuffle=True,
        reweighting=reweighting,
        reweighting_th=reweighting_th,
    )

# -----------------------------------------------------------------------------
# Checkpoint and training log paths
# -----------------------------------------------------------------------------
# Create a checkpoint folder under checkpoints/CM. The run folder name encodes
# the Transformer size so checkpoints from different architectures stay separate.
checkpoint_root = os.path.join("checkpoints", "CM")
run_name = f"{model_name}_L{n_layer}_H{n_head}_E{n_embd}"
checkpoint_dir = os.path.join(checkpoint_root, run_name)
os.makedirs(checkpoint_dir, exist_ok=True)
log_path = os.path.join(os.getcwd(), checkpoint_dir, "training_log.txt")
set_logger(log_path)
valid_token_ids = [stoi[ch] for ch in chars if ch not in {PAD_TOKEN, "\n"}]
token_remap = {token_id: idx for idx, token_id in enumerate(valid_token_ids)}
logging.info(
    f"RUN model={run_name} device={device} dtype={dtype} train={len(DN['train']['data'])} "
    f"val={len(DN['val']['data'])} L={sequence_length} q={len(valid_token_ids)} "
    f"batch={batch_size} max_iters={max_iters} lr={learning_rate} "
    f"eval_interval={eval_interval} eval_iters={eval_iters} "
    f"reweighting={reweighting} reweighting_th={reweighting_th} "
    f"compute_sampling_metrics={compute_sampling_metrics} cij_sample={cij_sample_size} out={checkpoint_dir}"
)
train_weights = torch.tensor(DN["train"]["weights"], dtype=torch.float32, device=device)
if reweighting and torch.any(train_weights == 0):
    raise ValueError("Reweighting is enabled, but train_weights contains zero-valued entries.")
logging.info(f"WEIGHTS train_Meff={train_weights.sum().item():.3f}")
if compute_sampling_metrics:
    natural_train = tokens_to_alignment_one_hot(
        DN["train"]["data"],
        valid_token_ids,
        token_remap,
        sequence_length,
        device,
    )
    if natural_train is None:
        raise ValueError("No valid fixed-length training sequences found for Cij evaluation.")
    cij_warmup = natural_train[: min(cij_warmup_size, natural_train.shape[0])]
    cij_warmup_weights = train_weights[:cij_warmup.shape[0]] if reweighting else None
    natural_weights = train_weights if reweighting else None
    fi_warmup = get_freq_single_point(data=cij_warmup, weights=cij_warmup_weights, pseudo_count=0.0)
    fij_warmup = get_freq_two_points(data=cij_warmup, weights=cij_warmup_weights, pseudo_count=0.0)
    _ = get_correlation_two_points(
        fi=fi_warmup,
        pi=fi_warmup,
        fij=fij_warmup,
        pij=fij_warmup,
    )
    fi_natural = get_freq_single_point(data=natural_train, weights=natural_weights, pseudo_count=0.0)
    fij_natural = get_freq_two_points(data=natural_train, weights=natural_weights, pseudo_count=0.0)
    logging.info(
        f"REF_CIJ natural={natural_train.shape[0]} warmup={cij_warmup.shape[0]} L={natural_train.shape[1]} "
        f"q={natural_train.shape[2]}"
    )

# -----------------------------------------------------------------------------
# Model initialization
# -----------------------------------------------------------------------------
# Instantiate the GPT-style Transformer with the selected vocabulary size and
# architecture, move it to the target device, compile it, and configure the
# attention backend.
model = GPTTransformer(
    vocab_size=vocab_size,
    embed_dim=n_embd,
    num_layers=n_layer,
    num_heads=n_head,
    mlp_ratio=4,
    dropout_p=0.1,
    pad_id=pad_token,
)
model.to(device)
model = torch.compile(model)  # initial overhead for GPU warmup, but faster overall
set_attention_backend(model, ptdtype)

num_params = sum(p.numel() for p in model.parameters()) / 1e6

# -----------------------------------------------------------------------------
# Optimizer and learning-rate scheduler
# -----------------------------------------------------------------------------
# AdamW updates the model weights. StepLR reduces the learning rate by gamma
# after scheduler_step_size optimizer steps.
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=0.1)
logging.info(f"MODEL params_M={num_params:.3f}")

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
# Repeatedly evaluate on the validation split, save checkpoints, sample training
# batches, compute next-token prediction loss, backpropagate, update the model,
# and advance the learning-rate scheduler.
best_val_loss = float("inf")
early_stopping_counter = 0
best_ckpt_path = None

for iter in tqdm(range(max_iters), desc="Training transformer", dynamic_ncols=True):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        model.eval()
        eval_loss = 0.0

        with torch.inference_mode():
            if eval_iters is None:
                eval_loss = evaluate_validation_loss(
                    model,
                    DN,
                    batch_size,
                    block_size,
                    pad_token,
                    device,
                )
            else:
                for i in range(eval_iters):
                    x, y, _, _ = get_batch("val", DN, batch_size, block_size, pad_token, device)
                    loss = model(x, targets=y)["loss"]
                    eval_loss += loss.item()
                eval_loss /= eval_iters

        if compute_sampling_metrics:
            sampled, sampled_sequences = sample_alignment_one_hot(model, valid_token_ids, token_remap)
            if sampled is None:
                pearson_cij = float("nan")
                entropy_nats = float("nan")
                sampled_count = 0
            else:
                sampled_count = sampled.shape[0]
                entropy_nats = -score_sequences(
                    sampled_sequences,
                    model,
                    pad_token=pad_token,
                ).mean().item()
                pi_sampled = get_freq_single_point(data=sampled, weights=None, pseudo_count=0.0)
                pij_sampled = get_freq_two_points(data=sampled, weights=None, pseudo_count=0.0)
                pearson_cij, _ = get_correlation_two_points(
                    fi=fi_natural,
                    pi=pi_sampled,
                    fij=fij_natural,
                    pij=pij_sampled,
                )

        original_model = model._orig_mod
        ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_iter{iter}.pt")
        torch.save(original_model.state_dict(), ckpt_path)
        if compute_sampling_metrics:
            logging.info(
                f"EVAL step={iter} val_loss={eval_loss:.4f} cij_pearson={pearson_cij:.4f} "
                f"entropy_nats={entropy_nats:.4f} sampled={sampled_count} ckpt={ckpt_path}"
            )
        else:
            improved = eval_loss < best_val_loss - early_stopping_min_delta
            if improved:
                best_val_loss = eval_loss
                early_stopping_counter = 0
                best_ckpt_path = ckpt_path
            else:
                early_stopping_counter += 1

            logging.info(
                f"EVAL step={iter} val_loss={eval_loss:.4f} best_val_loss={best_val_loss:.4f} "
                f"patience={early_stopping_counter}/{early_stopping_patience} ckpt={ckpt_path}"
            )
            if early_stopping_counter >= early_stopping_patience:
                logging.info(
                    f"EARLY_STOP step={iter} best_val_loss={best_val_loss:.4f} "
                    f"best_ckpt={best_ckpt_path}"
                )
                break

    model.train()
    if reweighting:
        xb, yb, pad_xb, Rb, Wb = get_batch(
            "train",
            DN,
            batch_size,
            block_size,
            pad_token,
            device,
            return_weights=True,
        )
    else:
        xb, yb, pad_xb, Rb = get_batch("train", DN, batch_size, block_size, pad_token, device)

    optimizer.zero_grad(set_to_none=True)
    with ctx:
        if reweighting:
            output = model(xb)
            logits = output["logits"]
            token_loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                yb.reshape(-1),
                ignore_index=pad_token,
                reduction="none",
            ).view(yb.shape)
            token_mask = (yb != pad_token).to(token_loss.dtype)
            loss = (token_loss * Wb[:, None]).sum() / (token_mask * Wb[:, None]).sum().clamp(min=1.0)
        else:
            output = model(xb, targets=yb)
            loss = output["loss"]  # expects model output as a dict with key "loss"

    if iter % log_steps == 0:
        current_lr = scheduler.get_last_lr()[0]
        logging.info(
            f"TRAIN step={iter} train_loss={loss.item():.4f} lr={current_lr:.6g}"
        )

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

logging.info("DONE")

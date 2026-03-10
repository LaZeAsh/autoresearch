"""
Autoresearch VLA training script.

This is the single mutable research surface for the VLA fork. The dataset
contract, preprocessing, splits, and evaluation live in prepare.py.

Usage:
    uv run train.py
"""

from __future__ import annotations

import contextlib
import gc
import math
import os
import time
from dataclasses import asdict, dataclass

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    DEFAULT_EVAL_BATCHES,
    INSTRUCTION_VOCAB_SIZE,
    MAX_SEQ_LEN,
    TIME_BUDGET,
    build_vla_runtime,
    cycle,
    evaluate_vla,
    format_eval_summary,
    move_batch_to_device,
)


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

# Data / context
HISTORY_FRAMES = 4
FRAME_STRIDE = 1
ACTION_CHUNK_SIZE = 4
PAST_ACTION_CHUNK_SIZE = 4
MAX_INSTRUCTION_TOKENS = 64
USE_STATE = True
USE_PAST_ACTIONS = True

# Vision encoder
TOKENS_PER_FRAME = 16  # 4x4 spatial grid per frame

# Context budget: visual + text tokens
MAX_CONTEXT_TOKENS = HISTORY_FRAMES * TOKENS_PER_FRAME + MAX_INSTRUCTION_TOKENS  # 128

# Trainable action policy
DEPTH = 6
N_HEAD = 8
N_EMBD = 512
MLP_RATIO = 4.0
DROPOUT = 0.0

# Optimization
DEVICE_BATCH_SIZE = 32
TOTAL_BATCH_SIZE = 32
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05
ADAM_BETAS = (0.9, 0.95)
GRAD_CLIP_NORM = 1.0
WAYPOINT_LOSS_WEIGHT = 0.25
WARMUP_RATIO = 0.05
WARMDOWN_RATIO = 0.50
FINAL_LR_FRAC = 0.10
USE_COMPILE = False
NUM_WORKERS = 0
PIN_MEMORY = True

# Evaluation
VAL_EVAL_BATCHES = DEFAULT_EVAL_BATCHES


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    max_seq_len: int
    max_context_tokens: int
    action_vocab_size: int
    history_frames: int
    tokens_per_frame: int
    image_channels: int
    image_size: int
    max_instruction_tokens: int
    use_state: bool
    state_dim: int
    use_past_actions: bool
    past_action_token_count: int
    action_dim: int
    action_chunk_size: int
    waypoint_dim: int
    n_layer: int
    n_head: int
    n_embd: int
    mlp_ratio: float
    dropout: float

    @property
    def target_action_token_count(self) -> int:
        return self.action_dim * self.action_chunk_size

    @property
    def state_token_count(self) -> int:
        return 1 if self.use_state and self.state_dim > 0 else 0

    @property
    def total_sequence_len(self) -> int:
        return self.max_context_tokens + self.state_token_count + self.past_action_token_count + self.target_action_token_count


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),), self.weight)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("N_EMBD must be divisible by N_HEAD")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = int(config.n_embd * config.mlp_ratio)
        self.gate = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.silu(self.gate(x)) * self.fc(x)))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """Lightweight CNN: each frame -> spatial patch tokens."""

    def __init__(self, image_channels: int, image_size: int, n_embd: int, tokens_per_frame: int):
        super().__init__()
        self.tokens_per_frame = tokens_per_frame
        grid_size = int(tokens_per_frame ** 0.5)  # 4 for 16 tokens
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, 7, stride=4, padding=3),   # 256->64
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),              # 64->32
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),             # 32->16
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),            # 16->8
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(grid_size),                         # 8->4
            nn.Conv2d(256, n_embd, 1),                               # channel proj
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = frames.shape
        x = frames.reshape(B * T, C, H, W)
        x = self.encoder(x)
        x = x.flatten(2).transpose(1, 2)  # [B*T, tokens_per_frame, n_embd]
        return x.reshape(B, T * self.tokens_per_frame, x.size(-1))


class LightweightActionPolicy(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        if config.total_sequence_len > config.max_seq_len:
            raise ValueError(
                f"Configured sequence length {config.total_sequence_len} exceeds MAX_SEQ_LEN={config.max_seq_len}"
            )

        self.vision_encoder = VisionEncoder(
            config.image_channels, config.image_size, config.n_embd, config.tokens_per_frame,
        )
        self.instruction_embed = nn.Embedding(INSTRUCTION_VOCAB_SIZE, config.n_embd, padding_idx=0)
        self.state_proj = (
            nn.Sequential(
                nn.Linear(config.state_dim, config.n_embd, bias=False),
                nn.GELU(),
                nn.Linear(config.n_embd, config.n_embd, bias=False),
            )
            if config.state_dim > 0 and config.use_state
            else None
        )
        self.action_embed = nn.Embedding(config.action_vocab_size, config.n_embd, padding_idx=0)
        self.type_embed = nn.Embedding(4, config.n_embd)  # 0=context, 1=state, 2=past_action, 3=future_action
        self.frame_temporal_embed = nn.Parameter(torch.zeros(config.history_frames, 1, config.n_embd))
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, config.n_embd))
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.final_norm = RMSNorm(config.n_embd)
        self.action_head = nn.Linear(config.n_embd, config.action_vocab_size, bias=False)
        self.waypoint_head = (
            nn.Linear(config.n_embd, config.waypoint_dim, bias=False) if config.waypoint_dim > 0 else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        def init_module(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.vision_encoder.apply(init_module)
        if self.state_proj is not None:
            self.state_proj.apply(init_module)
        self.blocks.apply(init_module)
        self.action_head.apply(init_module)
        if self.waypoint_head is not None:
            self.waypoint_head.apply(init_module)
        nn.init.normal_(self.instruction_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.action_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.type_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.frame_temporal_embed, mean=0.0, std=0.01)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.01)

    def _build_attention_mask(
        self,
        context_mask: torch.Tensor,
        state_tokens: torch.Tensor | None,
        past_action_tokens: torch.Tensor | None,
        future_action_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = context_mask.size(0)
        device = context_mask.device
        parts = [context_mask.bool()]
        if state_tokens is not None:
            parts.append(torch.ones(batch_size, state_tokens.size(1), dtype=torch.bool, device=device))
        if past_action_tokens is not None:
            parts.append(torch.ones(batch_size, past_action_tokens.size(1), dtype=torch.bool, device=device))
        parts.append(torch.ones(batch_size, future_action_tokens.size(1), dtype=torch.bool, device=device))
        keep_mask = torch.cat(parts, dim=1)
        seq_len = keep_mask.size(1)
        causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
        return causal.unsqueeze(0).unsqueeze(1) & keep_mask[:, None, None, :]

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | None]:
        target_action_tokens = batch["target_action_tokens"]
        past_action_tokens = batch["past_action_tokens"]
        batch_size = target_action_tokens.size(0)
        target_len = target_action_tokens.size(1)
        device = self.pos_embed.device

        # Encode frames (CPU -> GPU)
        frames = batch["frames"].to(device)
        visual_tokens = self.vision_encoder(frames)  # [B, T*tpf, n_embd]
        visual_tokens = visual_tokens + self.type_embed.weight[0]
        # Add per-frame temporal embedding
        tpf = self.config.tokens_per_frame
        temporal = self.frame_temporal_embed.expand(-1, tpf, -1).reshape(1, -1, self.config.n_embd)
        visual_tokens = visual_tokens + temporal

        # Encode instructions (CPU -> GPU)
        instruction_tokens = batch["instruction_tokens"].to(device)
        instruction_mask = batch["instruction_mask"].to(device)
        text_tokens = self.instruction_embed(instruction_tokens)
        text_tokens = text_tokens + self.type_embed.weight[0]

        # Combine context
        context_tokens = torch.cat([visual_tokens, text_tokens], dim=1)
        visual_mask = torch.ones(batch_size, visual_tokens.size(1), dtype=torch.bool, device=device)
        context_mask = torch.cat([visual_mask, instruction_mask], dim=1)

        # State
        state_tokens = None
        if self.state_proj is not None and batch["states"].numel() > 0:
            state_tokens = self.state_proj(batch["states"]).unsqueeze(1)
            state_tokens = state_tokens + self.type_embed.weight[1]

        # Past actions
        past_tokens = None
        if self.config.use_past_actions and past_action_tokens.numel() > 0:
            past_tokens = self.action_embed(past_action_tokens)
            past_tokens = past_tokens + self.type_embed.weight[2]

        # Future actions (teacher forcing)
        bos = torch.full(
            (batch_size, 1),
            fill_value=1,
            dtype=torch.long,
            device=target_action_tokens.device,
        )
        teacher_tokens = torch.cat([bos, target_action_tokens[:, :-1]], dim=1)
        future_tokens = self.action_embed(teacher_tokens)
        future_tokens = future_tokens + self.type_embed.weight[3]

        # Assemble full sequence
        sequence_parts = [context_tokens]
        if state_tokens is not None:
            sequence_parts.append(state_tokens)
        if past_tokens is not None:
            sequence_parts.append(past_tokens)
        sequence_parts.append(future_tokens)
        x = torch.cat(sequence_parts, dim=1)

        if x.size(1) > self.config.max_seq_len:
            raise ValueError(f"Sequence length {x.size(1)} exceeds configured maximum {self.config.max_seq_len}")

        x = x + self.pos_embed[:, : x.size(1)]
        attn_mask = self._build_attention_mask(context_mask, state_tokens, past_tokens, future_tokens)
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.final_norm(x)

        action_hidden = x[:, -target_len:]
        action_logits = self.action_head(action_hidden).float()

        waypoint_preds = None
        if self.waypoint_head is not None:
            pooled = action_hidden.view(batch_size, self.config.action_chunk_size, self.config.action_dim, -1)
            pooled = pooled.mean(dim=2)
            waypoint_preds = self.waypoint_head(pooled).float()

        return {
            "action_logits": action_logits,
            "waypoint_preds": waypoint_preds,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def masked_action_logits(logits: torch.Tensor, action_tokenizer) -> torch.Tensor:
    mask = action_tokenizer.valid_token_mask(logits.size(1), logits.device)
    return logits.masked_fill(~mask.unsqueeze(0), torch.finfo(logits.dtype).min)


def compute_loss(outputs: dict[str, torch.Tensor | None], batch: dict[str, torch.Tensor], action_tokenizer) -> tuple[torch.Tensor, dict[str, float]]:
    logits = masked_action_logits(outputs["action_logits"], action_tokenizer)
    targets = batch["target_action_tokens"]
    action_ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="mean",
    )

    waypoint_loss = torch.zeros((), device=logits.device)
    waypoint_preds = outputs.get("waypoint_preds")
    waypoint_targets = batch["waypoints"]
    if waypoint_preds is not None and waypoint_targets.numel() > 0 and waypoint_targets.size(-1) > 0:
        waypoint_loss = F.mse_loss(waypoint_preds, waypoint_targets, reduction="mean")

    total_loss = action_ce + WAYPOINT_LOSS_WEIGHT * waypoint_loss
    metrics = {
        "action_ce": action_ce.detach().item(),
        "waypoint_loss": waypoint_loss.detach().item() if waypoint_loss.numel() > 0 else 0.0,
    }
    return total_loss, metrics


def get_lr_multiplier(progress: float) -> float:
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    return cooldown + (1.0 - cooldown) * FINAL_LR_FRAC


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def build_model_config(runtime) -> ModelConfig:
    metadata = runtime.metadata
    config = ModelConfig(
        max_seq_len=MAX_SEQ_LEN,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        action_vocab_size=runtime.action_tokenizer.get_vocab_size(),
        history_frames=HISTORY_FRAMES,
        tokens_per_frame=TOKENS_PER_FRAME,
        image_channels=metadata.image_channels,
        image_size=metadata.image_size,
        max_instruction_tokens=MAX_INSTRUCTION_TOKENS,
        use_state=USE_STATE and metadata.has_state,
        state_dim=metadata.state_dim if USE_STATE else 0,
        use_past_actions=USE_PAST_ACTIONS,
        past_action_token_count=(PAST_ACTION_CHUNK_SIZE * metadata.action_dim) if USE_PAST_ACTIONS else 0,
        action_dim=metadata.action_dim,
        action_chunk_size=ACTION_CHUNK_SIZE,
        waypoint_dim=metadata.waypoint_dim,
        n_layer=DEPTH,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
    )
    if config.total_sequence_len > MAX_SEQ_LEN:
        raise ValueError(
            f"Model sequence length {config.total_sequence_len} exceeds MAX_SEQ_LEN={MAX_SEQ_LEN}. "
            "Reduce MAX_CONTEXT_TOKENS, ACTION_CHUNK_SIZE, or PAST_ACTION_CHUNK_SIZE."
        )
    return config


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda_amp = device.type == "cuda"


def autocast_context():
    return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_cuda_amp else contextlib.nullcontext()


runtime = build_vla_runtime(
    batch_size=DEVICE_BATCH_SIZE,
    history_frames=HISTORY_FRAMES,
    frame_stride=FRAME_STRIDE,
    action_chunk_size=ACTION_CHUNK_SIZE,
    past_action_chunk_size=PAST_ACTION_CHUNK_SIZE,
    max_instruction_tokens=MAX_INSTRUCTION_TOKENS,
    include_state=USE_STATE,
    include_past_actions=USE_PAST_ACTIONS,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

config = build_model_config(runtime)
print(f"Model config: {asdict(config)}")
print(f"Train windows: {len(runtime.train_dataset):,}")
print(f"Val windows:   {len(runtime.val_dataset):,}")

if TOTAL_BATCH_SIZE % DEVICE_BATCH_SIZE != 0:
    raise ValueError("TOTAL_BATCH_SIZE must be divisible by DEVICE_BATCH_SIZE")
grad_accum_steps = TOTAL_BATCH_SIZE // DEVICE_BATCH_SIZE

model = LightweightActionPolicy(config).to(device)
if USE_COMPILE:
    model = torch.compile(model, dynamic=False)
num_params = count_parameters(model)
trainable_params = count_parameters(model, trainable_only=True)

optimizer_kwargs = {
    "lr": LEARNING_RATE,
    "betas": ADAM_BETAS,
    "weight_decay": WEIGHT_DECAY,
}
if device.type == "cuda":
    optimizer_kwargs["fused"] = True
optimizer = torch.optim.AdamW(
    [parameter for parameter in model.parameters() if parameter.requires_grad],
    **optimizer_kwargs,
)
for group in optimizer.param_groups:
    group["initial_lr"] = group["lr"]

train_batches = cycle(runtime.train_loader)
batch_cpu, epoch = next(train_batches)

print(f"Device: {device}")
print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_total_loss = 0.0
smooth_action_ce = 0.0
total_training_time = 0.0
step = 0
total_windows = 0

while True:
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    micro_action_ce = 0.0
    micro_total_loss = 0.0
    for _ in range(grad_accum_steps):
        batch = move_batch_to_device(batch_cpu, device)
        with autocast_context():
            outputs = model(batch)
            loss, loss_metrics = compute_loss(outputs, batch, runtime.action_tokenizer)
        (loss / grad_accum_steps).backward()
        micro_total_loss += loss.detach().item()
        micro_action_ce += loss_metrics["action_ce"]
        total_windows += batch["frames"].size(0)
        batch_cpu, epoch = next(train_batches)

    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lr_multiplier = get_lr_multiplier(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lr_multiplier

    if GRAD_CLIP_NORM > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    if step > 2:
        total_training_time += dt

    train_total_loss = micro_total_loss / grad_accum_steps
    train_action_ce = micro_action_ce / grad_accum_steps
    if not math.isfinite(train_total_loss):
        raise RuntimeError(f"Training diverged with non-finite loss: {train_total_loss}")

    ema_beta = 0.9
    smooth_total_loss = ema_beta * smooth_total_loss + (1 - ema_beta) * train_total_loss
    smooth_action_ce = ema_beta * smooth_action_ce + (1 - ema_beta) * train_action_ce
    debiased_total = smooth_total_loss / (1 - ema_beta ** (step + 1))
    debiased_ce = smooth_action_ce / (1 - ema_beta ** (step + 1))

    windows_per_sec = TOTAL_BATCH_SIZE / max(dt, 1e-6)
    remaining = max(0.0, TIME_BUDGET - total_training_time)
    print(
        f"\rstep {step:05d} ({progress * 100:.1f}%) | "
        f"loss: {debiased_total:.4f} | action_ce: {debiased_ce:.4f} | "
        f"lr_mult: {lr_multiplier:.2f} | dt: {dt * 1000:.0f}ms | "
        f"win/sec: {windows_per_sec:.1f} | epoch: {epoch} | remaining: {remaining:.0f}s    ",
        end="",
        flush=True,
    )

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 2000 == 0:
        gc.collect()

    step += 1
    if step > 2 and total_training_time >= TIME_BUDGET:
        break

print()


# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

model.eval()
with autocast_context():
    metrics = evaluate_vla(
        model=model,
        val_loader=runtime.val_loader,
        action_tokenizer=runtime.action_tokenizer,
        metadata=runtime.metadata,
        device=device,
        max_batches=VAL_EVAL_BATCHES,
    )

t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0.0
windows_per_second = total_windows / max(total_training_time, 1e-6)

print("---")
for line in format_eval_summary(metrics):
    print(line)
print(f"training_seconds:   {total_training_time:.1f}")
print(f"total_seconds:      {t_end - t_start:.1f}")
print(f"peak_vram_mb:       {peak_vram_mb:.1f}")
print(f"windows_per_second: {windows_per_second:.1f}")
print(f"total_windows:      {total_windows}")
print(f"num_steps:          {step}")
print(f"num_params_M:       {num_params / 1e6:.1f}")
print(f"trainable_params_M: {trainable_params / 1e6:.1f}")
print(f"depth:              {DEPTH}")

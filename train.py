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

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    DEFAULT_EVAL_BATCHES,
    MAX_SEQ_LEN,
    TIME_BUDGET,
    build_vla_runtime,
    cycle,
    move_batch_to_device,
)


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

# Data / context
HISTORY_FRAMES = 4
FRAME_STRIDE = 1
ACTION_CHUNK_SIZE = 4
PAST_ACTION_CHUNK_SIZE = 1
MAX_INSTRUCTION_TOKENS = 64
USE_STATE = True
USE_PAST_ACTIONS = True
MODEL_IMAGE_SIZE = 96

# Vision / fusion
PATCH_SIZE = 12

# Model size
DEPTH = 8
N_HEAD = 8
N_EMBD = 512
MLP_RATIO = 4.0
DROPOUT = 0.0

# Optimization
DEVICE_BATCH_SIZE = 16
TOTAL_BATCH_SIZE = 128
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

# Flow matching
OBJECTIVE = "flow_matching"
ACTION_NORMALIZATION = "bounds"
ACTION_NORM_EPS = 1e-3
FLOW_LOSS_WEIGHT = 1.0
FLOW_T_MIN = 1e-3
FLOW_T_MAX = 1.0 - 1e-3
TIME_EMBED_DIM = 128

# Evaluation
VAL_EVAL_BATCHES = DEFAULT_EVAL_BATCHES


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    max_seq_len: int
    instruction_vocab_size: int
    action_vocab_size: int
    image_channels: int
    model_image_size: int
    history_frames: int
    patch_size: int
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
    time_embed_dim: int

    @property
    def target_action_token_count(self) -> int:
        return self.action_dim * self.action_chunk_size

    @property
    def state_token_count(self) -> int:
        return 1 if self.use_state and self.state_dim > 0 else 0

    @property
    def num_visual_tokens(self) -> int:
        if self.model_image_size % self.patch_size != 0:
            raise ValueError("MODEL_IMAGE_SIZE must be divisible by PATCH_SIZE")
        return (self.model_image_size // self.patch_size) ** 2

    @property
    def total_sequence_len(self) -> int:
        return (
            self.max_instruction_tokens
            + self.num_visual_tokens
            + self.state_token_count
            + self.past_action_token_count
            + self.target_action_token_count
        )


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
        self.fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.proj(x)
        return self.dropout(x)


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


class VLATransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        if config.total_sequence_len > config.max_seq_len:
            raise ValueError(
                f"Configured sequence length {config.total_sequence_len} exceeds MAX_SEQ_LEN={config.max_seq_len}"
            )

        self.instruction_embed = nn.Embedding(config.instruction_vocab_size, config.n_embd, padding_idx=0)
        self.action_embed = nn.Embedding(config.action_vocab_size, config.n_embd, padding_idx=0)
        self.visual_patch_embed = nn.Conv2d(
            in_channels=config.image_channels * config.history_frames,
            out_channels=config.n_embd,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.state_proj = (
            nn.Sequential(
                nn.Linear(config.state_dim, config.n_embd, bias=False),
                nn.GELU(),
                nn.Linear(config.n_embd, config.n_embd, bias=False),
            )
            if config.state_dim > 0 and config.use_state
            else None
        )
        self.future_action_proj = nn.Sequential(
            nn.Linear(1, config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd, bias=False),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(config.time_embed_dim, config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd, bias=False),
        )
        self.type_embed = nn.Embedding(5, config.n_embd)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, config.n_embd))
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.final_norm = RMSNorm(config.n_embd)
        self.flow_head = nn.Linear(config.n_embd, 1, bias=False)
        self.waypoint_head = (
            nn.Linear(config.n_embd, config.waypoint_dim, bias=False) if config.waypoint_dim > 0 else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.01)

    def _resize_frames(self, frames: torch.Tensor) -> torch.Tensor:
        batch_size, history_frames, channels, height, width = frames.shape
        frames = frames.reshape(batch_size, history_frames * channels, height, width)
        if height != self.config.model_image_size or width != self.config.model_image_size:
            frames = F.interpolate(
                frames,
                size=(self.config.model_image_size, self.config.model_image_size),
                mode="bilinear",
                align_corners=False,
            )
        return frames

    def _build_attention_mask(
        self,
        instruction_mask: torch.Tensor,
        visual_tokens: torch.Tensor,
        state_tokens: torch.Tensor | None,
        past_action_tokens: torch.Tensor | None,
        future_action_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = instruction_mask.size(0)
        device = instruction_mask.device
        parts = [
            instruction_mask.bool(),
            torch.ones(batch_size, visual_tokens.size(1), dtype=torch.bool, device=device),
        ]
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
        instruction_tokens = batch["instruction_tokens"]
        instruction_mask = batch["instruction_mask"]
        frames = batch["frames"]
        past_action_tokens = batch["past_action_tokens"]
        flow_noisy_actions = batch["flow_noisy_actions"]
        flow_t = batch["flow_t"]

        batch_size = instruction_tokens.size(0)
        target_len = flow_noisy_actions.size(1)

        instruction = self.instruction_embed(instruction_tokens)
        instruction = instruction + self.type_embed.weight[0]

        visual_input = self._resize_frames(frames)
        visual_tokens = self.visual_patch_embed(visual_input).flatten(2).transpose(1, 2)
        visual_tokens = visual_tokens + self.type_embed.weight[1]

        state_tokens = None
        if self.state_proj is not None and batch["states"].numel() > 0:
            state_tokens = self.state_proj(batch["states"]).unsqueeze(1)
            state_tokens = state_tokens + self.type_embed.weight[2]

        past_tokens = None
        if self.config.use_past_actions and past_action_tokens.numel() > 0:
            past_tokens = self.action_embed(past_action_tokens)
            past_tokens = past_tokens + self.type_embed.weight[3]

        if target_len != self.config.target_action_token_count:
            raise ValueError(
                f"Expected {self.config.target_action_token_count} future action slots, got {target_len}"
            )
        future_tokens = self.future_action_proj(flow_noisy_actions.unsqueeze(-1))
        time_embed = timestep_embedding(flow_t, self.config.time_embed_dim)
        time_embed = self.time_proj(time_embed).unsqueeze(1)
        future_tokens = future_tokens + time_embed + self.type_embed.weight[4]

        sequence_parts = [instruction, visual_tokens]
        if state_tokens is not None:
            sequence_parts.append(state_tokens)
        if past_tokens is not None:
            sequence_parts.append(past_tokens)
        sequence_parts.append(future_tokens)
        x = torch.cat(sequence_parts, dim=1)

        if x.size(1) > self.config.max_seq_len:
            raise ValueError(f"Sequence length {x.size(1)} exceeds configured maximum {self.config.max_seq_len}")

        x = x + self.pos_embed[:, : x.size(1)]
        attn_mask = self._build_attention_mask(instruction_mask, visual_tokens, state_tokens, past_tokens, future_tokens)
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.final_norm(x)

        action_hidden = x[:, -target_len:]
        pred_velocity = self.flow_head(action_hidden).squeeze(-1).float()

        waypoint_preds = None
        if self.waypoint_head is not None:
            pooled = action_hidden.view(batch_size, self.config.action_chunk_size, self.config.action_dim, -1)
            pooled = pooled.mean(dim=2)
            waypoint_preds = self.waypoint_head(pooled).float()

        return {
            "pred_velocity": pred_velocity,
            "waypoint_preds": waypoint_preds,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    if dim <= 0:
        raise ValueError("TIME_EMBED_DIM must be positive")
    half_dim = dim // 2
    if half_dim == 0:
        return timesteps.unsqueeze(1)
    exponent = -math.log(10_000.0) * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
    exponent = exponent / max(half_dim - 1, 1)
    freqs = exponent.exp()
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    embedding = torch.cat([args.cos(), args.sin()], dim=1)
    if dim % 2 != 0:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ActionNormalizer:
    def __init__(self, low: torch.Tensor, high: torch.Tensor, eps: float = ACTION_NORM_EPS):
        if low.shape != high.shape:
            raise ValueError("Action normalization bounds must have matching shapes")
        self.low = low
        self.high = high
        self.center = (low + high) * 0.5
        raw_scale = (high - low) * 0.5
        self.scale = torch.where(raw_scale.abs() >= eps, raw_scale, torch.ones_like(raw_scale))

    @classmethod
    def from_metadata(cls, metadata, device: torch.device) -> "ActionNormalizer":
        if ACTION_NORMALIZATION != "bounds":
            raise ValueError(f"Unsupported ACTION_NORMALIZATION={ACTION_NORMALIZATION!r}")
        low = torch.tensor(metadata.action_low, dtype=torch.float32, device=device)
        high = torch.tensor(metadata.action_high, dtype=torch.float32, device=device)
        return cls(low=low, high=high)

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.size(-1) != self.center.numel():
            raise ValueError(f"Expected action dim {self.center.numel()}, got {actions.size(-1)}")
        return (actions - self.center) / self.scale

    def denormalize(self, normalized_actions: torch.Tensor) -> torch.Tensor:
        if normalized_actions.size(-1) != self.center.numel():
            raise ValueError(f"Expected action dim {self.center.numel()}, got {normalized_actions.size(-1)}")
        return normalized_actions * self.scale + self.center


def prepare_flow_matching_batch(
    batch: dict[str, torch.Tensor],
    action_normalizer: ActionNormalizer,
) -> dict[str, torch.Tensor]:
    normalized_targets = action_normalizer.normalize(batch["action_targets"])
    target_slots = normalized_targets.view(normalized_targets.size(0), -1)
    noise = torch.randn_like(target_slots)
    t = torch.rand(target_slots.size(0), device=target_slots.device)
    t = t.mul(FLOW_T_MAX - FLOW_T_MIN).add(FLOW_T_MIN)
    noisy_actions = torch.lerp(noise, target_slots, t.unsqueeze(1))

    flow_batch = batch.copy()
    flow_batch["normalized_action_targets"] = target_slots
    flow_batch["flow_target"] = target_slots - noise
    flow_batch["flow_t"] = t
    flow_batch["flow_noisy_actions"] = noisy_actions
    return flow_batch


def reconstruct_normalized_actions(
    noisy_actions: torch.Tensor,
    pred_velocity: torch.Tensor,
    flow_t: torch.Tensor,
) -> torch.Tensor:
    return noisy_actions + (1.0 - flow_t.unsqueeze(1)) * pred_velocity


def compute_loss(
    outputs: dict[str, torch.Tensor | None],
    batch: dict[str, torch.Tensor],
    action_normalizer: ActionNormalizer,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_velocity = outputs["pred_velocity"]
    flow_target = batch["flow_target"]
    flow_loss = F.mse_loss(pred_velocity, flow_target, reduction="mean")
    pred_actions_normalized = reconstruct_normalized_actions(
        noisy_actions=batch["flow_noisy_actions"],
        pred_velocity=pred_velocity,
        flow_t=batch["flow_t"],
    )

    target_actions = batch["action_targets"]
    pred_actions = action_normalizer.denormalize(pred_actions_normalized.view_as(target_actions))
    action_mse = F.mse_loss(pred_actions, target_actions, reduction="mean")

    waypoint_loss = torch.zeros((), device=pred_velocity.device)
    waypoint_preds = outputs.get("waypoint_preds")
    waypoint_targets = batch["waypoints"]
    if waypoint_preds is not None and waypoint_targets.numel() > 0 and waypoint_targets.size(-1) > 0:
        waypoint_loss = F.mse_loss(waypoint_preds, waypoint_targets, reduction="mean")

    total_loss = FLOW_LOSS_WEIGHT * flow_loss + WAYPOINT_LOSS_WEIGHT * waypoint_loss
    metrics = {
        "flow_mse": flow_loss.detach().item(),
        "action_mse": action_mse.detach().item(),
        "waypoint_loss": waypoint_loss.detach().item() if waypoint_loss.numel() > 0 else 0.0,
    }
    return total_loss, metrics


@torch.no_grad()
def evaluate_flow_matching(
    model: torch.nn.Module,
    val_loader,
    action_normalizer: ActionNormalizer,
    device: torch.device | str = "cuda",
    max_batches: int = DEFAULT_EVAL_BATCHES,
) -> dict[str, float]:
    device = torch.device(device)
    model_was_training = model.training
    model.eval()

    total_flow_mse = 0.0
    total_action_mse = 0.0
    total_waypoint_mse = 0.0
    total_waypoint_batches = 0
    total_latency_ms = 0.0
    num_batches = 0

    for num_batches, batch in enumerate(val_loader, start=1):
        if max_batches is not None and num_batches > max_batches:
            break
        batch = move_batch_to_device(batch, device)
        batch = prepare_flow_matching_batch(batch, action_normalizer)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        outputs = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_latency_ms += (time.time() - t0) * 1000.0

        _, loss_metrics = compute_loss(outputs, batch, action_normalizer)
        total_flow_mse += loss_metrics["flow_mse"]
        total_action_mse += loss_metrics["action_mse"]

        waypoint_preds = outputs.get("waypoint_preds")
        waypoint_targets = batch["waypoints"]
        if waypoint_preds is not None and waypoint_targets.numel() > 0 and waypoint_targets.size(-1) > 0:
            total_waypoint_mse += F.mse_loss(waypoint_preds, waypoint_targets, reduction="mean").item()
            total_waypoint_batches += 1

    if num_batches == 0:
        raise ValueError("Validation loader is empty")

    evaluated_batches = min(num_batches, max_batches) if max_batches is not None else num_batches
    metrics = {
        "val_flow_mse": total_flow_mse / evaluated_batches,
        "val_action_mse": total_action_mse / evaluated_batches,
        "val_waypoint_mse": (
            total_waypoint_mse / total_waypoint_batches if total_waypoint_batches > 0 else float("nan")
        ),
        "latency_ms": total_latency_ms / evaluated_batches,
    }

    if model_was_training:
        model.train()
    return metrics


def format_flow_eval_summary(metrics: dict[str, float]) -> list[str]:
    waypoint = "nan" if math.isnan(metrics["val_waypoint_mse"]) else f"{metrics['val_waypoint_mse']:.6f}"
    return [
        f"val_flow_mse:      {metrics['val_flow_mse']:.6f}",
        f"val_action_mse:    {metrics['val_action_mse']:.6f}",
        f"val_waypoint_mse:  {waypoint}",
        f"latency_ms:        {metrics['latency_ms']:.2f}",
    ]


def get_lr_multiplier(progress: float) -> float:
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    return cooldown + (1.0 - cooldown) * FINAL_LR_FRAC


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def build_model_config(runtime) -> ModelConfig:
    metadata = runtime.metadata
    config = ModelConfig(
        max_seq_len=MAX_SEQ_LEN,
        instruction_vocab_size=runtime.instruction_tokenizer.get_vocab_size(),
        action_vocab_size=runtime.action_tokenizer.get_vocab_size(),
        image_channels=metadata.image_channels,
        model_image_size=MODEL_IMAGE_SIZE,
        history_frames=HISTORY_FRAMES,
        patch_size=PATCH_SIZE,
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
        time_embed_dim=TIME_EMBED_DIM,
    )
    if config.total_sequence_len > MAX_SEQ_LEN:
        raise ValueError(
            f"Model sequence length {config.total_sequence_len} exceeds MAX_SEQ_LEN={MAX_SEQ_LEN}. "
            "Reduce MAX_INSTRUCTION_TOKENS, HISTORY_FRAMES, ACTION_CHUNK_SIZE, or PATCH_SIZE."
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
if OBJECTIVE != "flow_matching":
    raise ValueError(f"Unsupported OBJECTIVE={OBJECTIVE!r}; this script currently implements flow_matching only")


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
action_normalizer = ActionNormalizer.from_metadata(runtime.metadata, device)
print(f"Model config: {asdict(config)}")
print(f"Train windows: {len(runtime.train_dataset):,}")
print(f"Val windows:   {len(runtime.val_dataset):,}")

if TOTAL_BATCH_SIZE % DEVICE_BATCH_SIZE != 0:
    raise ValueError("TOTAL_BATCH_SIZE must be divisible by DEVICE_BATCH_SIZE")
grad_accum_steps = TOTAL_BATCH_SIZE // DEVICE_BATCH_SIZE

model = VLATransformer(config).to(device)
if USE_COMPILE:
    model = torch.compile(model, dynamic=False)
num_params = count_parameters(model)

optimizer_kwargs = {
    "lr": LEARNING_RATE,
    "betas": ADAM_BETAS,
    "weight_decay": WEIGHT_DECAY,
}
if device.type == "cuda":
    optimizer_kwargs["fused"] = True
optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
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
smooth_flow_mse = 0.0
smooth_action_mse = 0.0
smooth_waypoint_loss = 0.0
total_training_time = 0.0
step = 0
total_windows = 0

while True:
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    micro_flow_mse = 0.0
    micro_action_mse = 0.0
    micro_waypoint_loss = 0.0
    micro_total_loss = 0.0
    for _ in range(grad_accum_steps):
        batch = move_batch_to_device(batch_cpu, device)
        batch = prepare_flow_matching_batch(batch, action_normalizer)
        with autocast_context():
            outputs = model(batch)
            loss, loss_metrics = compute_loss(outputs, batch, action_normalizer)
        (loss / grad_accum_steps).backward()
        micro_total_loss += loss.detach().item()
        micro_flow_mse += loss_metrics["flow_mse"]
        micro_action_mse += loss_metrics["action_mse"]
        micro_waypoint_loss += loss_metrics["waypoint_loss"]
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
    train_flow_mse = micro_flow_mse / grad_accum_steps
    train_action_mse = micro_action_mse / grad_accum_steps
    train_waypoint_loss = micro_waypoint_loss / grad_accum_steps
    if not math.isfinite(train_total_loss):
        raise RuntimeError(f"Training diverged with non-finite loss: {train_total_loss}")

    ema_beta = 0.9
    smooth_total_loss = ema_beta * smooth_total_loss + (1 - ema_beta) * train_total_loss
    smooth_flow_mse = ema_beta * smooth_flow_mse + (1 - ema_beta) * train_flow_mse
    smooth_action_mse = ema_beta * smooth_action_mse + (1 - ema_beta) * train_action_mse
    smooth_waypoint_loss = ema_beta * smooth_waypoint_loss + (1 - ema_beta) * train_waypoint_loss
    debiased_total = smooth_total_loss / (1 - ema_beta ** (step + 1))
    debiased_flow = smooth_flow_mse / (1 - ema_beta ** (step + 1))
    debiased_action = smooth_action_mse / (1 - ema_beta ** (step + 1))
    debiased_waypoint = smooth_waypoint_loss / (1 - ema_beta ** (step + 1))

    windows_per_sec = TOTAL_BATCH_SIZE / max(dt, 1e-6)
    remaining = max(0.0, TIME_BUDGET - total_training_time)
    print(
        f"\rstep {step:05d} ({progress * 100:.1f}%) | "
        f"loss: {debiased_total:.4f} | flow_mse: {debiased_flow:.4f} | "
        f"action_mse: {debiased_action:.4f} | waypoint: {debiased_waypoint:.4f} | "
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
    metrics = evaluate_flow_matching(
        model=model,
        val_loader=runtime.val_loader,
        action_normalizer=action_normalizer,
        device=device,
        max_batches=VAL_EVAL_BATCHES,
    )

t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0.0
windows_per_second = total_windows / max(total_training_time, 1e-6)

print("---")
for line in format_flow_eval_summary(metrics):
    print(line)
print(f"training_seconds:   {total_training_time:.1f}")
print(f"total_seconds:      {t_end - t_start:.1f}")
print(f"peak_vram_mb:       {peak_vram_mb:.1f}")
print(f"windows_per_second: {windows_per_second:.1f}")
print(f"total_windows:      {total_windows}")
print(f"num_steps:          {step}")
print(f"num_params_M:       {num_params / 1e6:.1f}")
print(f"depth:              {DEPTH}")
print(f"objective:          {OBJECTIVE}")

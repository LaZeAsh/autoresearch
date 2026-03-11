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
from transformers import AutoConfig, AutoModel, AutoProcessor

from prepare import (
    DEFAULT_EVAL_BATCHES,
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
HISTORY_FRAMES = 1
FRAME_STRIDE = 1
ACTION_CHUNK_SIZE = 4
PAST_ACTION_CHUNK_SIZE = 1
MAX_INSTRUCTION_TOKENS = 64
USE_STATE = True
USE_PAST_ACTIONS = True

# Frozen backbone
SMOLVLM_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
SMOLVLM_DTYPE = "bfloat16"
MAX_CONTEXT_TOKENS = 512
BACKBONE_SKIP_LAYERS = 8  # Use only first N of 32 LM layers (0 = use all)
LORA_RANK = 0  # LoRA rank for backbone adaptation (0 = no LoRA)
LORA_ALPHA = 16.0  # LoRA scaling factor

# Trainable action policy
DEPTH = 4
N_HEAD = 8
N_EMBD = 512
MLP_RATIO = 4.0
DROPOUT = 0.0

# Optimization
DEVICE_BATCH_SIZE = 16
TOTAL_BATCH_SIZE = 16
LEARNING_RATE = 1.5e-4
WEIGHT_DECAY = 0.005
ADAM_BETAS = (0.9, 0.999)
GRAD_CLIP_NORM = 1.0
WAYPOINT_LOSS_WEIGHT = 0.25
LABEL_SMOOTHING = 0.0
WARMUP_RATIO = 0.05
WARMDOWN_RATIO = 0.30
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
    smolvlm_model_id: str
    backbone_dtype: str
    backbone_hidden_size: int
    max_context_tokens: int
    frame_mean: tuple[float, ...]
    frame_std: tuple[float, ...]
    action_vocab_size: int
    history_frames: int
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


class LoRALinear(nn.Module):
    """Low-rank adaptation wrapper for a frozen Linear layer."""
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(base.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, base.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A @ self.lora_B) * self.scaling


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


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = int(config.n_embd * config.mlp_ratio * 2 / 3)
        hidden_dim = ((hidden_dim + 7) // 8) * 8
        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class FrozenSmolVLMActionPolicy(nn.Module):
    def __init__(self, config: ModelConfig, instruction_tokenizer):
        super().__init__()
        self.config = config
        if config.total_sequence_len > config.max_seq_len:
            raise ValueError(
                f"Configured sequence length {config.total_sequence_len} exceeds MAX_SEQ_LEN={config.max_seq_len}"
            )

        self.instruction_tokenizer = instruction_tokenizer
        self.backbone_dtype = getattr(torch, config.backbone_dtype)

        self.processor = AutoProcessor.from_pretrained(config.smolvlm_model_id)
        self.processor.image_processor.do_image_splitting = False
        self.processor.image_processor.do_rescale = False
        self.smolvlm = AutoModel.from_pretrained(
            config.smolvlm_model_id,
            dtype=self.backbone_dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        self.smolvlm.requires_grad_(False)
        self.smolvlm.eval()
        # Layer skipping: truncate LM decoder layers for faster inference
        if BACKBONE_SKIP_LAYERS > 0:
            total_layers = len(self.smolvlm.text_model.layers)
            keep = min(BACKBONE_SKIP_LAYERS, total_layers)
            self.smolvlm.text_model.layers = self.smolvlm.text_model.layers[:keep]

        # LoRA: add low-rank adapters to LM attention layers
        if LORA_RANK > 0:
            for layer in self.smolvlm.text_model.layers:
                layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, LORA_RANK, LORA_ALPHA)
                layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, LORA_RANK, LORA_ALPHA)

        # Pre-compute normalization constants for fast GPU preprocessing
        # Dataset normalization: frame = (raw - mean) / std
        # SmolVLM expects: (raw / 255 - 0.5) / 0.5 but since do_rescale=False,
        # the processor normally gets [0,1] images and does (x - 0.5)/0.5 = 2x - 1
        # Combined: smolvlm_input = 2*(frame*std + mean) - 1 = 2*std*frame + 2*mean - 1
        frame_mean = torch.tensor(config.frame_mean, dtype=torch.float32).view(1, 1, -1, 1, 1)
        frame_std = torch.tensor(config.frame_std, dtype=torch.float32).view(1, 1, -1, 1, 1)
        self.register_buffer("_norm_scale", 2.0 * frame_std)
        self.register_buffer("_norm_bias", 2.0 * frame_mean - 1.0)

        # Cache for tokenized input_ids per instruction
        self._input_ids_cache: dict[str, torch.Tensor] = {}

        self.context_proj = nn.Linear(config.backbone_hidden_size, config.n_embd, bias=False)
        self.action_embed = nn.Embedding(config.action_vocab_size, config.n_embd)
        self.start_token = nn.Parameter(torch.zeros(config.n_embd))
        self.state_proj = (
            nn.Sequential(
                nn.Linear(config.state_dim, config.n_embd, bias=False),
                nn.GELU(),
                nn.Linear(config.n_embd, config.n_embd, bias=False),
            )
            if config.state_dim > 0 and config.use_state
            else None
        )
        self.type_embed = nn.Embedding(4, config.n_embd)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, config.n_embd))
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.final_norm = RMSNorm(config.n_embd)
        self.action_head = nn.Linear(config.n_embd, config.action_vocab_size, bias=False)
        self.waypoint_head = (
            nn.Linear(config.n_embd, config.waypoint_dim, bias=False) if config.waypoint_dim > 0 else None
        )
        self._init_trainable_weights()

    def train(self, mode: bool = True):
        super().train(mode)
        self.smolvlm.eval()
        return self

    def _init_trainable_weights(self) -> None:
        def init_module(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)

        self.context_proj.apply(init_module)
        if self.state_proj is not None:
            self.state_proj.apply(init_module)
        self.blocks.apply(init_module)
        self.action_head.apply(init_module)
        if self.waypoint_head is not None:
            self.waypoint_head.apply(init_module)
        nn.init.normal_(self.action_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.type_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.01)
        nn.init.normal_(self.start_token, mean=0.0, std=0.02)

    def _decode_instruction_texts(self, instruction_tokens: torch.Tensor) -> list[str]:
        tokens_cpu = instruction_tokens.detach().cpu()
        return [self.instruction_tokenizer.decode(row) for row in tokens_cpu]

    def _get_input_ids(self, instruction: str) -> torch.Tensor:
        """Get tokenized input_ids for an instruction (cached).
        Uses the full processor once with a dummy 512x512 image to get
        correctly expanded image tokens, then caches for reuse."""
        cached = self._input_ids_cache.get(instruction)
        if cached is not None:
            return cached
        message = [
            {
                "role": "user",
                "content": [{"type": "image"} for _ in range(self.config.history_frames)]
                + [{"type": "text", "text": instruction}],
            }
        ]
        prompt = self.processor.apply_chat_template(message, add_generation_prompt=False)
        # Use a small dummy image just to get correct token expansion
        dummy_img = torch.zeros(3, 512, 512)
        out = self.processor(text=[prompt], images=[[dummy_img]], return_tensors="pt")
        ids = out["input_ids"][0]
        self._input_ids_cache[instruction] = ids
        return ids

    def _prepare_backbone_inputs(self, batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
        """Fast preprocessing: resize+normalize on GPU, cached text tokenization."""
        frames = batch["frames"]  # [B, T, C, H, W] on CPU, dataset-normalized
        B, T, C, H, W = frames.shape

        # Convert frames to SmolVLM normalization on GPU
        frames_gpu = frames.to(device=device, dtype=torch.float32)
        pixel_values = frames_gpu * self._norm_scale.to(device) + self._norm_bias.to(device)

        # Resize to 512x512 (SmolVLM's expected resolution)
        pixel_values = pixel_values.reshape(B * T, C, H, W)
        if H != 512 or W != 512:
            pixel_values = F.interpolate(pixel_values, size=(512, 512), mode="bilinear", align_corners=False)
        pixel_values = pixel_values.reshape(B, T, C, 512, 512).to(dtype=self.backbone_dtype)

        # Get cached input_ids for each instruction
        instructions = self._decode_instruction_texts(batch["instruction_tokens"])
        ids_list = [self._get_input_ids(text) for text in instructions]
        max_len = max(ids.shape[0] for ids in ids_list)
        input_ids = torch.zeros(B, max_len, dtype=torch.long, device=device)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for i, ids in enumerate(ids_list):
            input_ids[i, : ids.shape[0]] = ids.to(device)
            attention_mask[i, : ids.shape[0]] = 1

        pixel_attention_mask = torch.ones(B, T, 512, 512, dtype=torch.long, device=device)

        return {
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def _build_attention_mask(
        self,
        context_mask: torch.Tensor,
        state_tokens: torch.Tensor | None,
        past_action_tokens: torch.Tensor | None,
        action_input_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = context_mask.size(0)
        device = context_mask.device
        parts = [context_mask.bool()]
        if state_tokens is not None:
            parts.append(torch.ones(batch_size, state_tokens.size(1), dtype=torch.bool, device=device))
        if past_action_tokens is not None:
            parts.append(torch.ones(batch_size, past_action_tokens.size(1), dtype=torch.bool, device=device))
        parts.append(torch.ones(batch_size, action_input_tokens.size(1), dtype=torch.bool, device=device))
        keep_mask = torch.cat(parts, dim=1)
        seq_len = keep_mask.size(1)
        causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
        return causal.unsqueeze(0).unsqueeze(1) & keep_mask[:, None, None, :]

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | None]:
        target_action_tokens = batch["target_action_tokens"]
        past_action_tokens = batch["past_action_tokens"]

        batch_size = target_action_tokens.size(0)
        target_len = target_action_tokens.size(1)
        device = target_action_tokens.device

        # Backbone (frozen base weights, LoRA adapters are trainable)
        backbone_inputs = self._prepare_backbone_inputs(batch, device)
        if LORA_RANK > 0:
            backbone_outputs = self.smolvlm(**backbone_inputs)
        else:
            with torch.no_grad():
                backbone_outputs = self.smolvlm(**backbone_inputs)
        context_hidden = backbone_outputs.last_hidden_state
        context_mask = backbone_inputs["attention_mask"]
        if context_hidden.size(1) > self.config.max_context_tokens:
            context_hidden = context_hidden[:, -self.config.max_context_tokens :]
            context_mask = context_mask[:, -self.config.max_context_tokens :]
        context_tokens = self.context_proj(context_hidden.to(dtype=self.context_proj.weight.dtype))
        context_tokens = context_tokens + self.type_embed.weight[0]

        # State tokens
        state_tokens = None
        if self.state_proj is not None and batch["states"].numel() > 0:
            state_tokens = self.state_proj(batch["states"]).unsqueeze(1)
            state_tokens = state_tokens + self.type_embed.weight[1]

        # Past action tokens
        past_tokens = None
        if self.config.use_past_actions and past_action_tokens.numel() > 0:
            past_tokens = self.action_embed(past_action_tokens)
            past_tokens = past_tokens + self.type_embed.weight[2]

        # Target action tokens (shifted right for autoregressive prediction)
        # Position 0 gets start_token, positions 1..N-1 get target_tokens[0..N-2]
        target_embeds = self.action_embed(target_action_tokens)
        start = self.start_token.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        action_input = torch.cat([start, target_embeds[:, :-1]], dim=1)
        action_input = action_input + self.type_embed.weight[3]

        # Assemble sequence
        sequence_parts = [context_tokens]
        if state_tokens is not None:
            sequence_parts.append(state_tokens)
        if past_tokens is not None:
            sequence_parts.append(past_tokens)
        sequence_parts.append(action_input)
        x = torch.cat(sequence_parts, dim=1)

        if x.size(1) > self.config.max_seq_len:
            raise ValueError(f"Sequence length {x.size(1)} exceeds configured maximum {self.config.max_seq_len}")

        x = x + self.pos_embed[:, : x.size(1)]
        attn_mask = self._build_attention_mask(context_mask, state_tokens, past_tokens, action_input)
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.final_norm(x)

        # Action logits from the last target_len positions
        action_hidden = x[:, -target_len:]
        action_logits = self.action_head(action_hidden)

        # Waypoint prediction
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

def compute_loss(
    outputs: dict[str, torch.Tensor | None],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = outputs["action_logits"]
    targets = batch["target_action_tokens"]
    action_ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="mean",
        label_smoothing=LABEL_SMOOTHING,
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
    backbone_config = AutoConfig.from_pretrained(SMOLVLM_MODEL_ID)
    config = ModelConfig(
        max_seq_len=MAX_SEQ_LEN,
        smolvlm_model_id=SMOLVLM_MODEL_ID,
        backbone_dtype=SMOLVLM_DTYPE,
        backbone_hidden_size=backbone_config.text_config.hidden_size,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        frame_mean=metadata.frame_mean,
        frame_std=metadata.frame_std,
        action_vocab_size=runtime.action_tokenizer.get_vocab_size(),
        history_frames=HISTORY_FRAMES,
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

model = FrozenSmolVLMActionPolicy(config, runtime.instruction_tokenizer).to(device)
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
smooth_waypoint_loss = 0.0
total_training_time = 0.0
step = 0
total_windows = 0

while True:
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    micro_action_ce = 0.0
    micro_waypoint_loss = 0.0
    micro_total_loss = 0.0
    for _ in range(grad_accum_steps):
        batch = move_batch_to_device(batch_cpu, device)
        with autocast_context():
            outputs = model(batch)
            loss, loss_metrics = compute_loss(outputs, batch)
        (loss / grad_accum_steps).backward()
        micro_total_loss += loss.detach().item()
        micro_action_ce += loss_metrics["action_ce"]
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
    train_action_ce = micro_action_ce / grad_accum_steps
    train_waypoint_loss = micro_waypoint_loss / grad_accum_steps
    if not math.isfinite(train_total_loss):
        raise RuntimeError(f"Training diverged with non-finite loss: {train_total_loss}")

    ema_beta = 0.9
    smooth_total_loss = ema_beta * smooth_total_loss + (1 - ema_beta) * train_total_loss
    smooth_action_ce = ema_beta * smooth_action_ce + (1 - ema_beta) * train_action_ce
    smooth_waypoint_loss = ema_beta * smooth_waypoint_loss + (1 - ema_beta) * train_waypoint_loss
    debiased_total = smooth_total_loss / (1 - ema_beta ** (step + 1))
    debiased_ce = smooth_action_ce / (1 - ema_beta ** (step + 1))
    debiased_waypoint = smooth_waypoint_loss / (1 - ema_beta ** (step + 1))

    windows_per_sec = TOTAL_BATCH_SIZE / max(dt, 1e-6)
    remaining = max(0.0, TIME_BUDGET - total_training_time)
    print(
        f"\rstep {step:05d} ({progress * 100:.1f}%) | "
        f"loss: {debiased_total:.4f} | action_ce: {debiased_ce:.4f} | "
        f"waypoint: {debiased_waypoint:.4f} | "
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

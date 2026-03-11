"""
Fixed VLA dataset and evaluation harness for autoresearch experiments.

The human may bootstrap this file once to define the dataset contract, but after
that it should remain frozen while the agent iterates on train.py.

Dataset contract:
    - dataset root: $AUTORESEARCH_VLA_DATASET or ~/.cache/autoresearch/vla_dataset
    - dataset.json: dataset-level schema and metric weights
    - episodes.jsonl: one JSON object per episode
    - tensor payloads: .pt/.pth or .npy files for frames/actions/states/waypoints

Episode JSONL fields:
    {
        "episode_id": "unique-id",
        "instruction": "fly to the red gate",
        "frame_paths": ["frames/ep0001.pt"],   # tensor [T,C,H,W] or [T,H,W,C]
        "actions_path": "actions/ep0001.pt",   # tensor [T, action_dim]
        "states_path": "states/ep0001.pt",     # optional tensor [T, state_dim]
        "waypoints_path": "waypoints/ep0001.pt", # optional tensor [T, waypoint_dim]
        "split": "train"                       # optional: train or val
    }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants (fixed harness)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 1024
TIME_BUDGET = 300
DEFAULT_EVAL_BATCHES = 32
DEFAULT_VAL_RATIO = 0.1

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DEFAULT_DATASET_ROOT = os.path.join(CACHE_DIR, "vla_dataset")
DATASET_ROOT_ENV = "AUTORESEARCH_VLA_DATASET"
DEFAULT_LEROBOT_REPO = "LaZeAsh/uav-flow-lerobot-v3"
LEROBOT_CACHE_ENV = "AUTORESEARCH_LEROBOT_CACHE"
LEROBOT_REVISION_ENV = "AUTORESEARCH_LEROBOT_REVISION"
DEFAULT_LEROBOT_REVISION = "main"
LEROBOT_CAMERA_KEY = "observation.images.front"

PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
INSTRUCTION_VOCAB_SIZE = 259  # 256 bytes + pad/bos/eos

ACTION_SPECIAL_TOKENS = {
    "pad": 0,
    "bos": 1,
    "hover": 2,
    "stop": 3,
    "recover": 4,
}


# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetMetadata:
    backend: str
    source_id: str
    dataset_root: str
    dataset_path: str
    episodes_path: str
    image_size: int
    image_channels: int
    frame_mean: tuple[float, ...]
    frame_std: tuple[float, ...]
    instruction_template: str
    val_ratio: float
    state_dim: int
    state_mean: tuple[float, ...]
    state_std: tuple[float, ...]
    action_dim: int
    action_low: tuple[float, ...]
    action_high: tuple[float, ...]
    action_bins: int
    waypoint_dim: int
    score_weights: dict[str, float]

    @property
    def has_state(self) -> bool:
        return self.state_dim > 0

    @property
    def has_waypoints(self) -> bool:
        return self.waypoint_dim > 0


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    instruction: str
    frame_paths: tuple[str, ...]
    actions_path: str
    states_path: str | None
    waypoints_path: str | None
    split: str
    episode_index: int | None = None
    data_chunk_index: int | None = None
    data_file_index: int | None = None
    dataset_from_index: int | None = None
    dataset_to_index: int | None = None
    video_chunk_index: int | None = None
    video_file_index: int | None = None
    video_from_timestamp: float | None = None
    video_to_timestamp: float | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    batch_size: int
    history_frames: int
    frame_stride: int
    action_chunk_size: int
    past_action_chunk_size: int
    max_instruction_tokens: int
    include_state: bool
    include_past_actions: bool
    num_workers: int = 0
    pin_memory: bool = True


@dataclass(frozen=True)
class VLARuntime:
    metadata: DatasetMetadata
    instruction_tokenizer: "Tokenizer"
    action_tokenizer: "ActionTokenizer"
    train_dataset: "EpisodeWindowDataset"
    val_dataset: "EpisodeWindowDataset"
    train_loader: DataLoader
    val_loader: DataLoader


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------

class Tokenizer:
    """Fixed byte-level tokenizer for language instructions."""

    pad_token_id = PAD_TOKEN_ID
    bos_token_id = BOS_TOKEN_ID
    eos_token_id = EOS_TOKEN_ID

    def get_vocab_size(self) -> int:
        return INSTRUCTION_VOCAB_SIZE

    def get_pad_token_id(self) -> int:
        return self.pad_token_id

    def get_bos_token_id(self) -> int:
        return self.bos_token_id

    def get_eos_token_id(self) -> int:
        return self.eos_token_id

    def encode(self, text: str, max_length: int) -> torch.Tensor:
        encoded = text.encode("utf-8")
        tokens = [self.bos_token_id]
        tokens.extend(byte + 3 for byte in encoded[: max(0, max_length - 2)])
        tokens.append(self.eos_token_id)
        if len(tokens) < max_length:
            tokens.extend([self.pad_token_id] * (max_length - len(tokens)))
        else:
            tokens = tokens[:max_length]
            tokens[-1] = self.eos_token_id
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, token_ids: list[int] | torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        bytes_out = []
        for token_id in token_ids:
            if token_id in (self.pad_token_id, self.bos_token_id, self.eos_token_id):
                continue
            bytes_out.append(max(0, token_id - 3))
        return bytes(bytes_out).decode("utf-8", errors="ignore")


class ActionTokenizer:
    """Uniform scalar quantizer with per-dimension token offsets."""

    def __init__(self, metadata: DatasetMetadata):
        self.metadata = metadata
        self.action_dim = metadata.action_dim
        self.action_bins = metadata.action_bins
        self.special_tokens = ACTION_SPECIAL_TOKENS.copy()
        self.num_special_tokens = len(self.special_tokens)
        self.low = torch.tensor(metadata.action_low, dtype=torch.float32)
        self.high = torch.tensor(metadata.action_high, dtype=torch.float32)
        if torch.any(self.high <= self.low):
            raise ValueError("action_high must be strictly greater than action_low for every dimension")
        self.bin_width = (self.high - self.low) / metadata.action_bins
        self.vocab_size = self.num_special_tokens + self.action_dim * self.action_bins

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def get_pad_token_id(self) -> int:
        return self.special_tokens["pad"]

    def get_bos_token_id(self) -> int:
        return self.special_tokens["bos"]

    def _offsets(self, device: torch.device) -> torch.Tensor:
        return (
            torch.arange(self.action_dim, device=device, dtype=torch.long) * self.action_bins
            + self.num_special_tokens
        )

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 2 or actions.size(-1) != self.action_dim:
            raise ValueError(f"Expected actions of shape [T, {self.action_dim}], got {tuple(actions.shape)}")
        actions = actions.to(torch.float32)
        low = self.low.to(actions.device)
        high = self.high.to(actions.device)
        clipped = actions.clamp(low, high - 1e-6)
        bins = ((clipped - low) / self.bin_width.to(actions.device)).floor().to(torch.long)
        bins.clamp_(0, self.action_bins - 1)
        token_grid = bins + self._offsets(actions.device)
        return token_grid.reshape(-1)

    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim == 1:
            flat_tokens = token_ids
        else:
            flat_tokens = token_ids.reshape(-1)
        if flat_tokens.numel() % self.action_dim != 0:
            raise ValueError("Flat action token length must be divisible by action_dim")
        steps = flat_tokens.numel() // self.action_dim
        grid = flat_tokens.view(steps, self.action_dim)
        bins = torch.empty_like(grid)
        for dim in range(self.action_dim):
            lo = self.num_special_tokens + dim * self.action_bins
            hi = lo + self.action_bins
            if torch.any((grid[:, dim] < lo) | (grid[:, dim] >= hi)):
                raise ValueError("decode() received tokens outside the expected range for their action dimension")
            bins[:, dim] = grid[:, dim] - lo
        centers = self.low.to(grid.device) + (bins.to(torch.float32) + 0.5) * self.bin_width.to(grid.device)
        return centers

    def valid_token_mask(self, seq_len: int, device: torch.device | str) -> torch.Tensor:
        device = torch.device(device)
        mask = torch.zeros(seq_len, self.vocab_size, dtype=torch.bool, device=device)
        mask[:, : self.num_special_tokens] = True
        for pos in range(seq_len):
            dim = pos % self.action_dim
            lo = self.num_special_tokens + dim * self.action_bins
            hi = lo + self.action_bins
            mask[pos, lo:hi] = True
        return mask

    def invalid_rate(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError(f"Expected [B, S] predicted token ids, got {tuple(token_ids.shape)}")
        seq_len = token_ids.size(1)
        mask = self.valid_token_mask(seq_len, token_ids.device)
        valid = mask.unsqueeze(0).expand(token_ids.size(0), -1, -1)
        valid = valid.gather(2, token_ids.unsqueeze(-1)).squeeze(-1)
        return 1.0 - valid.float().mean()


# ---------------------------------------------------------------------------
# Dataset source loading
# ---------------------------------------------------------------------------

def _resolve_dataset_source(dataset_root: str | None = None) -> tuple[str, str]:
    source = dataset_root or os.environ.get(DATASET_ROOT_ENV)
    if source is None:
        if os.path.exists(DEFAULT_DATASET_ROOT):
            return "manifest", os.path.abspath(DEFAULT_DATASET_ROOT)
        return "lerobot", DEFAULT_LEROBOT_REPO

    source = source.strip()
    candidate_path = os.path.abspath(source)
    if os.path.exists(candidate_path):
        return "manifest", candidate_path
    if "/" in source and not source.startswith("/"):
        return "lerobot", source
    return "manifest", candidate_path


def _resolve_lerobot_cache_root() -> str:
    cache_root = os.environ.get(LEROBOT_CACHE_ENV)
    if cache_root:
        return os.path.abspath(cache_root)
    return os.path.join(CACHE_DIR, "lerobot")


def _resolve_lerobot_local_root(repo_id: str) -> str:
    return os.path.join(_resolve_lerobot_cache_root(), repo_id)


def _configure_hf_environment(cache_root: str) -> None:
    hf_home = os.path.join(cache_root, ".hf_home")
    os.makedirs(hf_home, exist_ok=True)
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_home, "datasets")
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_home, "hub")
    os.environ["HF_HUB_DISABLE_XET"] = "1"


def _snapshot_lerobot_repo(repo_id: str, allow_patterns: list[str] | str) -> str:
    from huggingface_hub import snapshot_download

    cache_root = _resolve_lerobot_cache_root()
    local_root = _resolve_lerobot_local_root(repo_id)
    os.makedirs(local_root, exist_ok=True)
    _configure_hf_environment(cache_root)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=os.environ.get(LEROBOT_REVISION_ENV, DEFAULT_LEROBOT_REVISION),
        local_dir=local_root,
        allow_patterns=allow_patterns,
    )
    return local_root


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _require_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(path)


def _as_float_tuple(value: Any, expected_len: int, field_name: str) -> tuple[float, ...]:
    if value is None:
        return tuple(0.0 for _ in range(expected_len))
    if len(value) != expected_len:
        raise ValueError(f"{field_name} must have length {expected_len}, got {len(value)}")
    return tuple(float(x) for x in value)


def _flatten_stat(value: Any, expected_len: int, field_name: str) -> tuple[float, ...]:
    flat = np.asarray(value, dtype=np.float32).reshape(-1)
    if flat.size != expected_len:
        raise ValueError(f"{field_name} must flatten to length {expected_len}, got {flat.size}")
    return tuple(float(x) for x in flat.tolist())


def _load_manifest_metadata(dataset_root: str) -> DatasetMetadata:
    dataset_path = os.path.join(dataset_root, "dataset.json")
    episodes_path = os.path.join(dataset_root, "episodes.jsonl")
    _require_file(dataset_path)
    _require_file(episodes_path)
    raw = _load_json(dataset_path)

    action_dim = int(raw["action_dim"])
    image_channels = int(raw.get("image_channels", 3))
    state_dim = int(raw.get("state_dim", 0))
    waypoint_dim = int(raw.get("waypoint_dim", 0))
    metadata = DatasetMetadata(
        backend="manifest",
        source_id=dataset_root,
        dataset_root=dataset_root,
        dataset_path=dataset_path,
        episodes_path=episodes_path,
        image_size=int(raw.get("image_size", 96)),
        image_channels=image_channels,
        frame_mean=_as_float_tuple(raw.get("frame_mean", [0.5] * image_channels), image_channels, "frame_mean"),
        frame_std=_as_float_tuple(raw.get("frame_std", [0.5] * image_channels), image_channels, "frame_std"),
        instruction_template=str(raw.get("instruction_template", "Task: {instruction}")),
        val_ratio=float(raw.get("val_ratio", DEFAULT_VAL_RATIO)),
        state_dim=state_dim,
        state_mean=_as_float_tuple(raw.get("state_mean", [0.0] * state_dim), state_dim, "state_mean"),
        state_std=_as_float_tuple(raw.get("state_std", [1.0] * state_dim), state_dim, "state_std"),
        action_dim=action_dim,
        action_low=_as_float_tuple(raw["action_low"], action_dim, "action_low"),
        action_high=_as_float_tuple(raw["action_high"], action_dim, "action_high"),
        action_bins=int(raw.get("action_bins", 256)),
        waypoint_dim=waypoint_dim,
        score_weights={
            "action_ce": float(raw.get("score_weights", {}).get("action_ce", 1.0)),
            "waypoint_mse": float(raw.get("score_weights", {}).get("waypoint_mse", 0.25)),
            "latency_ms": float(raw.get("score_weights", {}).get("latency_ms", 0.0025)),
            "invalid_action_rate": float(raw.get("score_weights", {}).get("invalid_action_rate", 1.0)),
        },
    )
    if not 0.0 < metadata.val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    if metadata.action_bins < 2:
        raise ValueError("action_bins must be >= 2")
    if metadata.image_size <= 0:
        raise ValueError("image_size must be > 0")
    return metadata


def _load_lerobot_metadata(repo_id: str) -> DatasetMetadata:
    local_root = _snapshot_lerobot_repo(
        repo_id,
        allow_patterns=[
            "meta/info.json",
            "meta/stats.json",
            "meta/tasks.parquet",
            "meta/episodes/**",
        ],
    )
    info_path = os.path.join(local_root, "meta", "info.json")
    stats_path = os.path.join(local_root, "meta", "stats.json")
    episodes_path = os.path.join(local_root, "meta", "episodes", "chunk-000", "file-000.parquet")
    _require_file(info_path)
    _require_file(stats_path)
    _require_file(episodes_path)
    info = _load_json(info_path)
    stats = _load_json(stats_path)

    image_feature = info["features"][LEROBOT_CAMERA_KEY]
    image_channels, image_height, image_width = (int(x) for x in image_feature["shape"])
    image_size = min(image_height, image_width)
    state_dim = int(info["features"]["observation.state"]["shape"][0])
    action_dim = int(info["features"]["action"]["shape"][0])
    waypoint_dim = 0

    metadata = DatasetMetadata(
        backend="lerobot",
        source_id=repo_id,
        dataset_root=local_root,
        dataset_path=info_path,
        episodes_path=episodes_path,
        image_size=image_size,
        image_channels=image_channels,
        frame_mean=_flatten_stat(stats[LEROBOT_CAMERA_KEY]["mean"], image_channels, "frame_mean"),
        frame_std=_flatten_stat(stats[LEROBOT_CAMERA_KEY]["std"], image_channels, "frame_std"),
        instruction_template="Task: {instruction}",
        val_ratio=DEFAULT_VAL_RATIO,
        state_dim=state_dim,
        state_mean=_as_float_tuple(stats["observation.state"]["mean"], state_dim, "state_mean"),
        state_std=_as_float_tuple(stats["observation.state"]["std"], state_dim, "state_std"),
        action_dim=action_dim,
        action_low=_as_float_tuple(stats["action"]["min"], action_dim, "action_low"),
        action_high=_as_float_tuple(stats["action"]["max"], action_dim, "action_high"),
        action_bins=256,
        waypoint_dim=waypoint_dim,
        score_weights={
            "action_ce": 1.0,
            "waypoint_mse": 0.25,
            "latency_ms": 0.0025,
            "invalid_action_rate": 1.0,
        },
    )
    if metadata.image_size <= 0:
        raise ValueError("image_size must be > 0")
    return metadata


def load_dataset_metadata(dataset_root: str | None = None) -> DatasetMetadata:
    backend, source = _resolve_dataset_source(dataset_root)
    if backend == "lerobot":
        return _load_lerobot_metadata(source)
    return _load_manifest_metadata(source)


def _stable_split(episode_id: str, val_ratio: float) -> str:
    digest = hashlib.sha1(episode_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_ratio else "train"


def _normalize_path(dataset_root: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(dataset_root, path)


def _load_manifest_episode_records(metadata: DatasetMetadata) -> list[EpisodeRecord]:
    records = []
    with open(metadata.episodes_path, "r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            episode_id = str(row["episode_id"])
            split = str(row.get("split") or _stable_split(episode_id, metadata.val_ratio)).lower()
            if split not in {"train", "val"}:
                raise ValueError(f"Invalid split {split!r} on line {line_number} of episodes.jsonl")
            frame_paths = row.get("frame_paths")
            if isinstance(frame_paths, str):
                frame_paths = [frame_paths]
            if not frame_paths:
                raise ValueError(f"Episode {episode_id!r} has no frame_paths")
            record = EpisodeRecord(
                episode_id=episode_id,
                instruction=str(row.get("instruction", "")),
                frame_paths=tuple(_normalize_path(metadata.dataset_root, path) for path in frame_paths),
                actions_path=_normalize_path(metadata.dataset_root, row["actions_path"]),
                states_path=(
                    _normalize_path(metadata.dataset_root, row["states_path"])
                    if row.get("states_path") is not None
                    else None
                ),
                waypoints_path=(
                    _normalize_path(metadata.dataset_root, row["waypoints_path"])
                    if row.get("waypoints_path") is not None
                    else None
                ),
                split=split,
            )
            records.append(record)
    if not records:
        raise ValueError("episodes.jsonl is empty")
    return records


def _load_lerobot_episode_records(metadata: DatasetMetadata) -> list[EpisodeRecord]:
    episode_table = pq.read_table(metadata.episodes_path)
    rows = episode_table.to_pylist()
    records = []
    for row in rows:
        episode_index = int(row["episode_index"])
        episode_id = f"ep-{episode_index:06d}"
        tasks = row.get("tasks") or []
        instruction = str(tasks[0]) if tasks else ""
        record = EpisodeRecord(
            episode_id=episode_id,
            instruction=instruction,
            frame_paths=(),
            actions_path="",
            states_path=None,
            waypoints_path=None,
            split=_stable_split(episode_id, metadata.val_ratio),
            episode_index=episode_index,
            data_chunk_index=int(row["data/chunk_index"]),
            data_file_index=int(row["data/file_index"]),
            dataset_from_index=int(row["dataset_from_index"]),
            dataset_to_index=int(row["dataset_to_index"]),
            video_chunk_index=int(row[f"videos/{LEROBOT_CAMERA_KEY}/chunk_index"]),
            video_file_index=int(row[f"videos/{LEROBOT_CAMERA_KEY}/file_index"]),
            video_from_timestamp=float(row[f"videos/{LEROBOT_CAMERA_KEY}/from_timestamp"]),
            video_to_timestamp=float(row[f"videos/{LEROBOT_CAMERA_KEY}/to_timestamp"]),
        )
        records.append(record)
    if not records:
        raise ValueError("LeRobot episodes metadata is empty")
    return records


def load_episode_records(metadata: DatasetMetadata) -> list[EpisodeRecord]:
    if metadata.backend == "lerobot":
        return _load_lerobot_episode_records(metadata)
    return _load_manifest_episode_records(metadata)


# ---------------------------------------------------------------------------
# Tensor loading and preprocessing
# ---------------------------------------------------------------------------

def _load_tensor(path: str) -> torch.Tensor:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".pt", ".pth"}:
        tensor = torch.load(path, map_location="cpu")
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        return tensor
    if ext == ".npy":
        return torch.from_numpy(np.load(path))
    raise ValueError(f"Unsupported tensor payload format for {path!r}; expected .pt/.pth or .npy")


def _center_crop_square(frames: torch.Tensor) -> torch.Tensor:
    height, width = frames.shape[-2:]
    if height == width:
        return frames
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return frames[..., top : top + side, left : left + side]


def _prepare_frames(frames: torch.Tensor, metadata: DatasetMetadata) -> torch.Tensor:
    if frames.ndim != 4:
        raise ValueError(f"Expected frame tensor with 4 dims [T,C,H,W] or [T,H,W,C], got {tuple(frames.shape)}")
    if frames.shape[-1] == metadata.image_channels and frames.shape[1] != metadata.image_channels:
        frames = frames.permute(0, 3, 1, 2)
    if frames.shape[1] != metadata.image_channels:
        raise ValueError(
            f"Expected {metadata.image_channels} image channels after permutation, got frame shape {tuple(frames.shape)}"
        )
    frames = frames.to(torch.float32)
    if frames.max().item() > 1.5:
        frames = frames / 255.0
    frames = _center_crop_square(frames)
    if frames.shape[-1] != metadata.image_size or frames.shape[-2] != metadata.image_size:
        frames = F.interpolate(
            frames,
            size=(metadata.image_size, metadata.image_size),
            mode="bilinear",
            align_corners=False,
        )
    mean = torch.tensor(metadata.frame_mean, dtype=torch.float32).view(1, -1, 1, 1)
    std = torch.tensor(metadata.frame_std, dtype=torch.float32).view(1, -1, 1, 1)
    return (frames - mean) / std.clamp_min(1e-6)


def _prepare_states(states: torch.Tensor, metadata: DatasetMetadata) -> torch.Tensor:
    if metadata.state_dim == 0:
        return torch.empty(states.shape[0], 0, dtype=torch.float32)
    if states.ndim != 2 or states.shape[-1] != metadata.state_dim:
        raise ValueError(f"Expected states [T, {metadata.state_dim}], got {tuple(states.shape)}")
    mean = torch.tensor(metadata.state_mean, dtype=torch.float32)
    std = torch.tensor(metadata.state_std, dtype=torch.float32)
    return (states.to(torch.float32) - mean) / std.clamp_min(1e-6)


def _validate_episode_payload(metadata: DatasetMetadata, record: EpisodeRecord, payload: dict[str, torch.Tensor]) -> None:
    frames = payload["frames"]
    actions = payload["actions"]
    if actions.ndim != 2 or actions.size(-1) != metadata.action_dim:
        raise ValueError(
            f"Episode {record.episode_id!r} actions must be [T, {metadata.action_dim}], got {tuple(actions.shape)}"
        )
    if frames.size(0) != actions.size(0):
        raise ValueError(
            f"Episode {record.episode_id!r} has mismatched lengths: frames={frames.size(0)}, actions={actions.size(0)}"
        )
    if metadata.has_state:
        states = payload["states"]
        if states.size(0) != actions.size(0):
            raise ValueError(
                f"Episode {record.episode_id!r} has mismatched states/actions lengths: "
                f"{states.size(0)} vs {actions.size(0)}"
            )
    if "waypoints" in payload:
        waypoints = payload["waypoints"]
        if waypoints.ndim != 2 or waypoints.size(-1) != metadata.waypoint_dim:
            raise ValueError(
                f"Episode {record.episode_id!r} waypoints must be [T, {metadata.waypoint_dim}], "
                f"got {tuple(waypoints.shape)}"
            )
        if waypoints.size(0) != actions.size(0):
            raise ValueError(
                f"Episode {record.episode_id!r} has mismatched waypoint/action lengths: "
                f"{waypoints.size(0)} vs {actions.size(0)}"
            )


class EpisodeCache:
    """Small LRU cache so repeated windows from the same episode do not thrash disk."""

    def __init__(self, metadata: DatasetMetadata, max_items: int = 8):
        self.metadata = metadata
        self.max_items = max_items
        self.cache: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()

    def get(self, record: EpisodeRecord) -> dict[str, torch.Tensor]:
        if record.episode_id in self.cache:
            payload = self.cache.pop(record.episode_id)
            self.cache[record.episode_id] = payload
            return payload

        frame_tensors = [_load_tensor(path) for path in record.frame_paths]
        frames = frame_tensors[0] if len(frame_tensors) == 1 else torch.cat(frame_tensors, dim=0)
        payload = {
            "frames": _prepare_frames(frames, self.metadata),
            "actions": _load_tensor(record.actions_path).to(torch.float32),
        }
        if record.states_path is not None:
            payload["states"] = _prepare_states(_load_tensor(record.states_path), self.metadata)
        elif self.metadata.has_state:
            raise ValueError(
                f"Dataset metadata declares state_dim={self.metadata.state_dim}, but episode {record.episode_id!r} "
                "is missing states_path"
            )
        if record.waypoints_path is not None:
            payload["waypoints"] = _load_tensor(record.waypoints_path).to(torch.float32)
        _validate_episode_payload(self.metadata, record, payload)
        self.cache[record.episode_id] = payload
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return payload

def _lerobot_data_file_path(metadata: DatasetMetadata, record: EpisodeRecord) -> str:
    if record.data_chunk_index is None or record.data_file_index is None:
        raise ValueError(f"Episode {record.episode_id!r} is missing LeRobot data file indices")
    info = _load_json(metadata.dataset_path)
    relative = info["data_path"].format(chunk_index=record.data_chunk_index, file_index=record.data_file_index)
    return os.path.join(metadata.dataset_root, relative)


def _lerobot_video_file_path(metadata: DatasetMetadata, record: EpisodeRecord) -> str:
    if record.video_chunk_index is None or record.video_file_index is None:
        raise ValueError(f"Episode {record.episode_id!r} is missing LeRobot video file indices")
    info = _load_json(metadata.dataset_path)
    relative = info["video_path"].format(
        video_key=LEROBOT_CAMERA_KEY,
        chunk_index=record.video_chunk_index,
        file_index=record.video_file_index,
    )
    return os.path.join(metadata.dataset_root, relative)


class LeRobotEpisodeCache:
    """Episode cache that reads LeRobot v3 parquet/video shards."""

    def __init__(self, metadata: DatasetMetadata, max_items: int = 8):
        self.metadata = metadata
        self.max_items = max_items
        self.cache: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()
        self.data_file_cache: dict[str, dict[str, torch.Tensor]] = {}

    def get(self, record: EpisodeRecord) -> dict[str, torch.Tensor]:
        if record.episode_id in self.cache:
            payload = self.cache.pop(record.episode_id)
            self.cache[record.episode_id] = payload
            return payload

        data_payload = self._load_data_slice(record)
        frames = self._load_video_frames(record, expected_length=data_payload["actions"].size(0))
        payload = {
            "frames": _prepare_frames(frames, self.metadata),
            "actions": data_payload["actions"],
        }
        if self.metadata.has_state:
            payload["states"] = _prepare_states(data_payload["states"], self.metadata)
        _validate_episode_payload(self.metadata, record, payload)
        self.cache[record.episode_id] = payload
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return payload

    def _load_data_slice(self, record: EpisodeRecord) -> dict[str, torch.Tensor]:
        data_path = _lerobot_data_file_path(self.metadata, record)
        if data_path not in self.data_file_cache:
            table = pq.read_table(data_path, columns=["observation.state", "action"])
            table_dict = table.to_pydict()
            self.data_file_cache[data_path] = {
                "states": torch.tensor(np.asarray(table_dict["observation.state"], dtype=np.float32)),
                "actions": torch.tensor(np.asarray(table_dict["action"], dtype=np.float32)),
            }

        if record.dataset_from_index is None or record.dataset_to_index is None:
            raise ValueError(f"Episode {record.episode_id!r} is missing LeRobot data row offsets")
        data_file = self.data_file_cache[data_path]
        start = record.dataset_from_index
        end = record.dataset_to_index
        return {
            "states": data_file["states"][start:end],
            "actions": data_file["actions"][start:end],
        }

    def _load_video_frames(self, record: EpisodeRecord, expected_length: int) -> torch.Tensor:
        video_path = _lerobot_video_file_path(self.metadata, record)
        start = float(record.video_from_timestamp or 0.0)
        end = float(record.video_to_timestamp or start)
        frames, _, _ = torchvision.io.read_video(video_path, start_pts=start, end_pts=end, pts_unit="sec")
        if frames.size(0) < expected_length:
            raise ValueError(
                f"Episode {record.episode_id!r} decoded only {frames.size(0)} frames, expected {expected_length}"
            )
        if frames.size(0) > expected_length:
            frames = frames[:expected_length]
        return frames


# ---------------------------------------------------------------------------
# Datasets and dataloaders
# ---------------------------------------------------------------------------

class EpisodeWindowDataset(Dataset):
    def __init__(
        self,
        metadata: DatasetMetadata,
        records: list[EpisodeRecord],
        split: str,
        instruction_tokenizer: Tokenizer,
        action_tokenizer: ActionTokenizer,
        runtime: RuntimeConfig,
        cache: EpisodeCache | LeRobotEpisodeCache | None = None,
    ):
        self.metadata = metadata
        self.records = [record for record in records if record.split == split]
        if not self.records:
            raise ValueError(f"No episodes found for split {split!r}")
        self.instruction_tokenizer = instruction_tokenizer
        self.action_tokenizer = action_tokenizer
        self.runtime = runtime
        self.cache = cache or EpisodeCache(metadata)
        self.samples = self._build_index()

    def _build_index(self) -> list[tuple[int, int]]:
        samples = []
        min_t = max(
            (self.runtime.history_frames - 1) * self.runtime.frame_stride,
            self.runtime.past_action_chunk_size if self.runtime.include_past_actions else 0,
        )
        for record_index, record in enumerate(self.records):
            if record.dataset_from_index is not None and record.dataset_to_index is not None:
                total_steps = record.dataset_to_index - record.dataset_from_index
            else:
                payload = self.cache.get(record)
                total_steps = payload["actions"].size(0)
            max_t = total_steps - self.runtime.action_chunk_size + 1
            for t in range(min_t, max_t):
                samples.append((record_index, t))
        if not samples:
            raise ValueError(
                f"No valid windows found for split {self.records[0].split!r}; "
                "check episode lengths against history_frames/action_chunk_size"
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record_index, t = self.samples[index]
        record = self.records[record_index]
        payload = self.cache.get(record)

        frame_indices = torch.arange(
            t - (self.runtime.history_frames - 1) * self.runtime.frame_stride,
            t + 1,
            self.runtime.frame_stride,
            dtype=torch.long,
        )
        frame_stack = payload["frames"].index_select(0, frame_indices)
        action_chunk = payload["actions"][t : t + self.runtime.action_chunk_size]
        target_action_tokens = self.action_tokenizer.encode(action_chunk)

        if self.runtime.include_past_actions and self.runtime.past_action_chunk_size > 0:
            past_actions = payload["actions"][t - self.runtime.past_action_chunk_size : t]
            past_action_tokens = self.action_tokenizer.encode(past_actions)
        else:
            past_action_tokens = torch.empty(0, dtype=torch.long)

        if self.runtime.include_state and self.metadata.has_state:
            state = payload["states"][t]
        else:
            state = torch.empty(0, dtype=torch.float32)

        if "waypoints" in payload:
            waypoint_chunk = payload["waypoints"][t : t + self.runtime.action_chunk_size]
        else:
            waypoint_chunk = torch.empty(self.runtime.action_chunk_size, 0, dtype=torch.float32)

        instruction = self.metadata.instruction_template.format(instruction=record.instruction.strip())
        instruction_tokens = self.instruction_tokenizer.encode(
            instruction,
            max_length=self.runtime.max_instruction_tokens,
        )
        instruction_mask = instruction_tokens != self.instruction_tokenizer.get_pad_token_id()

        return {
            "instruction_tokens": instruction_tokens,
            "instruction_mask": instruction_mask,
            "frames": frame_stack,
            "states": state,
            "past_action_tokens": past_action_tokens,
            "target_action_tokens": target_action_tokens,
            "action_targets": action_chunk.to(torch.float32),
            "waypoints": waypoint_chunk.to(torch.float32),
        }


def collate_vla_batch(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    batch = {
        "instruction_tokens": torch.stack([sample["instruction_tokens"] for sample in samples]),
        "instruction_mask": torch.stack([sample["instruction_mask"] for sample in samples]),
        "frames": torch.stack([sample["frames"] for sample in samples]),
        "states": torch.stack([sample["states"] for sample in samples]),
        "past_action_tokens": torch.stack([sample["past_action_tokens"] for sample in samples]),
        "target_action_tokens": torch.stack([sample["target_action_tokens"] for sample in samples]),
        "action_targets": torch.stack([sample["action_targets"] for sample in samples]),
        "waypoints": torch.stack([sample["waypoints"] for sample in samples]),
    }
    return batch


def build_vla_runtime(
    batch_size: int,
    history_frames: int,
    frame_stride: int,
    action_chunk_size: int,
    past_action_chunk_size: int,
    max_instruction_tokens: int,
    include_state: bool = True,
    include_past_actions: bool = True,
    dataset_root: str | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> VLARuntime:
    metadata = load_dataset_metadata(dataset_root)
    records = load_episode_records(metadata)
    if metadata.backend == "lerobot":
        _snapshot_lerobot_repo(metadata.source_id, allow_patterns=["data/**", "videos/**"])
    instruction_tokenizer = Tokenizer()
    action_tokenizer = ActionTokenizer(metadata)
    runtime = RuntimeConfig(
        batch_size=batch_size,
        history_frames=history_frames,
        frame_stride=frame_stride,
        action_chunk_size=action_chunk_size,
        past_action_chunk_size=past_action_chunk_size,
        max_instruction_tokens=max_instruction_tokens,
        include_state=include_state,
        include_past_actions=include_past_actions,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    cache: EpisodeCache | LeRobotEpisodeCache | None = None
    if metadata.backend == "lerobot":
        cache = LeRobotEpisodeCache(metadata)
    train_dataset = EpisodeWindowDataset(
        metadata=metadata,
        records=records,
        split="train",
        instruction_tokenizer=instruction_tokenizer,
        action_tokenizer=action_tokenizer,
        runtime=runtime,
        cache=cache,
    )
    val_dataset = EpisodeWindowDataset(
        metadata=metadata,
        records=records,
        split="val",
        instruction_tokenizer=instruction_tokenizer,
        action_tokenizer=action_tokenizer,
        runtime=runtime,
        cache=cache,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=metadata.backend != "lerobot",
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_vla_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=max(0, num_workers // 2),
        pin_memory=pin_memory,
        collate_fn=collate_vla_batch,
    )
    return VLARuntime(
        metadata=metadata,
        instruction_tokenizer=instruction_tokenizer,
        action_tokenizer=action_tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
    )


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device | str) -> dict[str, torch.Tensor]:
    keep_on_cpu = {"frames", "instruction_tokens", "instruction_mask"}
    moved = {}
    for key, value in batch.items():
        if key in keep_on_cpu:
            moved[key] = value
        else:
            moved[key] = value.to(device, non_blocking=True)
    return moved


def cycle(loader: DataLoader):
    epoch = 1
    while True:
        for batch in loader:
            yield batch, epoch
        epoch += 1


# ---------------------------------------------------------------------------
# Evaluation (fixed harness)
# ---------------------------------------------------------------------------

def _masked_action_logits(logits: torch.Tensor, action_tokenizer: ActionTokenizer) -> torch.Tensor:
    seq_len = logits.size(1)
    valid_mask = action_tokenizer.valid_token_mask(seq_len, logits.device)
    return logits.masked_fill(~valid_mask.unsqueeze(0), torch.finfo(logits.dtype).min)


def score_vla_metrics(metrics: dict[str, float], metadata: DatasetMetadata) -> float:
    score = -metadata.score_weights["action_ce"] * metrics["val_action_ce"]
    if not math.isnan(metrics["val_waypoint_mse"]):
        score -= metadata.score_weights["waypoint_mse"] * metrics["val_waypoint_mse"]
    score -= metadata.score_weights["latency_ms"] * metrics["latency_ms"]
    score -= metadata.score_weights["invalid_action_rate"] * metrics["invalid_action_rate"]
    return score


@torch.no_grad()
def evaluate_vla(
    model: torch.nn.Module,
    val_loader: DataLoader,
    action_tokenizer: ActionTokenizer,
    metadata: DatasetMetadata,
    device: torch.device | str = "cuda",
    max_batches: int = DEFAULT_EVAL_BATCHES,
) -> dict[str, float]:
    device = torch.device(device)
    model_was_training = model.training
    model.eval()

    total_action_ce = 0.0
    total_action_top1 = 0.0
    total_invalid = 0.0
    total_waypoint_mse = 0.0
    total_waypoint_batches = 0
    total_latency_ms = 0.0
    num_batches = 0

    for num_batches, batch in enumerate(val_loader, start=1):
        if max_batches is not None and num_batches > max_batches:
            break
        batch = move_batch_to_device(batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        outputs = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_latency_ms += (time.time() - t0) * 1000.0

        logits = _masked_action_logits(outputs["action_logits"], action_tokenizer)
        targets = batch["target_action_tokens"]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="mean",
        )
        preds = logits.argmax(dim=-1)
        total_action_ce += loss.item()
        total_action_top1 += (preds == targets).float().mean().item()
        total_invalid += action_tokenizer.invalid_rate(preds).item()

        waypoint_preds = outputs.get("waypoint_preds")
        waypoint_targets = batch["waypoints"]
        if waypoint_preds is not None and waypoint_targets.numel() > 0 and waypoint_targets.size(-1) > 0:
            total_waypoint_mse += F.mse_loss(waypoint_preds, waypoint_targets, reduction="mean").item()
            total_waypoint_batches += 1

    if num_batches == 0:
        raise ValueError("Validation loader is empty")

    evaluated_batches = min(num_batches, max_batches) if max_batches is not None else num_batches
    metrics = {
        "val_action_ce": total_action_ce / evaluated_batches,
        "val_action_top1": total_action_top1 / evaluated_batches,
        "val_waypoint_mse": (
            total_waypoint_mse / total_waypoint_batches if total_waypoint_batches > 0 else float("nan")
        ),
        "latency_ms": total_latency_ms / evaluated_batches,
        "invalid_action_rate": total_invalid / evaluated_batches,
    }
    metrics["score"] = score_vla_metrics(metrics, metadata)

    if model_was_training:
        model.train()
    return metrics


def format_eval_summary(metrics: dict[str, float]) -> list[str]:
    waypoint = "nan" if math.isnan(metrics["val_waypoint_mse"]) else f"{metrics['val_waypoint_mse']:.6f}"
    return [
        f"val_action_ce:      {metrics['val_action_ce']:.6f}",
        f"val_action_top1:    {metrics['val_action_top1']:.6f}",
        f"val_waypoint_mse:   {waypoint}",
        f"invalid_action_rate:{metrics['invalid_action_rate']:.6f}",
        f"latency_ms:         {metrics['latency_ms']:.2f}",
        f"score:              {metrics['score']:.6f}",
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_dataset_summary(metadata: DatasetMetadata, records: list[EpisodeRecord]) -> None:
    train_count = sum(1 for record in records if record.split == "train")
    val_count = sum(1 for record in records if record.split == "val")
    print(f"backend:            {metadata.backend}")
    print(f"source:             {metadata.source_id}")
    print(f"dataset_root:       {metadata.dataset_root}")
    print(f"episodes:           {len(records)}")
    print(f"train_episodes:     {train_count}")
    print(f"val_episodes:       {val_count}")
    print(f"image_size:         {metadata.image_size}")
    print(f"image_channels:     {metadata.image_channels}")
    print(f"state_dim:          {metadata.state_dim}")
    print(f"action_dim:         {metadata.action_dim}")
    print(f"action_bins:        {metadata.action_bins}")
    print(f"waypoint_dim:       {metadata.waypoint_dim}")
    print(f"val_ratio:          {metadata.val_ratio:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the fixed VLA dataset/eval harness")
    parser.add_argument("--dataset-root", type=str, default=None, help="Override $AUTORESEARCH_VLA_DATASET")
    parser.add_argument(
        "--print-schema",
        action="store_true",
        help="Print an example dataset.json and episodes.jsonl schema instead of validating data",
    )
    args = parser.parse_args()

    if args.print_schema:
        print(
            json.dumps(
                {
                    "image_size": 96,
                    "image_channels": 3,
                    "instruction_template": "Task: {instruction}",
                    "val_ratio": 0.1,
                    "state_dim": 9,
                    "state_mean": [0.0] * 9,
                    "state_std": [1.0] * 9,
                    "action_dim": 4,
                    "action_low": [-2.0, -2.0, -1.0, -1.0],
                    "action_high": [2.0, 2.0, 1.0, 1.0],
                    "action_bins": 256,
                    "waypoint_dim": 3,
                    "score_weights": {
                        "action_ce": 1.0,
                        "waypoint_mse": 0.25,
                        "latency_ms": 0.0025,
                        "invalid_action_rate": 1.0,
                    },
                },
                indent=2,
            )
        )
        print()
        print(
            json.dumps(
                {
                    "episode_id": "ep-000001",
                    "instruction": "take off and hover near the red gate",
                    "frame_paths": ["frames/ep-000001.pt"],
                    "actions_path": "actions/ep-000001.pt",
                    "states_path": "states/ep-000001.pt",
                    "waypoints_path": "waypoints/ep-000001.pt",
                    "split": "train",
                },
                indent=2,
            )
        )
    else:
        metadata = load_dataset_metadata(args.dataset_root)
        records = load_episode_records(metadata)
        print_dataset_summary(metadata, records)
        print()
        print("prepare.py validation: OK")

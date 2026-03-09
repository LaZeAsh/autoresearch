"""
One-time data preparation for autoresearch experiments.
Downloads data shards and trains a BPE tokenizer.

Usage:
    python prepare.py                  # full prep (download + tokenizer)
    python prepare.py --num-shards 8   # download only 8 shards (for testing)

Data and tokenizer are stored in ~/.cache/autoresearch/.
"""

import os
import sys
import time
import math
import argparse
import json
import pickle
from multiprocessing import Pool

import requests
import pyarrow as pa
import pyarrow.parquet as pq
import rustbpe
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048       # context length
TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
EVAL_TOKENS = 40 * 524288  # number of tokens for val eval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://huggingface.co/datasets/wangxiangyu0814/UAV-Flow/"


def parse_dataset_repo_id(base_url):
    """Extract `owner/name` from a Hugging Face dataset URL."""
    value = base_url.rstrip("/")
    if "/api/datasets/" in value:
        value = value.split("/api/datasets/", 1)[1]
    elif "/datasets/" in value:
        value = value.split("/datasets/", 1)[1]
    else:
        value = value.replace("https://huggingface.co/", "").replace("http://huggingface.co/", "")
    parts = [part for part in value.split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"BASE_URL must point to a Hugging Face dataset, got: {base_url}")
    return f"{parts[0]}/{parts[1]}"


DATASET_REPO_ID = parse_dataset_repo_id(BASE_URL)
DATASET_SLUG = DATASET_REPO_ID.replace("/", "__")
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data", DATASET_SLUG)
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer", DATASET_SLUG)
MAX_SHARD = 6542 # reserved local shard id for the validation output filename
VAL_SHARD = MAX_SHARD  # pinned validation shard (shard_06542)
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"
VOCAB_SIZE = 8192

# BPE split pattern (GPT-4 style, with \p{N}{1,2} instead of {1,3})
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def get_remote_parquet_urls():
    """Return the flat list of parquet URLs exposed by the dataset API."""
    api_url = f"https://huggingface.co/api/datasets/{DATASET_REPO_ID}/parquet"
    response = requests.get(api_url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    urls = []
    for config_splits in payload.values():
        for split_urls in config_splits.values():
            urls.extend(split_urls)
    if len(urls) < 2:
        raise RuntimeError(
            f"Expected at least 2 parquet files for train/val from {DATASET_REPO_ID}, found {len(urls)}"
        )
    return urls


def format_float(value):
    value = float(value)
    if not math.isfinite(value):
        return str(value)
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def format_state(values):
    return "[" + ", ".join(format_float(v) for v in values) + "]"


def build_spatial_text(episode_id, frame_idx, raw_logs, instruction):
    """Convert one UAV-Flow row into a text-only training document."""
    if not raw_logs:
        return None
    rows = []
    for values in raw_logs:
        if not isinstance(values, list):
            continue
        try:
            rows.append([float(v) for v in values])
        except (TypeError, ValueError):
            continue
    if not rows:
        return None

    start_state = rows[0]
    end_state = rows[-1]
    delta_state = [end - start for start, end in zip(start_state, end_state)]
    lines = [
        "task: infer the navigation instruction from raw drone flight logs",
        f"episode_id: {episode_id}",
        f"frame_idx: {frame_idx}",
        f"num_steps: {len(rows)}",
        f"log_dims: {len(rows[0])}",
        f"start_state: {format_state(start_state)}",
        f"end_state: {format_state(end_state)}",
        f"delta_end_start: {format_state(delta_state)}",
        "raw_logs:",
    ]
    for idx, values in enumerate(rows):
        lines.append(f"step_{idx:03d}: {format_state(values)}")
    lines.append("instruction:")
    lines.append(instruction.strip())
    return "\n".join(lines)


def log_entry_to_text(episode_id, frame_idx, log_blob):
    """Parse the dataset row and emit a text document for the tokenizer/model."""
    try:
        payload = json.loads(log_blob) if isinstance(log_blob, str) else log_blob
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    raw_logs = payload.get("raw_logs")
    instruction = payload.get("instruction_unified") or payload.get("instruction")
    if not raw_logs or not instruction:
        return None
    return build_spatial_text(episode_id, frame_idx, raw_logs, instruction)


def convert_remote_parquet(source_path, dest_path):
    """Rewrite a source parquet file into a local text-only shard."""
    parquet_file = pq.ParquetFile(source_path)
    writer = None
    wrote_any_rows = False
    try:
        for rg_idx in range(parquet_file.num_row_groups):
            rg = parquet_file.read_row_group(rg_idx, columns=["id", "frame_idx", "log"])
            ids = rg.column("id").to_pylist()
            frame_idxs = rg.column("frame_idx").to_pylist()
            logs = rg.column("log").to_pylist()
            texts = []
            for episode_id, frame_idx, log_blob in zip(ids, frame_idxs, logs):
                text = log_entry_to_text(episode_id, frame_idx, log_blob)
                if text is not None:
                    texts.append(text)
            if not texts:
                continue
            table = pa.table({"text": texts})
            if writer is None:
                writer = pq.ParquetWriter(dest_path, table.schema)
            writer.write_table(table)
            wrote_any_rows = True
    finally:
        if writer is not None:
            writer.close()
    if not wrote_any_rows:
        raise RuntimeError(f"No usable rows found in {source_path}")


def download_single_shard(task):
    """Download one remote parquet part and convert it into a local text shard."""
    source_url, filename = task
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return True

    download_tmp = filepath + ".download.tmp"
    converted_tmp = filepath + ".tmp"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(source_url, stream=True, timeout=30)
            response.raise_for_status()
            with open(download_tmp, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            convert_remote_parquet(download_tmp, converted_tmp)
            os.rename(converted_tmp, filepath)
            print(f"  Downloaded {filename}")
            return True
        except (requests.RequestException, IOError, RuntimeError, pa.ArrowInvalid, pa.ArrowException) as e:
            print(f"  Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [download_tmp, converted_tmp, filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False


def download_data(num_shards, download_workers=8):
    """Download training shards + pinned validation shard."""
    os.makedirs(DATA_DIR, exist_ok=True)
    remote_urls = get_remote_parquet_urls()
    train_urls = remote_urls[:-1]
    val_url = remote_urls[-1]
    if num_shards is None:
        num_train = len(train_urls)
    else:
        num_train = min(num_shards, len(train_urls))
    tasks = [(url, f"shard_{idx:05d}.parquet") for idx, url in enumerate(train_urls[:num_train])]
    tasks.append((val_url, VAL_FILENAME))

    # Count what's already downloaded
    existing = sum(1 for _, filename in tasks if os.path.exists(os.path.join(DATA_DIR, filename)))
    if existing == len(tasks):
        print(f"Data: all {len(tasks)} shards already downloaded at {DATA_DIR}")
        return

    needed = len(tasks) - existing
    print(f"Data: downloading {needed} shards ({existing} already exist)...")

    workers = max(1, min(download_workers, needed))
    with Pool(processes=workers) as pool:
        results = pool.map(download_single_shard, tasks)

    ok = sum(1 for r in results if r)
    print(f"Data: {ok}/{len(tasks)} shards ready at {DATA_DIR}")

# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def list_parquet_files():
    """Return sorted list of parquet file paths in the data directory."""
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]


def text_iterator(max_chars=1_000_000_000, doc_cap=10_000):
    """Yield documents from training split (all shards except pinned val shard)."""
    parquet_paths = [p for p in list_parquet_files() if not p.endswith(VAL_FILENAME)]
    nchars = 0
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def train_tokenizer():
    """Train BPE tokenizer using rustbpe, save as tiktoken pickle."""
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    parquet_files = list_parquet_files()
    if len(parquet_files) < 2:
        print("Tokenizer: need at least 2 data shards (1 train + 1 val). Download more data first.")
        sys.exit(1)

    # --- Train with rustbpe ---
    print("Tokenizer: training BPE tokenizer...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)

    # Build tiktoken encoding from trained merges
    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    # Save tokenizer
    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")

    # --- Build token_bytes lookup for BPB evaluation ---
    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    # Sanity check
    test = "Hello world! Numbers: 123. Unicode: 你好"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper. Training is handled above."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def _document_batches(split, tokenizer_batch_size=128):
    """Infinite iterator over document batches from parquet files."""
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No parquet files found. Run prepare.py first."
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
    else:
        parquet_paths = [val_path]
    epoch = 1
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    """
    BOS-aligned dataloader with best-fit packing.
    Every row starts with BOS. Documents packed using best-fit to minimize cropping.
    When no document fits remaining space, crops shortest doc to fill exactly.
    100% utilization (no padding).
    """
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    # Pre-allocate buffers: [inputs (B*T) | targets (B*T)]
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    # No doc fits — crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte. Special tokens (byte length 0)
    are excluded from both sums.
    Uses fixed MAX_SEQ_LEN so results are comparable across configs.
    """
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer for autoresearch")
    parser.add_argument("--num-shards", type=int, default=10, help="Number of training shards to download (-1 = all available remote shards). Validation is always added.")
    parser.add_argument("--download-workers", type=int, default=8, help="Number of parallel download workers")
    args = parser.parse_args()

    num_shards = None if args.num_shards == -1 else args.num_shards

    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Download data
    download_data(num_shards, download_workers=args.download_workers)
    print()

    # Step 2: Train tokenizer
    train_tokenizer()
    print()
    print("Done! Ready to train.")

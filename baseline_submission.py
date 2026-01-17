import csv
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

DIM = 2048
STATS = ("gc_content", "cpg_density", "max_homopolymer_ratio")
NUC_TO_INT = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
}
K_RANGE = tuple(range(4, 11))
HASH_DIM = DIM - len(STATS)
POWERS = {k: np.power(4, np.arange(k - 1, -1, -1, dtype=np.int64)) for k in K_RANGE}
def _longest_homopolymer(encoded: np.ndarray) -> int:
    if encoded.size == 0:
        return 0
    longest = 1
    current = 1
    for i in range(1, encoded.size):
        if encoded[i] == encoded[i - 1]:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 1
    return longest


def _normalise(vec: np.ndarray) -> None:
    norm = np.linalg.norm(vec)
    if norm:
        vec /= norm


def embed_sequence(seq: str) -> np.ndarray:
    """Convert a DNA sequence into a 2,048-dim feature vector."""
    if HASH_DIM <= 0:
        raise ValueError("DIM must exceed the number of global statistics")

    cleaned = seq.strip().upper()
    encoded = np.fromiter((NUC_TO_INT.get(base, -1) for base in cleaned), dtype=np.int16)
    hash_vec = np.zeros(HASH_DIM, dtype=np.float32)

    channels_used = 0
    for k in K_RANGE:
        if encoded.size < k:
            continue
        windows = sliding_window_view(encoded, k)
        mask = (windows >= 0).all(axis=1)
        if not mask.any():
            continue
        valid_windows = windows[mask].astype(np.int64, copy=False)
        values = (valid_windows * POWERS[k]).sum(axis=1)
        buckets = np.bincount(values % HASH_DIM, minlength=HASH_DIM).astype(np.float32)
        if buckets.any():
            _normalise(buckets)
            hash_vec += buckets
            channels_used += 1

    if channels_used and hash_vec.any():
        hash_vec /= channels_used

    valid_mask = encoded >= 0
    valid_encoded = encoded[valid_mask]
    valid_len = int(valid_encoded.size)
    gc_content = 0.0
    cpg_density = 0.0
    max_homopolymer_ratio = 0.0

    if valid_len:
        gc_mask = (valid_encoded == NUC_TO_INT["G"]) | (valid_encoded == NUC_TO_INT["C"])
        gc_content = float(gc_mask.sum() / valid_len)

        if valid_len > 1:
            cpg_pairs = (valid_encoded[:-1] == NUC_TO_INT["C"]) & (valid_encoded[1:] == NUC_TO_INT["G"])
            cpg_density = float(cpg_pairs.sum() / (valid_len - 1))

        longest_run = _longest_homopolymer(valid_encoded)
        max_homopolymer_ratio = float(longest_run / valid_len)

    vec = np.zeros(DIM, dtype=np.float32)
    vec[:HASH_DIM] = hash_vec
    vec[HASH_DIM:] = np.array([gc_content, cpg_density, max_homopolymer_ratio], dtype=np.float32)
    return vec


def main() -> None:
    root = Path(__file__).resolve().parent
    test_path = root / "test.csv"
    submission_path = root / "baseline_submission.csv"

    test_df = pd.read_csv(test_path)
    embeddings = np.vstack([embed_sequence(seq) for seq in test_df["seq"]])

    column_names = [f"emb_{idx:04d}" for idx in range(DIM)]
    submission_df = pd.DataFrame(embeddings, columns=column_names)
    submission_df.insert(0, "ID", test_df["ID"].values)

    submission_df.to_csv(submission_path, index=False, quoting=csv.QUOTE_NONE)


if __name__ == "__main__":
    main()

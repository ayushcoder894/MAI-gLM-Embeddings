import csv
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

DIM = 2048
NUM_STATS = 8
NUC_TO_INT = {"A":0, "C":1, "G":2, "T":3}

K_RANGE = tuple(range(4, 11))

HASH_DIM = DIM - NUM_STATS

# primes for double-hash
P1, P2 = 1315423911, 2654435761

POWERS = {k: np.power(4, np.arange(k-1, -1, -1, dtype=np.int64)) for k in K_RANGE}


def _normalise(v):
    n = np.linalg.norm(v)
    if n > 0:
        v /= n


def _shannon_entropy(encoded):
    if encoded.size == 0:
        return 0.0
    counts = np.bincount(encoded, minlength=4) / encoded.size
    nonzero = counts[counts > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def _longest_homopolymer(enc):
    if enc.size == 0:
        return 0
    longest = 1
    curr = 1
    for i in range(1, enc.size):
        if enc[i] == enc[i-1]:
            curr += 1
            longest = max(longest, curr)
        else:
            curr = 1
    return longest


def embed_sequence(seq: str) -> np.ndarray:
    seq = seq.strip().upper()
    encoded = np.fromiter((NUC_TO_INT.get(b, -1) for b in seq), dtype=np.int16)

    feature_vec = np.zeros(HASH_DIM, dtype=np.float32)
    channels = 0

    valid = encoded >= 0
    clean = encoded[valid].astype(np.int64, copy=False)

    # ----------------------------
    # A) multi-k hashed k-mers
    # ----------------------------
    for k in K_RANGE:
        if clean.size < k:
            continue

        windows = sliding_window_view(clean, k)
        vals = (windows * POWERS[k]).sum(axis=1)

        # double-hash to reduce collisions
        h1 = (vals * P1) % HASH_DIM
        h2 = (vals * P2) % HASH_DIM
        merged = (h1 ^ h2) % HASH_DIM

        buckets = np.bincount(merged, minlength=HASH_DIM).astype(np.float32)

        if buckets.any():
            _normalise(buckets)
            feature_vec += buckets
            channels += 1

    # average per-k channels
    if channels:
        feature_vec /= channels

    # ----------------------------
    # B) gapped k-mers (gap=1)
    # ----------------------------
    if clean.size >= 3:
        a = clean[:-2]
        c = clean[2:]
        vals = a * 17 + c * 19
        h = (vals * P1) % HASH_DIM
        buckets = np.bincount(h, minlength=HASH_DIM).astype(np.float32)
        _normalise(buckets)
        feature_vec += 0.3 * buckets   # small weight

    # ----------------------------
    # C) rolling hash encoding
    # ----------------------------
    roll = np.zeros(HASH_DIM, dtype=np.float32)
    h = 0
    for base in clean:
        h = (h * 131 + base) % HASH_DIM
        roll[h] += 1.0
    _normalise(roll)
    feature_vec += 0.5 * roll

    # ----------------------------
    # D) Global statistics
    # ----------------------------
    stats = np.zeros(NUM_STATS, dtype=np.float32)

    L = clean.size
    if L > 0:
        A = (clean == 0).sum()
        C = (clean == 1).sum()
        G = (clean == 2).sum()
        T = (clean == 3).sum()

        gc = (G + C) / L
        at_skew = (A - T) / max(A + T, 1)
        gc_skew = (G - C) / max(G + C, 1)

        cpg = 0
        if L > 1:
            cpg = ((clean[:-1] == 1) & (clean[1:] == 2)).sum() / (L - 1)

        entropy = _shannon_entropy(clean)
        hom = _longest_homopolymer(clean) / L

        stats[:] = (gc, cpg, at_skew, gc_skew, entropy, hom, L/10000, 0.0)

    # ----------------------------
    # Final vector
    # ----------------------------
    v = np.zeros(DIM, dtype=np.float32)
    v[:HASH_DIM] = feature_vec
    v[HASH_DIM:] = stats
    return v


def main():
    root = Path(__file__).resolve().parent
    test_df = pd.read_csv(root / "test.csv")

    embeddings = np.vstack([embed_sequence(s) for s in test_df["seq"]])

    cols = [f"emb_{i:04d}" for i in range(DIM)]
    out = pd.DataFrame(embeddings, columns=cols)
    out.insert(0, "ID", test_df["ID"])
    out.to_csv(root / "improved_submission.csv", index=False, quoting=csv.QUOTE_NONE)


from tqdm import tqdm

def main() -> None:
    root = Path(__file__).resolve().parent
    test_path = root / "test.csv"
    submission_path = root / "improved_submission.csv"

    test_df = pd.read_csv(test_path)

    print("Starting embedding...")
    embeddings = np.vstack([
        embed_sequence(seq)
        for seq in tqdm(test_df["seq"], desc="Embedding sequences", ncols=80)
    ])

    column_names = [f"emb_{idx:04d}" for idx in range(DIM)]
    submission_df = pd.DataFrame(embeddings, columns=column_names)
    submission_df.insert(0, "ID", test_df["ID"].values)

    submission_df.to_csv(submission_path, index=False, quoting=csv.QUOTE_NONE)
    print(f"Saved output → {submission_path}")

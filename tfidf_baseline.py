import csv
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ----------------------
# CONFIG
# ----------------------
TOTAL_DIM = 2048
STATS_DIM = 32
HASH_DIM = TOTAL_DIM - STATS_DIM

NUC_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}

K_SMALL = (4, 5, 6)
K_LARGE = (7, 8, 9)

# weights for small/large k
SMALL_GROUP_WEIGHT = 0.7
LARGE_GROUP_WEIGHT = 1.3

# seeds for hashing different channels
SEED_BASE_KMER = 0x9E3779B1
SEED_BASE_SHIFT = 0x85EBCA77
SEED_ROLLING = 0xC2B2AE35


# ----------------------
# LOW-LEVEL UTILS
# ----------------------
def _encode_sequence(seq: str) -> np.ndarray:
    cleaned = seq.strip().upper()
    return np.fromiter((NUC_TO_INT.get(b, -1) for b in cleaned), dtype=np.int16)


def _l2_normalise(vec: np.ndarray) -> None:
    n = np.linalg.norm(vec)
    if n > 0:
        vec /= n


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


def _shannon_entropy(encoded: np.ndarray) -> float:
    if encoded.size == 0:
        return 0.0
    counts = np.bincount(encoded, minlength=4).astype(np.float64)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def _xor_hash(x: int, seed: int) -> int:
    # 32-bit mixing
    x ^= seed
    x ^= (x >> 13)
    x ^= (x << 17) & 0xFFFFFFFF
    x ^= (x >> 5)
    return x & 0xFFFFFFFF


# ----------------------
# FEATURE CHANNELS
# ----------------------
def _hashed_kmer_channel(encoded: np.ndarray, k: int, seed: int) -> np.ndarray:
    """
    Hash-based k-mer histogram over HASH_DIM bins.
    Uses rolling base-4 code + xor hash to reduce collisions.
    """
    L = encoded.size
    if L < k:
        return np.zeros(HASH_DIM, dtype=np.float32)

    # ignore invalid bases
    if (encoded < 0).any():
        mask = encoded >= 0
        encoded = encoded[mask]
        L = encoded.size
        if L < k:
            return np.zeros(HASH_DIM, dtype=np.float32)

    pow4 = 4 ** (k - 1)
    hist = np.zeros(HASH_DIM, dtype=np.float32)

    # initial code
    code = 0
    for i in range(k):
        code = code * 4 + int(encoded[i])

    bucket = _xor_hash(code, seed) % HASH_DIM
    hist[bucket] += 1.0

    for i in range(k, L):
        # slide window by 1
        code = (code - int(encoded[i - k]) * pow4) * 4 + int(encoded[i])
        bucket = _xor_hash(code, seed) % HASH_DIM
        hist[bucket] += 1.0

    _l2_normalise(hist)
    return hist


def _rolling_hash_channel(encoded: np.ndarray, seed: int) -> np.ndarray:
    """
    Rolling hash over sequence positions.
    A single SNP or indel affects the downstream fingerprint.
    """
    if encoded.size == 0:
        return np.zeros(HASH_DIM, dtype=np.float32)

    # filter invalid
    encoded = encoded[encoded >= 0]
    if encoded.size == 0:
        return np.zeros(HASH_DIM, dtype=np.float32)

    hist = np.zeros(HASH_DIM, dtype=np.float32)
    h = seed
    for base in encoded:
        h = (h * 131 + int(base) + 7) & 0xFFFFFFFF
        bucket = h % HASH_DIM
        hist[bucket] += 1.0

    _l2_normalise(hist)
    return hist


def _neighbor_pair_stats(encoded: np.ndarray) -> np.ndarray:
    """
    16-dim normalized counts of all dinucleotide pairs.
    Provides transition/transversion and motif-type signals.
    """
    stats = np.zeros(16, dtype=np.float32)
    encoded = encoded[encoded >= 0]
    if encoded.size < 2:
        return stats

    pairs = encoded[:-1] * 4 + encoded[1:]
    counts = np.bincount(pairs, minlength=16).astype(np.float32)
    total = counts.sum()
    if total > 0:
        stats = counts / total
    return stats


def _global_stats(encoded: np.ndarray, raw_seq: str) -> np.ndarray:
    """
    32-dim global statistics:
    - GC, CpG, AT skew, GC skew
    - entropy, homopolymer ratio, length scaling, invalid fraction
    - 16-dim dinucleotide distribution
    - 8 simple motif counts (normalized)
    """
    stats = np.zeros(STATS_DIM, dtype=np.float32)

    valid = encoded >= 0
    clean = encoded[valid]
    L = clean.size

    if L > 0:
        A = (clean == 0).sum()
        C = (clean == 1).sum()
        G = (clean == 2).sum()
        T = (clean == 3).sum()

        gc = (G + C) / L
        at_skew = (A - T) / max(A + T, 1)
        gc_skew = (G - C) / max(G + C, 1)

        cpg = 0.0
        if L > 1:
            cpg = ((clean[:-1] == 1) & (clean[1:] == 2)).sum() / (L - 1)

        entropy = _shannon_entropy(clean)
        hom = _longest_homopolymer(clean) / L
        invalid_frac = 1.0 - (L / max(len(raw_seq.strip()), 1))

        length_scaled = L / 1000.0  # scale length a bit

        # motifs (simple)
        seq = raw_seq.strip().upper()
        motifs = ["CG", "CAG", "CTG", "AATAAA", "TATA", "GCGC", "ATAT", "GGG"]
        motif_vals = []
        for m in motifs:
            if len(seq) >= len(m):
                cnt = seq.count(m)
                motif_vals.append(cnt / max(L - len(m) + 1, 1))
            else:
                motif_vals.append(0.0)

        # neighbor pair stats (16 dims)
        pair_vec = _neighbor_pair_stats(encoded)

        # assemble
        idx = 0
        base_stats = [
            gc,
            cpg,
            at_skew,
            gc_skew,
            entropy,
            hom,
            length_scaled,
            invalid_frac,
        ]
        stats[idx : idx + len(base_stats)] = np.array(base_stats, dtype=np.float32)
        idx += len(base_stats)

        stats[idx : idx + pair_vec.size] = pair_vec
        idx += pair_vec.size

        motif_arr = np.array(motif_vals, dtype=np.float32)
        stats[idx : idx + motif_arr.size] = motif_arr
        # remaining dims (if any) stay zero

    return stats


def embed_sequence(seq: str) -> np.ndarray:
    """
    Main embedding function: returns 2048-dim vector.
    Highly mutation-sensitive, purely classical (no NN).
    """
    encoded = _encode_sequence(seq)
    feature_vec = np.zeros(HASH_DIM, dtype=np.float32)

    # ------------------------
    # 1) Multi-k exact k-mers
    # ------------------------
    if encoded.size > 0:
        # small k
        sum_small = sum(K_SMALL)
        for k in K_SMALL:
            w = SMALL_GROUP_WEIGHT * (k / sum_small)
            ch = _hashed_kmer_channel(encoded, k, seed=SEED_BASE_KMER + k * 13)
            feature_vec += w * ch

        # large k (more mutation-sensitive)
        sum_large = sum(K_LARGE)
        for k in K_LARGE:
            w = LARGE_GROUP_WEIGHT * (k / sum_large)
            ch = _hashed_kmer_channel(encoded, k, seed=SEED_BASE_KMER + k * 37)
            feature_vec += w * ch

        # -----------------------------------
        # 2) Shifted windows for indel signal
        # -----------------------------------
        for shift, shift_weight in ((-1, 0.45), (+1, 0.45)):
            if encoded.size + shift <= 0:
                continue
            if shift < 0:
                shifted = encoded[-shift:]
            else:
                shifted = encoded[:-shift]
            if shifted.size == 0:
                continue

            for k in K_LARGE:
                ch = _hashed_kmer_channel(shifted, k, seed=SEED_BASE_SHIFT + k * (17 + shift))
                feature_vec += shift_weight * ch

        # ------------------------
        # 3) Rolling hash channel
        # ------------------------
        roll_ch = _rolling_hash_channel(encoded, seed=SEED_ROLLING)
        feature_vec += 0.8 * roll_ch

    # ------------------------
    # 4) Global stats (32 dim)
    # ------------------------
    stats_vec = _global_stats(encoded, seq)

    # ------------------------
    # 5) Final 2048-dim vector
    # ------------------------
    out = np.zeros(TOTAL_DIM, dtype=np.float32)
    out[:HASH_DIM] = feature_vec
    out[HASH_DIM:] = stats_vec

    # final normalization (good for cosine distance)
    _l2_normalise(out)
    return out


def main() -> None:
    root = Path.cwd()
    test_path = root / "test.csv"
    submission_path = root / "gLM_mutation_sensitive_submission.csv"

    if not test_path.exists():
        raise FileNotFoundError(f"Missing test file: {test_path}")

    print(f"Loading {test_path} ...", flush=True)
    test_df = pd.read_csv(test_path)
    sequences = test_df["seq"].tolist()
    num_samples = len(sequences)

    print(
        f"Embedding {num_samples} sequences "
        f"(HASH_DIM={HASH_DIM}, STATS_DIM={STATS_DIM})...",
        flush=True,
    )

    embeddings = np.zeros((num_samples, TOTAL_DIM), dtype=np.float32)
    for idx in tqdm(range(num_samples), desc="Embedding sequences", unit="seq", ncols=80):
        embeddings[idx] = embed_sequence(sequences[idx])

    column_names = [f"emb_{i:04d}" for i in range(TOTAL_DIM)]
    submission_df = pd.DataFrame(embeddings, columns=column_names)
    submission_df.insert(0, "ID", test_df["ID"].values)

    print(f"Saving to {submission_path} ...", flush=True)
    submission_df.to_csv(submission_path, index=False, quoting=csv.QUOTE_NONE, float_format="%.6f")
    print("Done.")


if __name__ == "__main__":
    main()

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

TOTAL_DIM = 2048
MULTI_K = (3, 4, 5, 6)
NUC_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}


def _encode_sequence(seq: str) -> np.ndarray:
    cleaned = seq.strip().upper()
    return np.array([NUC_TO_INT.get(base, -1) for base in cleaned], dtype=np.int16)


def _extract_kmers(encoded: np.ndarray, k: int) -> dict:
    """Extract k-mer counts from encoded sequence."""
    kmer_counts = {}
    
    for i in range(len(encoded) - k + 1):
        window = encoded[i:i + k]
        if (window >= 0).all():
            kmer_int = 0
            for base_val in window:
                kmer_int = kmer_int * 4 + base_val
            kmer_counts[kmer_int] = kmer_counts.get(kmer_int, 0) + 1
    
    return kmer_counts


def _sparse_to_dense(kmer_counts: dict, k: int) -> np.ndarray:
    """Convert sparse k-mer counts to dense vector."""
    vocab_size = 4 ** k
    dense = np.zeros(vocab_size, dtype=np.float32)
    
    for kmer_int, count in kmer_counts.items():
        if 0 <= kmer_int < vocab_size:
            dense[kmer_int] = count
    
    return dense


def _random_projection(vector: np.ndarray, target_dim: int, seed: int = 42) -> np.ndarray:
    """Apply Johnson-Lindenstrauss random projection."""
    source_dim = vector.shape[0]
    
    rng = np.random.RandomState(seed)
    projection = rng.randn(source_dim, target_dim).astype(np.float32)
    projection *= np.sqrt(3.0 / source_dim)
    
    projected = vector @ projection
    
    norm = np.linalg.norm(projected)
    if norm > 0:
        projected /= norm
    
    return projected


def _compute_multiscale_kmer_embedding(encoded: np.ndarray, target_dim: int) -> np.ndarray:
    """Generate multi-scale k-mer embedding with random projection."""
    all_counts = []
    
    for k in MULTI_K:
        kmer_counts = _extract_kmers(encoded, k)
        dense = _sparse_to_dense(kmer_counts, k)
        
        total = dense.sum()
        if total > 0:
            dense /= total
        
        all_counts.append(dense)
    
    combined = np.concatenate(all_counts)
    
    if combined.shape[0] > target_dim:
        embedded = _random_projection(combined, target_dim, seed=42)
    else:
        embedded = np.zeros(target_dim, dtype=np.float32)
        embedded[:combined.shape[0]] = combined
        norm = np.linalg.norm(embedded)
        if norm > 0:
            embedded /= norm
    
    return embedded


def main() -> None:
    root = Path(__file__).resolve().parent
    test_path = root / "test.csv"
    submission_path = root / "kmer_projection_submission.csv"

    if not test_path.exists():
        raise FileNotFoundError(f"Missing test file: {test_path}")

    total_features = sum(4 ** k for k in MULTI_K)
    print(f"Generating multi-scale k-mer embeddings (k={MULTI_K}, dim={TOTAL_DIM})...", flush=True)
    print(f"Total raw k-mer features: {total_features} → projecting to {TOTAL_DIM}", flush=True)
    
    test_df = pd.read_csv(test_path)
    sequences = test_df["seq"].tolist()
    num_samples = len(sequences)

    embeddings = np.zeros((num_samples, TOTAL_DIM), dtype=np.float32)

    for idx in tqdm(range(num_samples), desc="Embedding sequences", unit="seq"):
        encoded = _encode_sequence(sequences[idx])
        embedding = _compute_multiscale_kmer_embedding(encoded, TOTAL_DIM)
        embeddings[idx] = embedding

    column_names = [f"emb_{idx:04d}" for idx in range(TOTAL_DIM)]
    submission_df = pd.DataFrame(embeddings, columns=column_names)
    submission_df.insert(0, "ID", test_df["ID"].values)

    submission_df.to_csv(submission_path, index=False, quoting=csv.QUOTE_NONE, float_format="%.6f")
    print(f"\nSaved submission to {submission_path}")


if __name__ == "__main__":
    main()

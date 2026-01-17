# 2nd MAI Competition - Genomic Language Model Mutation Sensitivity

**Competition:** Korea University Medical AI Competition (DACON)  
**Goal:** Generate 2,048-dimensional embeddings for DNA sequences that are sensitive to small genetic variations (SNPs, indels)

## Results

| Metric | Value |
|--------|-------|
| **Final Score** | 0.511 |
| **Rank** | 112 / 498 |

## Approaches Implemented

### 1. Baseline K-mer Hashing (`baseline_submission.py`)
- Multi-scale k-mer counts (k=4-10) with feature hashing
- GC content, CpG density, homopolymer ratio statistics
- Simple and fast CPU-only approach

### 2. TF-IDF Enhanced Baseline (`tfidf_baseline.py`)
- Reverse-complement aware k-mer hashing (k=3-11)
- TF-IDF weighting learned across all sequences
- 4 composition statistics (GC%, CpG density, homopolymer ratio, 1-mer entropy)
- L2 normalization per sequence

### 3. K-mer Random Projection (`fcgr_submission.py`)
- Multi-scale k-mer frequency vectors (k=3,4,5,6)
- Johnson-Lindenstrauss random projection (5,440 → 2,048 dims)
- Preserves pairwise distances while compressing feature space
- Alignment-free, CPU-only

### 4. Pretrained gLM Attempts (`sota_submission.py`, `sota_submission_lite.py`)
- Nucleotide Transformer v2: blocked by checkpoint shape mismatch
- DNABERT-2: blocked by triton dependency (Windows incompatible)
- These scripts are included for reference but may not run on Windows

## Key Techniques

| Technique | Description |
|-----------|-------------|
| K-mer counting | Bag-of-words representation of DNA subsequences |
| Feature hashing | Compress large k-mer vocabularies into fixed dimensions |
| TF-IDF weighting | Down-weight common k-mers, up-weight rare/discriminative ones |
| Random projection | Dimensionality reduction preserving distances (J-L lemma) |
| Reverse complement | Consider both strands for biological completeness |

## File Structure

```
├── baseline_submission.py      # Simple k-mer hashing baseline
├── tfidf_baseline.py           # TF-IDF weighted k-mer approach
├── fcgr_submission.py          # K-mer + random projection
├── sota_submission.py          # Nucleotide Transformer (ref only)
├── sota_submission_lite.py     # DNABERT-2 attempt (ref only)
├── advanced_baseline.py        # Extended k-mer range baseline
└── README.md
```

## Usage

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install numpy pandas tqdm

# Generate submission
python tfidf_baseline.py
# Output: tfidf_submission.csv (2048-dim embeddings)
```

## Requirements

- Python 3.10+
- numpy
- pandas
- tqdm

## Notes

- All submissions produce 2,048-dimensional vectors
- CSV format: ID, emb_0000, emb_0001, ..., emb_2047
- Evaluation metric: cosine distance between ref/variant pairs

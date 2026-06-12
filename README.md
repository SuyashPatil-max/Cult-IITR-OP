cat > /mnt/user-data/outputs/README.md << 'ENDOFFILE'
<div align="center">

# 🎬 Recommendation Systems for Personalized Content Discovery

### *Item-Based Collaborative Filtering · Singular Value Decomposition*
### *Netflix Prize Dataset · K=10 Evaluation · 500 Users*

<br />

| Metric | Item-CF | SVD |
|--------|---------|-----|
| **RMSE ↓** | **0.9422** ✅ | 1.0529 |
| **MAE ↓** | **0.7149** ✅ | 0.8852 |
| **MAP@10 ↑** | 0.005689 | **0.013685** ✅ |
| **Precision@10 ↑** | 0.0170 | **0.0374** ✅ |
| **Recall@10 ↑** | 0.006733 | **0.01642** ✅ |
| **NDCG@10 ↑** | 0.017604 | **0.039113** ✅ |

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Findings](#-key-findings)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Module Reference](#-module-reference)
- [Evaluation Methodology](#-evaluation-methodology)
- [Results & Analysis](#-results--analysis)
- [Future Improvements](#-future-improvements)
- [Citation](#-citation)

---

## 🎯 Overview

This project implements and compares two recommendation system paradigms on the **Netflix Prize Dataset** — one of the most influential benchmarks in the history of machine learning.

The goal is not merely to predict ratings, but to build a recommendation engine that **effectively captures user preferences** and **delivers meaningful, personalized content recommendations**.

**Two models are designed, implemented, and evaluated:**

| Model | Paradigm | Key Idea |
|-------|----------|----------|
| **Item-CF** | Memory-based Collaborative Filtering | Find items rated similarly; predict via weighted average of user's known ratings |
| **SVD** | Model-based Matrix Factorisation | Learn compressed user & item latent vectors; predict via dot product |

---

## 🔑 Key Findings

```
┌─────────────────────────────────────────────────────────────────────┐
│  CORE FINDING: Rating accuracy and ranking quality are DECOUPLED.   │
│                                                                     │
│  ► Item-CF predicts individual ratings more precisely (RMSE 0.942)  │
│  ► SVD recommends content users actually want (MAP@10 2.4× higher)  │
└─────────────────────────────────────────────────────────────────────┘
```

1. **Adjusted Cosine beats Raw Cosine** — Mean-centering user ratings removes individual rating bias and significantly improves RMSE accuracy in Item-CF.

2. **Latent factors capture global taste structure** — SVD's 100-dimensional embeddings model implicit genre preferences, actor affinities, and cross-genre taste clusters that local item similarity misses.

3. **SVD is 38× faster at evaluation** — RMSE eval: 8.8s vs 339.9s. At production scale, SVD's prediction vastly outperforms Item-CF's neighbour intersection.

4. **Sparsity is the dominant challenge** — Both models achieve MAP@10 < 0.02, reflecting the inherent difficulty: each user has rated < 0.05% of the 17,770-movie corpus.

---

## 📊 Dataset

**Netflix Prize Dataset** — [[Kaggle Link]](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)

| Property | Value |
|----------|-------|
| Total Ratings | 100,480,507 |
| Unique Users | 480,189 |
| Unique Movies | 17,770 |
| Rating Scale | 1–5 (integer) |
| Includes | Timestamps, Movie metadata |
| Sparsity | > 99.95% |

**Expected files in `data/`:**
```
data/
├── combined_data_1.txt
├── combined_data_2.txt
├── combined_data_3.txt
├── combined_data_4.txt
└── movie_titles.csv
```

> ⚠️ Due to memory constraints, the pipeline defaults to loading `combined_data_1.txt` only. You can change this to load all four files — see [Module Reference](#-module-reference).

---

## 📁 Project Structure

```
├── data/
│   ├── combined_data_1.txt         # Raw Netflix ratings (download separately)
│   ├── combined_data_2.txt
│   ├── combined_data_3.txt
│   ├── combined_data_4.txt
│   ├── movie_titles.csv
│   └── processed/                  # Auto-generated after preprocessing
│       ├── ratings_clean.parquet
│       ├── train.parquet
│       ├── val.parquet
│       ├── test.parquet
│       ├── train_matrix.npz
│       └── item_cf_similarity.pkl
│
├── results/
│   ├── models/                     # Saved trained model files
│   └── evaluation_results.json     # Final evaluation output
│
├── src/
│   ├── preprocessing.py            # Data loading, filtering, splitting, matrix building
│   ├── collaborative_filtering.py  # Item-Based CF with adjusted cosine similarity
│   ├── svd_model.py                # SVD via Surprise library
│   └── evaluation.py              # All metrics: RMSE, MAE, MAP@K, Precision, Recall, NDCG
│
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.9+
- 8 GB RAM minimum (16 GB recommended for full dataset)
- ~5 GB disk space for data + artefacts

### 1. Clone the repository

```bash
git clone https://github.com/your-username/netflix-recsys.git
cd netflix-recsys
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows
```

### 3. Install dependencies

```bash
pip install numpy pandas scipy scikit-learn scikit-surprise tqdm pyarrow
```

### 4. Download the dataset

Download from [Kaggle](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) and place all files in the `data/` directory as shown in the structure above.

---

## 🚀 How to Run

Follow these four steps **in order**. Each step builds on the outputs of the previous one.

---

### Step 1 — Preprocess the Data

```bash
python src/preprocessing.py
```

**What happens:**
- Parses the raw Netflix `.txt` files (alternating movie ID headers and user-rating lines)
- Filters out low-activity users (< 20 ratings) and low-coverage movies (< 50 ratings)
- Subsamples to 50,000 users and 5,000 movies by default for memory safety
- Builds a sparse user–movie interaction matrix in CSR format
- Splits the data per-user using a **temporal sort**: the most recent 20% becomes test, the next 10% becomes validation, and the remainder is training — this prevents data leakage
- Saves all outputs to `data/processed/` as Parquet and `.npz` files

**Outputs created:**
```
data/processed/ratings_clean.parquet
data/processed/train.parquet
data/processed/val.parquet
data/processed/test.parquet
data/processed/train_matrix.npz
```

> On first run this takes a few minutes. All subsequent runs load instantly from cache.

---

### Step 2 — Train Item-Based Collaborative Filtering

```bash
python src/collaborative_filtering.py
```

**What happens:**
- Loads the train matrix from `data/processed/`
- Computes per-user mean ratings (ignoring unrated movies)
- Mean-centres each user's ratings before computing similarity (adjusted cosine) — this removes individual rating bias
- Computes item-item cosine similarity in batches of 500 items at a time to manage memory
- Retains only the top-50 most similar neighbours per item (top-K pruning)
- Saves the similarity matrix to `data/processed/item_cf_similarity.pkl`
- Runs a demo: prints Top-10 recommendations for a sample user and outputs a human-readable explanation for the top recommendation

**Output created:**
```
data/processed/item_cf_similarity.pkl
```

> ⚠️ The first run computes the full item-item similarity matrix and takes **5–10 minutes**. All subsequent runs load from cache instantly. Delete the `.pkl` file only if you want to recompute from scratch.

---

### Step 3 — Train SVD (Matrix Factorization)

```bash
python src/svd_model.py
```

**What happens:**
- Loads train and validation DataFrames from `data/processed/`
- Converts the data into a Surprise `Trainset` object
- Trains SVD with 100 latent factors over 25 SGD epochs, with bias terms for users, items, and the global mean
- Logs training time and validation RMSE after fitting
- Saves the trained model to `results/models/`
- Runs a demo: generates Top-10 recommendations for a sample user

**Output created:**
```
results/models/svd_model.pkl
```

> Optional hyperparameter tuning via GridSearchCV is available but disabled by default for speed. See [Module Reference](#-module-reference) for details.

---

### Step 4 — Evaluate Both Models

```bash
python src/evaluation.py
```

**What happens:**
- Loads both trained models and the held-out test set
- **Rating prediction evaluation:** runs predictions on all (user, movie) pairs in the test set and computes RMSE and MAE
- **Ranking evaluation:** for each of 500 randomly sampled test users, scores all unseen movies and generates a Top-10 recommendation list, then computes MAP@10, Precision@10, Recall@10, and NDCG@10
- A movie is counted as **relevant** if its actual user rating is **≥ 3.5 stars**
- Prints a side-by-side comparison report to the console
- Saves full results to `results/evaluation_results.json`

**Output created:**
```
results/evaluation_results.json
```

---

## 📚 Module Reference

### `preprocessing.py`

Handles all data ingestion and preparation. Key settings you can change at the top of the file:

| Setting | Default | What it controls |
|---------|---------|-----------------|
| `MAX_USERS` | 50,000 | Cap on unique users loaded — reduce if RAM is limited |
| `MAX_MOVIES` | 5,000 | Cap on unique movies loaded |
| `MIN_USER_RATINGS` | 20 | Minimum ratings per user (filters inactive users) |
| `MIN_MOVIE_RATINGS` | 50 | Minimum ratings per movie (filters obscure titles) |
| `TEST_SIZE` | 0.20 | Fraction of each user's history held out for testing |
| `VALIDATION_SIZE` | 0.10 | Fraction used for validation |
| `use_subset` | `True` | Load only `combined_data_1.txt`; set `False` for all four files |

After `load()` runs, the following index maps are available for converting between raw IDs and 0-based matrix indices:

```
user2idx   → raw user ID  → matrix row index
movie2idx  → raw movie ID → matrix column index
idx2user   → matrix row index  → raw user ID
idx2movie  → matrix column index → raw movie ID
```

---

### `collaborative_filtering.py`

Item-Based Collaborative Filtering. Key settings:

| Setting | Default | What it controls |
|---------|---------|-----------------|
| `top_k_similar` | 50 | Number of neighbours retained per item — higher = more accurate but more memory |
| `min_common` | 5 | Minimum shared raters for a similarity to be considered valid |
| `use_adjusted` | `True` | Use adjusted cosine (recommended) vs raw cosine |
| `batch_size` | 500 | Items processed per similarity batch — reduce if you hit memory errors |

**How predictions work:** For a (user, movie) pair, the model looks at the movie's top-K similar neighbours, finds which of those the user has already rated, and computes a similarity-weighted average of those ratings. If no rated neighbours exist, it falls back to the user's mean rating.

**Explainability:** The `explain()` method outputs a human-readable string showing which previously liked movies drove the recommendation and their similarity scores.

---

### `svd_model.py`

Matrix Factorization via scikit-surprise. Key hyperparameters:

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `n_factors` | 100 | Number of latent dimensions — captures hidden taste dimensions |
| `n_epochs` | 25 | Number of SGD training passes over the data |
| `lr_all` | 0.005 | Learning rate for all parameters |
| `reg_all` | 0.1 | L2 regularisation strength — higher = less overfitting |
| `biased` | `True` | Whether to include user, item, and global bias terms |

**How predictions work:** The model learns a vector for each user and each movie in a low-dimensional latent space. A rating prediction is: global mean + user bias + item bias + dot product of user and movie latent vectors. This allows the model to generalise across users and movies it has seen, capturing patterns like genre affinity.

**Optional tuning:** Pass a `param_grid` dict to `model.tune()` to run GridSearchCV and find the best hyperparameters automatically. This is disabled by default but can be enabled for a potential boost in ranking quality.

---

### `evaluation.py`

Model-agnostic evaluation framework. Works with both Item-CF and SVD via a shared interface.

**Metrics explained:**

| Metric | Type | What it measures |
|--------|------|-----------------|
| **RMSE** | Accuracy | How close predicted ratings are to actual ratings (penalises large errors more) |
| **MAE** | Accuracy | Average absolute error between predicted and actual ratings |
| **MAP@10** | Ranking | How well the model ranks relevant items in the top-10 list, averaged over users |
| **Precision@10** | Ranking | What fraction of the top-10 recommendations are genuinely liked by the user |
| **Recall@10** | Ranking | What fraction of all movies the user liked were captured in the top-10 |
| **NDCG@10** | Ranking | Like Precision but rewards putting the best items higher in the list |

> **Relevance threshold:** A movie counts as "relevant" if the user actually rated it **≥ 3.5 stars**. This is fixed per the project specification.

**Key design:** Recommendations are generated only for movies the user has **not** seen in training, ensuring the evaluation reflects genuine discovery rather than re-surfacing known items.

---

## 📐 Evaluation Methodology

### Train / Validation / Test Split

```
Per-user temporal sort → most recent 20% = test | next 10% = val | rest = train
Users with < 5 ratings → placed entirely in train
```

This design **prevents data leakage**: the model learns from past interactions and is evaluated on genuinely future, unseen interactions — mirroring a real deployment scenario.

### Ranking Evaluation Pipeline

```
Step 1: Identify users with ≥ 1 relevant movie in test set (rating ≥ 3.5)
Step 2: Subsample 500 users for tractable evaluation
Step 3: For each user → score ALL unseen movies → return Top-10
Step 4: Compute MAP@10, Precision@10, Recall@10, NDCG@10
```

### Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Relevance threshold | 3.5 stars | Per problem specification; filters genuine preference signal |
| Evaluation users | 500 (sampled) | Balances evaluation quality with compute time |
| Unseen movies only | Yes | Ensures recommendations are actually novel |
| Temporal split | Yes | Mimics real recommendation scenario; prevents leakage |

---

## 📈 Results & Analysis

### Full Metric Table

| Metric | Item-CF (Adj. Cosine) | SVD (Surprise) | Winner |
|--------|----------------------|----------------|--------|
| RMSE ↓ | **0.9422** | 1.0529 | Item-CF |
| MAE ↓ | **0.7149** | 0.8852 | Item-CF |
| MAP@10 ↑ | 0.005689 | **0.013685** | SVD (+2.4×) |
| Precision@10 ↑ | 0.0170 | **0.0374** | SVD (+2.2×) |
| Recall@10 ↑ | 0.006733 | **0.01642** | SVD (+2.4×) |
| NDCG@10 ↑ | 0.017604 | **0.039113** | SVD (+2.2×) |
| Users Evaluated | 500 | 500 | — |
| RMSE Eval Time | 339.85 s | **8.80 s** | SVD (38×) |
| Ranking Eval Time | 266.95 s | **16.72 s** | SVD (16×) |

### The Rating Accuracy vs. Ranking Quality Trade-off

RMSE measures how close predictions are to actual ratings — uniformly, including low ratings. A model that accurately predicts 2-star ratings for bad movies scores well on RMSE but has no practical recommendation value.

MAP@10 only rewards models that surface genuinely liked content at the top. SVD's latent space captures the global structure of taste that local item-item similarity misses — hence 2.4× better ranking quality.

### Computational Profile

```
Item-CF:  ~607 s total evaluation  (339.85s RMSE + 266.95s ranking)
SVD:      ~25.5 s total evaluation  (8.80s RMSE  + 16.72s ranking)
```

At production scale, SVD is far more practical: each prediction is a single dot product operation, while Item-CF requires intersecting neighbour lists with a user's rating history at inference time.

---

## 🚀 Future Improvements

### Short-Term

- [ ] **SVD hyperparameter tuning** — run GridSearchCV over `n_factors`, `n_epochs`, `lr_all`, `reg_all` for a potential ranking boost
- [ ] **Full dataset training** — expand from `combined_data_1.txt` to all four files for richer coverage
- [ ] **Vectorised Item-CF prediction** — replace Python-level neighbour loop with NumPy batch operations; expected ~10× inference speedup
- [ ] **Min-common rater tuning** — experiment with higher `min_common` thresholds to reduce spurious similarity scores

### Medium-Term

- [ ] **Alternating Least Squares (ALS)** — parallelisable matrix factorisation; better suited for implicit feedback signals
- [ ] **Neural Collaborative Filtering (NCF)** — MLP over user/item embeddings; typically yields 2–5% NDCG improvement
- [ ] **Temporal dynamics** — down-weight older ratings; model how user tastes shift over time
- [ ] **Hybrid ensemble** — combine Item-CF similarity scores + SVD latent factors as features in a LightGBM ranker

### Long-Term

- [ ] **Graph Neural Networks (LightGCN)** — propagates preference signals over the user-item bipartite graph
- [ ] **Diversity & serendipity metrics** — avoid filter bubbles; reward novel-yet-relevant recommendations
- [ ] **Cold-start fallback** — genre + release year content similarity for new users and new movies with no rating history

---

## 📄 Citation

**Dataset citation:**
```
Netflix Prize Dataset. (2006). Netflix, Inc.
https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data
```

---

<div align="center">

**Built on the Netflix Prize Dataset**

`python` · `numpy` · `scipy` · `scikit-surprise` · `pandas` · `tqdm`

![RMSE](https://img.shields.io/badge/RMSE-0.9422-E50914?style=flat-square)
![MAP@10](https://img.shields.io/badge/MAP@10-0.01369-46D369?style=flat-square)
![Users Evaluated](https://img.shields.io/badge/Users%20Evaluated-500-141414?style=flat-square)

</div>
ENDOFFILE

# Recommendation Systems for Personalized Content Discovery

A movie recommendation system built on the Netflix Prize Dataset, comparing Item-Based Collaborative Filtering and SVD (Matrix Factorization) approaches.

---

## Dataset

Download the Netflix Prize Dataset from Kaggle:
[https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)

Place the following files inside a `data/` folder at the project root:

```
data/
├── combined_data_1.txt
├── combined_data_2.txt
├── combined_data_3.txt
├── combined_data_4.txt
└── movie_titles.csv
```

---

## Project Structure

```
├── data/                        # Raw dataset files (not committed)
│   └── processed/               # Auto-generated cached files
├── src/
│   ├── preprocessing.py         # Data loading, cleaning, train/test split
│   ├── collaborative_filtering.py  # Item-Based CF model
│   ├── svd_model.py             # SVD model via Surprise
│   └── evaluation.py            # RMSE, MAE, MAP@10, Precision, Recall, NDCG
└── README.md
```

---

## Setup

**Python 3.8+ required**

Install dependencies:

```bash
pip install numpy pandas scipy scikit-learn scikit-surprise tqdm
```

---

## How to Run

### 1. Preprocess the data

```bash
python src/preprocessing.py
```

This loads the raw Netflix files, filters users and movies, and saves train/validation/test splits to `data/processed/`.

### 2. Train and evaluate Item-Based CF

```bash
python src/collaborative_filtering.py
```

### 3. Train and evaluate SVD

```bash
python src/svd_model.py
```

### 4. Run evaluation on both models

```bash
python src/evaluation.py
```

Results are saved to `evaluation_results.json`.

---

## Results

| Model | RMSE | MAE | MAP@10 | Precision@10 | NDCG@10 |
|---|---|---|---|---|---|
| Item-CF (Adjusted Cosine) | 0.9422 | 0.7149 | 0.0057 | 0.0170 | 0.0176 |
| SVD (Surprise) | 1.0529 | 0.8852 | 0.0137 | 0.0374 | 0.0391 |

> A movie is considered **relevant** for ranking metrics if its actual user rating is ≥ 3.5.

---

## Notes

- The dataset is large (~100M ratings). By default, the preprocessing script caps users and movies for memory safety. You can adjust `MAX_USERS` and `MAX_MOVIES` in `preprocessing.py`.
- The item-item similarity matrix is cached to disk after the first run. Delete `data/processed/item_cf_similarity.pkl` to recompute.
- Evaluated on 500 users for ranking metrics.

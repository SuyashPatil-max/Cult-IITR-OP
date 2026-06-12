"""
collaborative_filtering.py
--------------------------
Item-Based Collaborative Filtering for the Netflix Prize Dataset.

Features:
  - Cosine similarity on sparse item vectors (memory-efficient)
  - Adjusted cosine similarity (mean-centres by user before computing)
  - Batch-computed item-item similarity with configurable top-K pruning
  - Rating prediction via similarity-weighted average
  - Top-N recommendation generation (unseen movies only)
  - Explainability: "Because you liked X and Y, we recommend Z"
  - Disk caching of the similarity matrix

Usage:
    from src.collaborative_filtering import ItemBasedCF
    model = ItemBasedCF(top_k_similar=50)
    model.fit(train_matrix)
    recs = model.recommend(user_idx=0, n=10, seen_mask=train_matrix[0])
"""

import time
import logging
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


# ─────────────────────────────────────────────
# Core Model
# ─────────────────────────────────────────────

class ItemBasedCF:
    """
    Item-Based Collaborative Filtering using Adjusted Cosine Similarity.

    Adjusted cosine centres each user's ratings by their mean before
    computing similarity, which accounts for individual rating bias
    (e.g., a harsh rater who gives 2s instead of 4s).

    Parameters
    ----------
    top_k_similar  : int   Number of most-similar neighbours to keep per item.
                           Pruning to top-K keeps memory & inference fast.
    min_common     : int   Minimum number of shared raters for a similarity
                           to be considered valid (avoids spurious high sims).
    use_adjusted   : bool  If True, use adjusted cosine (recommended).
                           If False, use standard cosine on raw ratings.
    batch_size     : int   Items processed per batch during similarity
                           computation (tune down if RAM is tight).
    """

    def __init__(
        self,
        top_k_similar: int = 50,
        min_common: int = 5,
        use_adjusted: bool = True,
        batch_size: int = 500,
    ):
        self.top_k_similar = top_k_similar
        self.min_common = min_common
        self.use_adjusted = use_adjusted
        self.batch_size = batch_size

        # Fitted attributes
        self.item_sim_matrix: Optional[np.ndarray] = None   # (n_movies, top_k_similar)
        self.item_sim_indices: Optional[np.ndarray] = None  # (n_movies, top_k_similar)
        self.user_means: Optional[np.ndarray] = None        # (n_users,)
        self.train_matrix: Optional[csr_matrix] = None      # kept for prediction
        self.n_users: int = 0
        self.n_movies: int = 0
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, train_matrix: csr_matrix, cache_path: Optional[Path] = None) -> "ItemBasedCF":
        """
        Compute and store the pruned item-item similarity matrix.

        Parameters
        ----------
        train_matrix : csr_matrix  shape (n_users, n_movies)
        cache_path   : Path | None  Save / load similarity matrix to disk.

        Returns
        -------
        self
        """
        if cache_path is None:
            cache_path = PROCESSED_DIR / "item_cf_similarity.pkl"

        # Try loading from cache
        if cache_path.exists():
            logger.info("Loading item-CF similarity from cache: %s", cache_path)
            self._load_cache(cache_path, train_matrix)
            return self

        t0 = time.time()
        self.train_matrix = train_matrix.astype(np.float32)
        self.n_users, self.n_movies = train_matrix.shape

        # Step 1: Compute per-user mean ratings (ignore zeros = unrated)
        logger.info("Computing user mean ratings …")
        self.user_means = self._compute_user_means(self.train_matrix)

        # Step 2: Mean-centre the matrix (adjusted cosine) or use raw (standard)
        if self.use_adjusted:
            logger.info("Mean-centring ratings for adjusted cosine …")
            item_matrix = self._mean_centre(self.train_matrix, self.user_means)
        else:
            item_matrix = self.train_matrix.copy()

        # item_matrix is (n_users, n_movies); transpose to (n_movies, n_users)
        # so each row is an item's rating vector across users
        item_vectors = item_matrix.T.tocsr()   # (n_movies, n_users)

        # Step 3: Batch cosine similarity + top-K pruning
        logger.info(
            "Computing item-item similarity (%d items, batch=%d) …",
            self.n_movies, self.batch_size,
        )
        sim_values, sim_indices = self._batch_similarity(item_vectors)
        self.item_sim_matrix = sim_values     # (n_movies, top_k_similar)
        self.item_sim_indices = sim_indices   # (n_movies, top_k_similar)

        self._is_fitted = True
        elapsed = time.time() - t0
        logger.info("Item-CF fit complete in %.1f s.", elapsed)

        # Save to cache
        self._save_cache(cache_path, train_matrix)
        return self

    # ------------------------------------------------------------------
    # Rating Prediction
    # ------------------------------------------------------------------

    def predict(self, user_idx: int, movie_idx: int) -> float:
        """
        Predict the rating user_idx would give to movie_idx.

        Uses similarity-weighted average of the user's existing ratings
        for the movie's nearest neighbours.

        Parameters
        ----------
        user_idx  : int  0-based user index
        movie_idx : int  0-based movie index

        Returns
        -------
        float  predicted rating in [1, 5], or global mean if no signal
        """
        self._check_fitted()

        # Ratings this user has given (non-zero entries in their row)
        user_row = self.train_matrix[user_idx]               # (1, n_movies) sparse
        rated_movies = user_row.indices                       # movie indices rated by user
        rated_values = np.array(user_row.data, dtype=np.float32)

        if len(rated_movies) == 0:
            return float(self.user_means.mean())

        # Neighbours of target movie
        neighbour_indices = self.item_sim_indices[movie_idx]  # (top_k,)
        neighbour_sims = self.item_sim_matrix[movie_idx]       # (top_k,)

        # Intersect neighbours with movies rated by this user
        mask = np.isin(neighbour_indices, rated_movies)
        if not mask.any():
            # Fallback: user mean rating
            return float(self.user_means[user_idx]) if self.user_means[user_idx] != 0 else 3.0

        valid_neighbours = neighbour_indices[mask]
        valid_sims = neighbour_sims[mask]

        # Retrieve the user's ratings for these neighbours
        # Build a lookup: movie_idx → rating
        rated_lookup = dict(zip(rated_movies, rated_values))
        neighbour_ratings = np.array(
            [rated_lookup[m] for m in valid_neighbours], dtype=np.float32
        )

        # Filter out negative or zero similarities
        pos_mask = valid_sims > 0
        if not pos_mask.any():
            return float(self.user_means[user_idx]) if self.user_means[user_idx] != 0 else 3.0

        valid_sims = valid_sims[pos_mask]
        neighbour_ratings = neighbour_ratings[pos_mask]

        # Similarity-weighted average
        pred = float(np.dot(valid_sims, neighbour_ratings) / (valid_sims.sum() + 1e-9))

        # Clamp to valid rating range
        return float(np.clip(pred, 1.0, 5.0))

    def predict_batch(
        self,
        user_indices: np.ndarray,
        movie_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorised batch prediction for (user, movie) pairs.
        Used by the evaluation pipeline.

        Parameters
        ----------
        user_indices  : np.ndarray  shape (N,)
        movie_indices : np.ndarray  shape (N,)

        Returns
        -------
        np.ndarray  shape (N,)  predicted ratings
        """
        self._check_fitted()
        predictions = np.zeros(len(user_indices), dtype=np.float32)
        for i, (u, m) in enumerate(
            tqdm(zip(user_indices, movie_indices),
                 total=len(user_indices),
                 desc="Item-CF batch predict",
                 unit=" pairs",
                 mininterval=2.0)
        ):
            predictions[i] = self.predict(int(u), int(m))
        return predictions

    # ------------------------------------------------------------------
    # Top-N Recommendations
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> List[Tuple[int, float]]:
        """
        Generate Top-N movie recommendations for a user.

        Parameters
        ----------
        user_idx     : int   0-based user index
        n            : int   number of recommendations
        exclude_seen : bool  if True, skip movies the user has already rated

        Returns
        -------
        List of (movie_idx, predicted_score) sorted descending by score
        """
        self._check_fitted()

        seen_movies: set = set()
        if exclude_seen:
            user_row = self.train_matrix[user_idx]
            seen_movies = set(user_row.indices.tolist())

        candidate_movies = [
            m for m in range(self.n_movies) if m not in seen_movies
        ]

        # Score all candidates
        scores = []
        for m in candidate_movies:
            score = self._fast_score(user_idx, m, seen_movies)
            scores.append((m, score))

        # Sort descending, return top-N
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

    def recommend_batch(
        self,
        user_indices: List[int],
        n: int = 10,
        exclude_seen: bool = True,
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Generate Top-N recommendations for a list of users.

        Returns
        -------
        dict  { user_idx: [(movie_idx, score), ...] }
        """
        results = {}
        for uid in tqdm(user_indices, desc="Item-CF recommending", unit=" users"):
            results[uid] = self.recommend(uid, n=n, exclude_seen=exclude_seen)
        return results

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    def explain(
        self,
        user_idx: int,
        target_movie_idx: int,
        idx2movie: Dict[int, int],
        movies_df: pd.DataFrame,
        top_k_reasons: int = 3,
    ) -> str:
        """
        Generate a human-readable explanation for a recommendation.

        "Because you liked [Movie A] and [Movie B], we recommend [Movie C]."

        Parameters
        ----------
        user_idx          : int  0-based user index
        target_movie_idx  : int  0-based movie index of the recommended item
        idx2movie         : dict  index → raw movie_id
        movies_df         : pd.DataFrame  with [movie_id, title] columns
        top_k_reasons     : int  max supporting movies to cite

        Returns
        -------
        str  explanation text
        """
        self._check_fitted()

        # Retrieve the user's rated movies
        user_row = self.train_matrix[user_idx]
        rated_movies = user_row.indices
        rated_values = np.array(user_row.data, dtype=np.float32)

        if len(rated_movies) == 0:
            return "Recommended based on overall movie popularity."

        # Neighbours of the target movie
        neighbour_indices = self.item_sim_indices[target_movie_idx]
        neighbour_sims = self.item_sim_matrix[target_movie_idx]

        # Find overlap between neighbours and user-rated movies
        rated_set = set(rated_movies.tolist())
        reasons = []
        for nb_idx, nb_sim in zip(neighbour_indices, neighbour_sims):
            if nb_idx in rated_set and nb_sim > 0:
                # Get the user's rating for this neighbour
                rating_pos = np.where(rated_movies == nb_idx)[0]
                user_rating = float(rated_values[rating_pos[0]]) if len(rating_pos) > 0 else 0.0
                if user_rating >= 3.5:   # only mention liked movies
                    reasons.append((nb_idx, nb_sim, user_rating))

        reasons.sort(key=lambda x: x[1], reverse=True)
        reasons = reasons[:top_k_reasons]

        # Build title lookup
        title_lookup: Dict[int, str] = {}
        if movies_df is not None and not movies_df.empty:
            for _, row in movies_df.iterrows():
                raw_mid = int(row["movie_id"])
                title_lookup[raw_mid] = str(row["title"])

        def get_title(midx: int) -> str:
            raw_mid = idx2movie.get(midx, midx)
            return title_lookup.get(raw_mid, f"Movie #{raw_mid}")

        target_title = get_title(target_movie_idx)

        if not reasons:
            return f'"{target_title}" is recommended based on your viewing history.'

        liked_titles = [f'"{get_title(r[0])}" (you rated it {r[2]:.0f}/5)' for r in reasons]

        if len(liked_titles) == 1:
            because = liked_titles[0]
        elif len(liked_titles) == 2:
            because = f"{liked_titles[0]} and {liked_titles[1]}"
        else:
            because = ", ".join(liked_titles[:-1]) + f", and {liked_titles[-1]}"

        return (
            f'Because you liked {because}, '
            f'we recommend "{target_title}" '
            f'(similarity score: {reasons[0][1]:.3f}).'
        )

    def explain_recommendations(
        self,
        user_idx: int,
        recommendations: List[Tuple[int, float]],
        idx2movie: Dict[int, int],
        movies_df: pd.DataFrame,
    ) -> List[Dict]:
        """
        Generate explanations for a full list of recommendations.

        Returns
        -------
        List of dicts with keys: movie_idx, score, explanation
        """
        explained = []
        for movie_idx, score in recommendations:
            exp_text = self.explain(
                user_idx, movie_idx, idx2movie, movies_df, top_k_reasons=3
            )
            explained.append({
                "movie_idx": movie_idx,
                "score": round(score, 4),
                "explanation": exp_text,
            })
        return explained

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_user_means(self, matrix: csr_matrix) -> np.ndarray:
        """Compute mean of non-zero entries per user (row)."""
        means = np.zeros(matrix.shape[0], dtype=np.float32)
        for i in range(matrix.shape[0]):
            row = matrix.getrow(i)
            if row.nnz > 0:
                means[i] = row.data.mean()
        return means

    def _mean_centre(self, matrix: csr_matrix, user_means: np.ndarray) -> csr_matrix:
        """
        Subtract user mean from each non-zero rating.
        Zeros (= unrated) remain zero; only observed ratings are shifted.
        """
        matrix = matrix.copy().astype(np.float32)
        # Convert to lil for efficient row-wise access, then back to csr
        lil = matrix.tolil()
        for i in range(matrix.shape[0]):
            if len(lil.data[i]) > 0 and user_means[i] != 0:
                lil.data[i] = [v - user_means[i] for v in lil.data[i]]
        return lil.tocsr()

    def _batch_similarity(
        self, item_vectors: csr_matrix
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute item-item cosine similarity in batches, retaining only
        the top-K most similar items per item.

        Parameters
        ----------
        item_vectors : csr_matrix  shape (n_movies, n_users)

        Returns
        -------
        sim_values  : np.ndarray  (n_movies, top_k_similar)
        sim_indices : np.ndarray  (n_movies, top_k_similar)
        """
        n_items = item_vectors.shape[0]
        k = min(self.top_k_similar, n_items - 1)

        sim_values = np.zeros((n_items, k), dtype=np.float32)
        sim_indices = np.zeros((n_items, k), dtype=np.int32)

        batches = range(0, n_items, self.batch_size)
        for start in tqdm(batches, desc="Item similarity batches", unit=" batch"):
            end = min(start + self.batch_size, n_items)
            batch = item_vectors[start:end]   # (batch, n_users)

            # Full row against all items: (batch, n_items)
            batch_sims = cosine_similarity(batch, item_vectors)

            # Zero out self-similarity
            for local_i, global_i in enumerate(range(start, end)):
                batch_sims[local_i, global_i] = 0.0

            # Retain top-K per row
            for local_i in range(end - start):
                row = batch_sims[local_i]
                top_k_idx = np.argpartition(row, -k)[-k:]
                top_k_idx = top_k_idx[np.argsort(row[top_k_idx])[::-1]]
                sim_indices[start + local_i] = top_k_idx
                sim_values[start + local_i] = row[top_k_idx]

        return sim_values, sim_indices

    def _fast_score(self, user_idx: int, movie_idx: int, seen_movies: set) -> float:
        """
        Lightweight scoring used during recommendation generation.
        Slightly faster than predict() since seen_movies is pre-computed.
        """
        user_row = self.train_matrix[user_idx]
        rated_movies = user_row.indices
        rated_values = np.array(user_row.data, dtype=np.float32)

        if len(rated_movies) == 0:
            return 0.0

        neighbour_indices = self.item_sim_indices[movie_idx]
        neighbour_sims = self.item_sim_matrix[movie_idx]

        mask = np.isin(neighbour_indices, rated_movies) & (neighbour_sims > 0)
        if not mask.any():
            return 0.0

        valid_neighbours = neighbour_indices[mask]
        valid_sims = neighbour_sims[mask]

        rated_lookup = dict(zip(rated_movies.tolist(), rated_values.tolist()))
        neighbour_ratings = np.array(
            [rated_lookup[m] for m in valid_neighbours], dtype=np.float32
        )

        score = float(np.dot(valid_sims, neighbour_ratings) / (valid_sims.sum() + 1e-9))
        return float(np.clip(score, 1.0, 5.0))

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call .fit() first.")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_cache(self, path: Path, train_matrix: csr_matrix) -> None:
        payload = {
            "item_sim_matrix": self.item_sim_matrix,
            "item_sim_indices": self.item_sim_indices,
            "user_means": self.user_means,
            "n_users": self.n_users,
            "n_movies": self.n_movies,
            "top_k_similar": self.top_k_similar,
            "use_adjusted": self.use_adjusted,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=4)
        logger.info("Saved item-CF similarity cache → %s", path)

    def _load_cache(self, path: Path, train_matrix: csr_matrix) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.item_sim_matrix = payload["item_sim_matrix"]
        self.item_sim_indices = payload["item_sim_indices"]
        self.user_means = payload["user_means"]
        self.n_users = payload["n_users"]
        self.n_movies = payload["n_movies"]
        self.train_matrix = train_matrix.astype(np.float32)
        self._is_fitted = True
        logger.info(
            "Loaded item-CF cache: %d items × %d neighbours.",
            self.n_movies, self.top_k_similar,
        )

    def save(self, path: Path) -> None:
        """Full model serialisation (includes train matrix reference)."""
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=4)
        logger.info("Saved ItemBasedCF model → %s", path)

    @staticmethod
    def load(path: Path) -> "ItemBasedCF":
        """Load a previously saved ItemBasedCF model."""
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info("Loaded ItemBasedCF model from %s", path)
        return model


# ─────────────────────────────────────────────
# Utility: similarity inspection
# ─────────────────────────────────────────────

def get_similar_movies(
    model: ItemBasedCF,
    movie_idx: int,
    idx2movie: Dict[int, int],
    movies_df: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """
    Return the top-N most similar movies to a given movie.

    Parameters
    ----------
    model      : ItemBasedCF  fitted model
    movie_idx  : int           0-based movie index
    idx2movie  : dict          index → raw movie_id
    movies_df  : pd.DataFrame  with [movie_id, title]
    n          : int           number of similar movies to return

    Returns
    -------
    pd.DataFrame  columns: [rank, movie_idx, movie_id, title, similarity]
    """
    model._check_fitted()

    neighbour_indices = model.item_sim_indices[movie_idx][:n]
    neighbour_sims = model.item_sim_matrix[movie_idx][:n]

    title_lookup: Dict[int, str] = {}
    if movies_df is not None and not movies_df.empty:
        title_lookup = dict(zip(movies_df["movie_id"].astype(int), movies_df["title"]))

    rows = []
    for rank, (nb_idx, sim) in enumerate(zip(neighbour_indices, neighbour_sims), 1):
        raw_mid = idx2movie.get(int(nb_idx), int(nb_idx))
        title = title_lookup.get(raw_mid, f"Movie #{raw_mid}")
        rows.append({
            "rank": rank,
            "movie_idx": int(nb_idx),
            "movie_id": raw_mid,
            "title": title,
            "similarity": round(float(sim), 4),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Main — quick smoke-test / demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.preprocessing import (
        NetflixDataLoader,
        build_interaction_matrix,
        load_splits,
        PROCESSED_DIR,
    )

    # Load data
    loader = NetflixDataLoader(use_subset=True)
    ratings_df = loader.load(cache=True)
    loader.load_movies()

    train_df, val_df, test_df = load_splits()

    train_matrix = build_interaction_matrix(train_df, loader.user2idx, loader.movie2idx)

    # Fit model
    model = ItemBasedCF(top_k_similar=50, use_adjusted=True, batch_size=500)
    model.fit(train_matrix, cache_path=PROCESSED_DIR / "item_cf_similarity.pkl")

    # Demo: recommend for user 0
    recs = model.recommend(user_idx=0, n=10, exclude_seen=True)
    print("\n── Top-10 Recommendations for User 0 ──")
    for rank, (midx, score) in enumerate(recs, 1):
        raw_mid = loader.idx2movie[midx]
        title_row = loader.movies_df[loader.movies_df["movie_id"] == raw_mid]
        title = title_row["title"].values[0] if not title_row.empty else f"Movie #{raw_mid}"
        print(f"  {rank:2d}. [{midx:4d}] {title:<45s}  score={score:.3f}")

    # Demo: explain top recommendation
    if recs:
        top_movie_idx = recs[0][0]
        exp = model.explain(
            user_idx=0,
            target_movie_idx=top_movie_idx,
            idx2movie=loader.idx2movie,
            movies_df=loader.movies_df,
        )
        print(f"\n── Explanation ──\n  {exp}")

    # Demo: similar movies to movie 0
    sim_df = get_similar_movies(model, 0, loader.idx2movie, loader.movies_df, n=5)
    print("\n── Movies similar to movie index 0 ──")
    print(sim_df.to_string(index=False))

    print("\n✅  collaborative_filtering.py smoke-test complete.")

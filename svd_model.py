"""
svd_model.py
------------
Matrix Factorization via Surprise SVD for the Netflix Prize Dataset.

Features:
  - SVD with tunable hyperparameters (n_factors, n_epochs, lr, reg)
  - GridSearchCV-based hyperparameter tuning (optional, time-gated)
  - Training pipeline with timing and progress logging
  - Prediction pipeline (single + batch)
  - Top-N recommendation generation (unseen movies only)
  - Full model serialisation / deserialisation
  - Surprise Trainset ↔ DataFrame conversion utilities

Usage:
    from src.svd_model import SVDModel
    model = SVDModel()
    model.fit(train_df)
    recs = model.recommend(user_id=12345, n=10, all_movie_ids=movie_ids, seen_movie_ids=seen)
"""

import time
import logging
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("results/models")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# ─────────────────────────────────────────────
# Default Hyperparameters
# (tuned for RMSE on Netflix-scale data)
# ─────────────────────────────────────────────
DEFAULT_PARAMS = {
    "n_factors": 100,      # latent dimension size
    "n_epochs": 25,        # SGD passes over training data
    "lr_all": 0.005,       # learning rate for all parameters
    "reg_all": 0.1,        # L2 regularisation for all parameters
    "biased": True,        # include user/item/global bias terms
    "random_state": RANDOM_SEED,
    "verbose": False,
}

# Grid to search during optional hyperparameter tuning
TUNING_GRID = {
    "n_factors": [50, 100, 150],
    "n_epochs": [20, 30],
    "lr_all": [0.003, 0.005, 0.007],
    "reg_all": [0.05, 0.1, 0.2],
}


# ─────────────────────────────────────────────
# Core Model
# ─────────────────────────────────────────────

class SVDModel:
    """
    Wrapper around Surprise's SVD algorithm for the Netflix Prize task.

    The underlying algorithm uses stochastic gradient descent to minimise:
        min  Σ (r_ui - μ - b_u - b_i - p_u · q_i)²  +  λ(||p_u||² + ||q_i||²)

    where:
        μ   = global mean rating
        b_u = user bias
        b_i = item bias
        p_u = user latent factor vector  (n_factors,)
        q_i = item latent factor vector  (n_factors,)

    Parameters
    ----------
    params : dict | None
        SVD hyperparameters. Defaults to DEFAULT_PARAMS if None.
    """

    def __init__(self, params: Optional[Dict] = None):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.algo = None          # surprise.SVD instance (set after fit)
        self.trainset = None      # surprise Trainset (kept for inner-id lookup)
        self.train_df: Optional[pd.DataFrame] = None
        self._is_fitted: bool = False
        self._train_time: float = 0.0

        # Raw-id caches built at fit time
        self._all_raw_iids: Optional[List] = None   # all movie raw ids seen in training

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "SVDModel":
        """
        Build a Surprise Trainset from train_df and fit SVD.

        Parameters
        ----------
        train_df : pd.DataFrame  columns: [user_id, movie_id, rating]

        Returns
        -------
        self
        """
        try:
            from surprise import SVD, Dataset, Reader
        except ImportError:
            raise ImportError(
                "scikit-surprise is required. Install with: pip install scikit-surprise"
            )

        logger.info(
            "Building Surprise Trainset from %d ratings (%d users, %d movies) …",
            len(train_df),
            train_df["user_id"].nunique(),
            train_df["movie_id"].nunique(),
        )

        reader = Reader(rating_scale=(1, 5))
        dataset = Dataset.load_from_df(
            train_df[["user_id", "movie_id", "rating"]], reader
        )
        self.trainset = dataset.build_full_trainset()
        self.train_df = train_df.copy()

        # Cache all raw item ids for recommendation generation
        self._all_raw_iids = [
            self.trainset.to_raw_iid(inner_iid)
            for inner_iid in self.trainset.all_items()
        ]

        logger.info(
            "Fitting SVD: n_factors=%d, n_epochs=%d, lr=%.4f, reg=%.3f …",
            self.params["n_factors"],
            self.params["n_epochs"],
            self.params["lr_all"],
            self.params["reg_all"],
        )

        self.algo = SVD(
            n_factors=self.params["n_factors"],
            n_epochs=self.params["n_epochs"],
            lr_all=self.params["lr_all"],
            reg_all=self.params["reg_all"],
            biased=self.params["biased"],
            random_state=self.params["random_state"],
            verbose=self.params["verbose"],
        )

        t0 = time.time()
        self.algo.fit(self.trainset)
        self._train_time = time.time() - t0
        self._is_fitted = True

        logger.info("SVD training complete in %.1f s.", self._train_time)
        return self

    # ------------------------------------------------------------------
    # Hyperparameter Tuning (optional)
    # ------------------------------------------------------------------

    def tune(
        self,
        train_df: pd.DataFrame,
        param_grid: Optional[Dict] = None,
        cv: int = 3,
        measures: List[str] = ["rmse", "mae"],
        n_jobs: int = -1,
        time_limit_minutes: float = 15.0,
    ) -> Dict:
        """
        Run GridSearchCV over param_grid and store best params.

        Parameters
        ----------
        train_df         : pd.DataFrame  training data
        param_grid       : dict | None   grid to search; defaults to TUNING_GRID
        cv               : int           number of CV folds
        measures         : list          Surprise metrics to optimise
        n_jobs           : int           parallel jobs (-1 = all cores)
        time_limit_minutes : float       soft time budget (warns if exceeded)

        Returns
        -------
        dict  best hyperparameters found
        """
        try:
            from surprise import Dataset, Reader
            from surprise.model_selection import GridSearchCV as SurpriseGridSearchCV
            from surprise import SVD
        except ImportError:
            raise ImportError("scikit-surprise is required.")

        if param_grid is None:
            param_grid = TUNING_GRID

        logger.info(
            "Starting GridSearchCV: %d combinations × %d folds …",
            _count_grid(param_grid), cv,
        )

        reader = Reader(rating_scale=(1, 5))
        dataset = Dataset.load_from_df(
            train_df[["user_id", "movie_id", "rating"]], reader
        )

        t0 = time.time()
        gs = SurpriseGridSearchCV(
            SVD,
            param_grid,
            measures=measures,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
            joblib_verbose=0,
        )
        gs.fit(dataset)
        elapsed = (time.time() - t0) / 60.0

        if elapsed > time_limit_minutes:
            logger.warning("Tuning took %.1f min (budget was %.1f min).", elapsed, time_limit_minutes)

        best_params = gs.best_params["rmse"]
        best_score = gs.best_score["rmse"]

        logger.info(
            "Best RMSE=%.4f | params=%s | elapsed=%.1f min",
            best_score, best_params, elapsed,
        )

        # Merge best params into model config and refit
        self.params.update(best_params)
        logger.info("Refitting with best params …")
        self.fit(train_df)

        return best_params

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, user_id, movie_id) -> float:
        """
        Predict the rating user_id would give to movie_id.

        Parameters
        ----------
        user_id  : raw user identifier (int or str)
        movie_id : raw movie identifier (int or str)

        Returns
        -------
        float  predicted rating in [1, 5]
        """
        self._check_fitted()
        pred = self.algo.predict(uid=str(user_id), iid=str(movie_id), clip=True)
        return float(pred.est)

    def predict_batch(
        self,
        user_ids: np.ndarray,
        movie_ids: np.ndarray,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Batch-predict ratings for aligned (user_id, movie_id) arrays.

        Parameters
        ----------
        user_ids  : array-like  raw user ids, shape (N,)
        movie_ids : array-like  raw movie ids, shape (N,)

        Returns
        -------
        np.ndarray  shape (N,)  predicted ratings
        """
        self._check_fitted()
        pairs = zip(user_ids, movie_ids)
        if show_progress:
            pairs = tqdm(
                pairs,
                total=len(user_ids),
                desc="SVD batch predict",
                unit=" pairs",
                mininterval=2.0,
            )

        predictions = np.array(
            [float(self.algo.predict(str(u), str(m), clip=True).est) for u, m in pairs],
            dtype=np.float32,
        )
        return predictions

    def predict_df(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a 'predicted_rating' column to a ratings DataFrame.

        Parameters
        ----------
        ratings_df : pd.DataFrame  columns: [user_id, movie_id, rating]

        Returns
        -------
        pd.DataFrame  original + 'predicted_rating' column
        """
        self._check_fitted()
        preds = self.predict_batch(
            ratings_df["user_id"].values,
            ratings_df["movie_id"].values,
        )
        out = ratings_df.copy()
        out["predicted_rating"] = preds
        return out

    # ------------------------------------------------------------------
    # Top-N Recommendations
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_id,
        n: int = 10,
        all_movie_ids: Optional[List] = None,
        seen_movie_ids: Optional[set] = None,
    ) -> List[Tuple[int, float]]:
        """
        Generate Top-N recommendations for a single user.

        Parameters
        ----------
        user_id        : raw user identifier
        n              : number of recommendations
        all_movie_ids  : list of raw movie ids to score
                         (defaults to all training movie ids)
        seen_movie_ids : set of raw movie ids to exclude

        Returns
        -------
        List of (raw_movie_id, predicted_score) sorted descending
        """
        self._check_fitted()

        candidates = all_movie_ids if all_movie_ids is not None else self._all_raw_iids
        if seen_movie_ids:
            candidates = [m for m in candidates if m not in seen_movie_ids]

        scores = [
            (mid, float(self.algo.predict(str(user_id), str(mid), clip=True).est))
            for mid in candidates
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

    def recommend_batch(
        self,
        user_ids: List,
        n: int = 10,
        train_df: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Generate Top-N recommendations for a list of users.

        Parameters
        ----------
        user_ids  : list of raw user ids
        n         : number of recommendations per user
        train_df  : pd.DataFrame | None  if provided, exclude seen movies per user

        Returns
        -------
        dict  { user_id: [(movie_id, score), ...] }
        """
        self._check_fitted()

        # Build seen-movies lookup once
        seen_lookup: Dict = {}
        if train_df is not None:
            grouped = train_df.groupby("user_id")["movie_id"].apply(set)
            seen_lookup = grouped.to_dict()

        results = {}
        for uid in tqdm(user_ids, desc="SVD recommending", unit=" users"):
            seen = seen_lookup.get(uid, set())
            results[uid] = self.recommend(
                user_id=uid,
                n=n,
                seen_movie_ids=seen,
            )
        return results

    # ------------------------------------------------------------------
    # Latent Factor Inspection
    # ------------------------------------------------------------------

    def get_user_factors(self, user_id) -> Optional[np.ndarray]:
        """
        Return the latent factor vector for a user.

        Returns
        -------
        np.ndarray  shape (n_factors,) or None if user is unknown
        """
        self._check_fitted()
        try:
            inner_uid = self.trainset.to_inner_uid(str(user_id))
            return self.algo.pu[inner_uid]
        except ValueError:
            logger.warning("User %s not in training set.", user_id)
            return None

    def get_item_factors(self, movie_id) -> Optional[np.ndarray]:
        """
        Return the latent factor vector for a movie.

        Returns
        -------
        np.ndarray  shape (n_factors,) or None if movie is unknown
        """
        self._check_fitted()
        try:
            inner_iid = self.trainset.to_inner_iid(str(movie_id))
            return self.algo.qi[inner_iid]
        except ValueError:
            logger.warning("Movie %s not in training set.", movie_id)
            return None

    def get_user_bias(self, user_id) -> Optional[float]:
        """Return the learned bias term for a user."""
        self._check_fitted()
        try:
            inner_uid = self.trainset.to_inner_uid(str(user_id))
            return float(self.algo.bu[inner_uid])
        except ValueError:
            return None

    def get_item_bias(self, movie_id) -> Optional[float]:
        """Return the learned bias term for a movie."""
        self._check_fitted()
        try:
            inner_iid = self.trainset.to_inner_iid(str(movie_id))
            return float(self.algo.bi[inner_iid])
        except ValueError:
            return None

    def get_global_mean(self) -> float:
        """Return the global mean rating learned during training."""
        self._check_fitted()
        return float(self.algo.trainset.global_mean)

    # ------------------------------------------------------------------
    # Cross-validation helper
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        ratings_df: pd.DataFrame,
        cv: int = 5,
        measures: List[str] = ["rmse", "mae"],
        n_jobs: int = 1,
    ) -> pd.DataFrame:
        """
        Run k-fold cross-validation on ratings_df and return result table.

        Parameters
        ----------
        ratings_df : pd.DataFrame
        cv         : int  number of folds
        measures   : list of Surprise metrics
        n_jobs     : int  parallel jobs

        Returns
        -------
        pd.DataFrame  per-fold metrics + mean/std summary
        """
        try:
            from surprise import SVD, Dataset, Reader
            from surprise.model_selection import cross_validate as surprise_cv
        except ImportError:
            raise ImportError("scikit-surprise is required.")

        reader = Reader(rating_scale=(1, 5))
        dataset = Dataset.load_from_df(
            ratings_df[["user_id", "movie_id", "rating"]], reader
        )
        algo = SVD(**{k: v for k, v in self.params.items()})

        logger.info("Running %d-fold cross-validation …", cv)
        cv_results = surprise_cv(
            algo, dataset, measures=measures, cv=cv, n_jobs=n_jobs, verbose=False
        )

        rows = []
        for fold in range(cv):
            row = {"fold": fold + 1}
            for m in measures:
                row[m.upper()] = round(cv_results[f"test_{m}"][fold], 4)
            rows.append(row)

        df = pd.DataFrame(rows)
        summary_row = {"fold": "mean±std"}
        for m in measures:
            vals = df[m.upper()]
            summary_row[m.upper()] = f"{vals.mean():.4f} ± {vals.std():.4f}"
        df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

        logger.info("Cross-validation results:\n%s", df.to_string(index=False))
        return df

    # ------------------------------------------------------------------
    # Model info
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        """Return a summary dict of the model configuration and fit status."""
        info = {
            "model": "Surprise SVD",
            "is_fitted": self._is_fitted,
            "train_time_sec": round(self._train_time, 2),
            **{f"param_{k}": v for k, v in self.params.items()},
        }
        if self._is_fitted and self.trainset is not None:
            info["n_train_users"] = self.trainset.n_users
            info["n_train_items"] = self.trainset.n_items
            info["n_train_ratings"] = self.trainset.n_ratings
            info["global_mean"] = round(self.get_global_mean(), 4)
        return info

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        """
        Serialise the full model to disk.

        Parameters
        ----------
        path : Path | None  defaults to results/models/svd_model.pkl

        Returns
        -------
        Path  where the model was saved
        """
        if path is None:
            path = MODELS_DIR / "svd_model.pkl"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=4)
        logger.info("SVD model saved → %s", path)
        return path

    @staticmethod
    def load(path: Optional[Path] = None) -> "SVDModel":
        """
        Load a serialised SVDModel from disk.

        Parameters
        ----------
        path : Path | None  defaults to results/models/svd_model.pkl

        Returns
        -------
        SVDModel
        """
        if path is None:
            path = MODELS_DIR / "svd_model.pkl"
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info("SVD model loaded from %s", path)
        return model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call .fit() first.")


# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────

def _count_grid(param_grid: Dict) -> int:
    """Count total combinations in a hyperparameter grid."""
    count = 1
    for v in param_grid.values():
        count *= len(v)
    return count


def build_antitest_set(
    trainset,
    test_df: pd.DataFrame,
) -> List[Tuple]:
    """
    Build a Surprise anti-testset restricted to (user, movie) pairs
    present in test_df. Used by Surprise's evaluate() pipeline.

    Parameters
    ----------
    trainset : surprise.Trainset
    test_df  : pd.DataFrame  with [user_id, movie_id, rating]

    Returns
    -------
    list of (raw_uid, raw_iid, global_mean) tuples
    """
    global_mean = trainset.global_mean
    anti = []
    for _, row in test_df.iterrows():
        raw_uid = str(row["user_id"])
        raw_iid = str(row["movie_id"])
        try:
            trainset.to_inner_uid(raw_uid)
            trainset.to_inner_iid(raw_iid)
            anti.append((raw_uid, raw_iid, global_mean))
        except ValueError:
            pass   # cold-start user or item — skip
    return anti


def ratings_df_to_surprise(ratings_df: pd.DataFrame):
    """
    Convert a ratings DataFrame to a Surprise (trainset, testset) pair
    suitable for .test() evaluation.

    Returns
    -------
    (trainset, testset)
    """
    try:
        from surprise import Dataset, Reader
    except ImportError:
        raise ImportError("scikit-surprise is required.")

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        ratings_df[["user_id", "movie_id", "rating"]], reader
    )
    trainset = data.build_full_trainset()
    testset = trainset.build_testset()
    return trainset, testset


# ─────────────────────────────────────────────
# Main — smoke-test / demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.preprocessing import NetflixDataLoader, load_splits

    # Load data
    loader = NetflixDataLoader(use_subset=True)
    loader.load(cache=True)
    loader.load_movies()
    train_df, val_df, test_df = load_splits()

    # Fit SVD
    model = SVDModel(params={"n_factors": 100, "n_epochs": 20})
    model.fit(train_df)

    # Print model summary
    summary = model.summary()
    print("\n── SVD Model Summary ──")
    for k, v in summary.items():
        print(f"  {k:<30s}: {v}")

    # Batch-predict on a sample of the test set
    sample = test_df.sample(n=min(500, len(test_df)), random_state=RANDOM_SEED)
    preds = model.predict_batch(sample["user_id"].values, sample["movie_id"].values)
    actuals = sample["rating"].values
    rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    print(f"\n  Sample RMSE on {len(sample)} test pairs: {rmse:.4f}")

    # Top-10 recommendations for a random user
    sample_user = train_df["user_id"].sample(1, random_state=RANDOM_SEED).values[0]
    seen = set(train_df[train_df["user_id"] == sample_user]["movie_id"].tolist())
    all_movies = train_df["movie_id"].unique().tolist()
    recs = model.recommend(user_id=sample_user, n=10, all_movie_ids=all_movies, seen_movie_ids=seen)

    print(f"\n── Top-10 Recommendations for user {sample_user} ──")
    title_lookup = {}
    if loader.movies_df is not None:
        title_lookup = dict(zip(loader.movies_df["movie_id"].astype(int), loader.movies_df["title"]))
    for rank, (mid, score) in enumerate(recs, 1):
        title = title_lookup.get(int(mid), f"Movie #{mid}")
        print(f"  {rank:2d}. {title:<45s}  score={score:.3f}")

    # Save model
    saved_path = model.save()
    print(f"\n✅  svd_model.py smoke-test complete. Model saved → {saved_path}")

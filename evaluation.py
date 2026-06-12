"""
evaluation.py
-------------
Unified evaluation pipeline for the Netflix Prize recommendation system.

Metrics implemented:
  - RMSE        (rating prediction accuracy)
  - MAE         (rating prediction accuracy)
  - MAP@K       (Mean Average Precision at K)
  - Precision@K (fraction of top-K recs that are relevant)
  - Recall@K    (fraction of relevant items captured in top-K)
  - NDCG@K      (Normalised Discounted Cumulative Gain at K)

Design:
  - All ranking metrics use: relevant = actual rating >= RELEVANCE_THRESHOLD (3.5)
  - Recommendations are generated from UNSEEN movies only
  - Reusable, model-agnostic functions (work with both Item-CF and SVD)
  - Evaluation results serialised to results/

Usage:
    from src.evaluation import Evaluator
    evaluator = Evaluator(k=10, relevance_threshold=3.5)
    results = evaluator.evaluate_model(model, test_df, train_df, model_name="SVD")
    evaluator.print_report(results)
"""

import time
import json
import logging
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
RELEVANCE_THRESHOLD = 3.5   # rating >= this → item is "relevant"
DEFAULT_K = 10


# ─────────────────────────────────────────────
# 1. Rating-Prediction Metrics
# ─────────────────────────────────────────────

def compute_rmse(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """
    Root Mean Squared Error between actual and predicted ratings.

    Parameters
    ----------
    actuals     : np.ndarray  ground-truth ratings
    predictions : np.ndarray  predicted ratings

    Returns
    -------
    float  RMSE
    """
    actuals = np.asarray(actuals, dtype=np.float32)
    predictions = np.asarray(predictions, dtype=np.float32)
    return float(np.sqrt(np.mean((actuals - predictions) ** 2)))


def compute_mae(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """
    Mean Absolute Error between actual and predicted ratings.

    Parameters
    ----------
    actuals     : np.ndarray  ground-truth ratings
    predictions : np.ndarray  predicted ratings

    Returns
    -------
    float  MAE
    """
    actuals = np.asarray(actuals, dtype=np.float32)
    predictions = np.asarray(predictions, dtype=np.float32)
    return float(np.mean(np.abs(actuals - predictions)))


# ─────────────────────────────────────────────
# 2. Per-User Ranking Metric Helpers
# ─────────────────────────────────────────────

def _average_precision_at_k(
    recommended: List,
    relevant: set,
    k: int,
) -> float:
    """
    Compute Average Precision@K for a single user.

    AP@K = (1 / min(|relevant|, K)) * Σ_i [P@i * rel(i)]

    where rel(i) = 1 if the item at rank i is relevant, else 0.

    Parameters
    ----------
    recommended : list   ordered list of recommended item ids (top-K)
    relevant    : set    set of relevant item ids for this user
    k           : int    cutoff rank

    Returns
    -------
    float  AP@K in [0, 1]
    """
    if not relevant:
        return 0.0

    recommended = recommended[:k]
    hits = 0
    precision_sum = 0.0

    for rank, item in enumerate(recommended, start=1):
        if item in relevant:
            hits += 1
            precision_sum += hits / rank

    # Normalise by min(|relevant|, K) — standard MAP definition
    normaliser = min(len(relevant), k)
    return precision_sum / normaliser if normaliser > 0 else 0.0


def _precision_at_k(recommended: List, relevant: set, k: int) -> float:
    """
    Precision@K = (# relevant in top-K) / K

    Parameters
    ----------
    recommended : list  ordered recommendations
    relevant    : set   relevant items for this user
    k           : int   cutoff rank

    Returns
    -------
    float  P@K in [0, 1]
    """
    if not relevant or k == 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def _recall_at_k(recommended: List, relevant: set, k: int) -> float:
    """
    Recall@K = (# relevant in top-K) / |relevant|

    Parameters
    ----------
    recommended : list  ordered recommendations
    relevant    : set   relevant items for this user
    k           : int   cutoff rank

    Returns
    -------
    float  R@K in [0, 1]
    """
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def _ndcg_at_k(recommended: List, relevant: set, k: int) -> float:
    """
    Normalised Discounted Cumulative Gain@K.

    DCG@K  = Σ_i rel(i) / log2(i + 1)
    IDCG@K = DCG of ideal (all relevant items first)
    NDCG@K = DCG@K / IDCG@K

    Parameters
    ----------
    recommended : list  ordered recommendations
    relevant    : set   relevant items for this user
    k           : int   cutoff rank

    Returns
    -------
    float  NDCG@K in [0, 1]
    """
    if not relevant:
        return 0.0

    top_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank, item in enumerate(top_k, start=1)
        if item in relevant
    )

    # Ideal DCG: relevant items occupy the top positions
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


# ─────────────────────────────────────────────
# 3. MAP@K  (reusable standalone function)
# ─────────────────────────────────────────────

def compute_map_at_k(
    recommendations: Dict,
    test_df: pd.DataFrame,
    k: int = DEFAULT_K,
    relevance_threshold: float = RELEVANCE_THRESHOLD,
    user_id_col: str = "user_id",
    item_id_col: str = "movie_id",
    rating_col: str = "rating",
) -> float:
    """
    Compute Mean Average Precision@K across all users in test_df.

    Relevant item definition: actual rating >= relevance_threshold.
    Only users with at least one relevant item contribute to the mean.

    Parameters
    ----------
    recommendations      : dict  { user_id: [(item_id, score), ...] }
                           recommendations must be pre-sorted descending by score
                           and should contain only UNSEEN items
    test_df              : pd.DataFrame  ground-truth ratings
    k                    : int    ranking cutoff
    relevance_threshold  : float  min rating to be considered relevant
    user_id_col          : str
    item_id_col          : str
    rating_col           : str

    Returns
    -------
    float  MAP@K in [0, 1]
    """
    # Build per-user relevant-item sets from test data
    relevant_items: Dict = (
        test_df[test_df[rating_col] >= relevance_threshold]
        .groupby(user_id_col)[item_id_col]
        .apply(set)
        .to_dict()
    )

    ap_scores = []
    for uid, recs in recommendations.items():
        relevant = relevant_items.get(uid, set())
        if not relevant:
            continue  # skip users with no relevant test items

        recommended_ids = [item_id for item_id, _ in recs]
        ap = _average_precision_at_k(recommended_ids, relevant, k)
        ap_scores.append(ap)

    if not ap_scores:
        logger.warning("MAP@%d: no users had relevant test items.", k)
        return 0.0

    return float(np.mean(ap_scores))


# ─────────────────────────────────────────────
# 4. Full Ranking Metrics Suite
# ─────────────────────────────────────────────

def compute_ranking_metrics(
    recommendations: Dict,
    test_df: pd.DataFrame,
    k: int = DEFAULT_K,
    relevance_threshold: float = RELEVANCE_THRESHOLD,
    user_id_col: str = "user_id",
    item_id_col: str = "movie_id",
    rating_col: str = "rating",
) -> Dict[str, float]:
    """
    Compute MAP@K, Precision@K, Recall@K, and NDCG@K in a single pass.

    Parameters
    ----------
    recommendations     : dict  { user_id: [(item_id, score), ...] }
    test_df             : pd.DataFrame
    k                   : int
    relevance_threshold : float

    Returns
    -------
    dict  { "MAP@K": float, "Precision@K": float,
            "Recall@K": float, "NDCG@K": float,
            "n_users_evaluated": int }
    """
    relevant_items: Dict = (
        test_df[test_df[rating_col] >= relevance_threshold]
        .groupby(user_id_col)[item_id_col]
        .apply(set)
        .to_dict()
    )

    ap_list, prec_list, rec_list, ndcg_list = [], [], [], []

    for uid, recs in tqdm(
        recommendations.items(),
        desc=f"Computing ranking metrics @{k}",
        unit=" users",
        mininterval=2.0,
    ):
        relevant = relevant_items.get(uid, set())
        if not relevant:
            continue

        recommended_ids = [item_id for item_id, _ in recs]

        ap_list.append(_average_precision_at_k(recommended_ids, relevant, k))
        prec_list.append(_precision_at_k(recommended_ids, relevant, k))
        rec_list.append(_recall_at_k(recommended_ids, relevant, k))
        ndcg_list.append(_ndcg_at_k(recommended_ids, relevant, k))

    n_evaluated = len(ap_list)
    if n_evaluated == 0:
        logger.warning("No users had relevant test items. All ranking metrics = 0.")
        return {f"MAP@{k}": 0.0, f"Precision@{k}": 0.0,
                f"Recall@{k}": 0.0, f"NDCG@{k}": 0.0,
                "n_users_evaluated": 0}

    return {
        f"MAP@{k}": round(float(np.mean(ap_list)), 6),
        f"Precision@{k}": round(float(np.mean(prec_list)), 6),
        f"Recall@{k}": round(float(np.mean(rec_list)), 6),
        f"NDCG@{k}": round(float(np.mean(ndcg_list)), 6),
        "n_users_evaluated": n_evaluated,
    }


# ─────────────────────────────────────────────
# 5. Recommendation Generator
#    (model-agnostic, handles Item-CF + SVD)
# ─────────────────────────────────────────────

def generate_recommendations(
    model,
    model_type: str,
    eval_users: List,
    train_df: pd.DataFrame,
    all_movie_ids: List,
    n: int = DEFAULT_K,
    idx2movie: Optional[Dict] = None,
    movie2idx: Optional[Dict] = None,
) -> Dict:
    """
    Generate Top-N recommendations from UNSEEN movies for a list of users.

    Supports:
        model_type = "item_cf"  →  uses ItemBasedCF.recommend(user_idx, ...)
        model_type = "svd"      →  uses SVDModel.recommend(user_id, ...)

    Parameters
    ----------
    model         : fitted ItemBasedCF or SVDModel
    model_type    : str  "item_cf" or "svd"
    eval_users    : list  user identifiers to evaluate
                          - item_cf: 0-based integer indices
                          - svd: raw user_ids
    train_df      : pd.DataFrame  used to build seen-movie masks
    all_movie_ids : list  all candidate movie ids
                          - item_cf: 0-based integer indices
                          - svd: raw movie_ids
    n             : int   top-N recommendations per user
    idx2movie     : dict  item_cf only — index → raw_movie_id (for output normalisation)
    movie2idx     : dict  item_cf only — raw_movie_id → index

    Returns
    -------
    dict  { raw_user_id: [(raw_movie_id, score), ...] }
          Keys are always raw user IDs for consistent evaluation.
    """
    model_type = model_type.lower().strip()
    recommendations: Dict = {}

    # ── Item-CF path ─────────────────────────────────────────────────
    if model_type == "item_cf":
        if idx2movie is None or movie2idx is None:
            raise ValueError("idx2movie and movie2idx are required for item_cf evaluation.")

        # Build seen-movie sets keyed by user_idx
        seen_by_user_idx: Dict[int, set] = {}
        for uid_idx in eval_users:
            raw_uid = idx2movie.get(uid_idx, uid_idx)    # may be user idx not movie
            seen_by_user_idx[uid_idx] = set()

        # Safer: pull seen movies from train_df using raw ids
        user_idx_to_raw = {}
        # We need the user index map — derive from train_df if idx2movie covers users
        # Fall back: use model's train_matrix row directly (done inside recommend())

        for uid_idx in tqdm(eval_users, desc="Item-CF recs", unit=" users"):
            recs_idx = model.recommend(
                user_idx=uid_idx,
                n=n,
                exclude_seen=True,
            )
            # Convert movie indices → raw movie ids
            recs_raw = [
                (int(idx2movie.get(midx, midx)), float(score))
                for midx, score in recs_idx
            ]
            recommendations[uid_idx] = recs_raw

    # ── SVD path ─────────────────────────────────────────────────────
    elif model_type == "svd":
        # Build seen-movie lookup once
        seen_lookup: Dict = (
            train_df.groupby("user_id")["movie_id"]
            .apply(set)
            .to_dict()
        )

        for uid in tqdm(eval_users, desc="SVD recs", unit=" users"):
            seen = seen_lookup.get(uid, set())
            recs = model.recommend(
                user_id=uid,
                n=n,
                all_movie_ids=all_movie_ids,
                seen_movie_ids=seen,
            )
            recommendations[uid] = recs

    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'item_cf' or 'svd'.")

    logger.info(
        "Generated %d-item recommendations for %d users.",
        n, len(recommendations),
    )
    return recommendations


# ─────────────────────────────────────────────
# 6. Main Evaluator Class
# ─────────────────────────────────────────────

class Evaluator:
    """
    Unified evaluation class for both Item-CF and SVD models.

    Parameters
    ----------
    k                   : int    ranking cutoff (default 10)
    relevance_threshold : float  min rating to be relevant (default 3.5)
    n_eval_users        : int    max users to evaluate ranking metrics
                                 (subsample for speed; use None for all)
    random_state        : int
    """

    def __init__(
        self,
        k: int = DEFAULT_K,
        relevance_threshold: float = RELEVANCE_THRESHOLD,
        n_eval_users: int = 1000,
        random_state: int = RANDOM_SEED,
    ):
        self.k = k
        self.relevance_threshold = relevance_threshold
        self.n_eval_users = n_eval_users
        self.random_state = random_state
        self._results_log: List[Dict] = []

    # ------------------------------------------------------------------
    # Main evaluation entry point
    # ------------------------------------------------------------------

    def evaluate_model(
        self,
        model,
        model_type: str,
        test_df: pd.DataFrame,
        train_df: pd.DataFrame,
        model_name: str = "Model",
        all_movie_ids: Optional[List] = None,
        idx2movie: Optional[Dict] = None,
        movie2idx: Optional[Dict] = None,
        user2idx: Optional[Dict] = None,
    ) -> Dict:
        """
        Run the full evaluation suite on a fitted model.

        Steps:
          1. Batch-predict ratings on test_df  → RMSE, MAE
          2. Sample eval users with relevant test items
          3. Generate Top-K recommendations (unseen movies only)
          4. Compute MAP@K, Precision@K, Recall@K, NDCG@K

        Parameters
        ----------
        model        : fitted ItemBasedCF or SVDModel
        model_type   : str  "item_cf" or "svd"
        test_df      : pd.DataFrame  [user_id, movie_id, rating]
        train_df     : pd.DataFrame  [user_id, movie_id, rating]
        model_name   : str  display name for logging
        all_movie_ids: list | None  candidate movies for recs
        idx2movie    : dict | None  required for item_cf
        movie2idx    : dict | None  required for item_cf
        user2idx     : dict | None  required for item_cf

        Returns
        -------
        dict  full results including all metrics and timing
        """
        results: Dict = {"model": model_name, "k": self.k}
        logger.info("── Evaluating %s ──", model_name)

        # ── Step 1: RMSE / MAE ───────────────────────────────────────
        logger.info("Step 1/3: Computing RMSE & MAE on %d test pairs …", len(test_df))
        t_rmse_start = time.time()

        if model_type == "item_cf":
            # Map raw ids → indices for item-CF
            uid_col = test_df["user_id"].map(user2idx).dropna().astype(int)
            mid_col = test_df["movie_id"].map(movie2idx).dropna().astype(int)
            # Keep only rows where both mappings exist (known users & movies)
            valid_mask = (
                test_df["user_id"].isin(user2idx) &
                test_df["movie_id"].isin(movie2idx)
            )
            test_known = test_df[valid_mask].copy()
            u_indices = test_known["user_id"].map(user2idx).values.astype(int)
            m_indices = test_known["movie_id"].map(movie2idx).values.astype(int)
            preds = model.predict_batch(u_indices, m_indices)
            actuals = test_known["rating"].values.astype(np.float32)

        else:  # svd
            preds = model.predict_batch(
                test_df["user_id"].values,
                test_df["movie_id"].values,
            )
            actuals = test_df["rating"].values.astype(np.float32)

        results["RMSE"] = round(compute_rmse(actuals, preds), 6)
        results["MAE"] = round(compute_mae(actuals, preds), 6)
        results["rmse_eval_time_sec"] = round(time.time() - t_rmse_start, 2)
        logger.info("  RMSE=%.4f | MAE=%.4f", results["RMSE"], results["MAE"])

        # ── Step 2: Sample evaluation users ──────────────────────────
        logger.info("Step 2/3: Selecting evaluation users …")

        # Only consider users who have relevant items in test set
        users_with_relevant = (
            test_df[test_df["rating"] >= self.relevance_threshold]["user_id"]
            .unique()
            .tolist()
        )

        if model_type == "item_cf":
            # Further restrict to users known in train (have an index)
            users_with_relevant = [
                u for u in users_with_relevant if u in user2idx
            ]

        rng = np.random.RandomState(self.random_state)
        if self.n_eval_users and len(users_with_relevant) > self.n_eval_users:
            eval_raw_users = rng.choice(
                users_with_relevant, size=self.n_eval_users, replace=False
            ).tolist()
        else:
            eval_raw_users = users_with_relevant

        logger.info("  Evaluating ranking metrics on %d users.", len(eval_raw_users))

        # For item_cf, convert raw user ids → indices
        if model_type == "item_cf":
            eval_users_for_model = [user2idx[u] for u in eval_raw_users]
        else:
            eval_users_for_model = eval_raw_users

        # Resolve all_movie_ids
        if all_movie_ids is None:
            if model_type == "item_cf":
                all_movie_ids = list(range(model.n_movies))
            else:
                all_movie_ids = train_df["movie_id"].unique().tolist()

        # ── Step 3: Generate recommendations & compute ranking metrics
        logger.info("Step 3/3: Generating recommendations & computing ranking metrics …")
        t_rank_start = time.time()

        recommendations_raw = generate_recommendations(
            model=model,
            model_type=model_type,
            eval_users=eval_users_for_model,
            train_df=train_df,
            all_movie_ids=all_movie_ids,
            n=self.k,
            idx2movie=idx2movie,
            movie2idx=movie2idx,
        )

        # Normalise keys: item_cf returns user_idx keys → remap to raw user ids
        if model_type == "item_cf":
            idx2user = {v: k for k, v in user2idx.items()}
            recommendations_normalised = {
                idx2user[uid_idx]: recs
                for uid_idx, recs in recommendations_raw.items()
                if uid_idx in idx2user
            }
        else:
            recommendations_normalised = recommendations_raw

        # Filter test_df to eval users only
        test_eval = test_df[test_df["user_id"].isin(eval_raw_users)]

        ranking_metrics = compute_ranking_metrics(
            recommendations=recommendations_normalised,
            test_df=test_eval,
            k=self.k,
            relevance_threshold=self.relevance_threshold,
        )

        results.update(ranking_metrics)
        results["ranking_eval_time_sec"] = round(time.time() - t_rank_start, 2)

        # Store recommendations for later use (e.g. recommendation.py)
        results["_recommendations"] = recommendations_normalised

        # Log
        logger.info(
            "  MAP@%d=%.4f | P@%d=%.4f | R@%d=%.4f | NDCG@%d=%.4f",
            self.k, results.get(f"MAP@{self.k}", 0),
            self.k, results.get(f"Precision@{self.k}", 0),
            self.k, results.get(f"Recall@{self.k}", 0),
            self.k, results.get(f"NDCG@{self.k}", 0),
        )

        self._results_log.append({k: v for k, v in results.items() if not k.startswith("_")})
        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, results: Dict) -> None:
        """Pretty-print a single model's evaluation results."""
        line = "─" * 50
        print(f"\n{line}")
        print(f"  Evaluation Report: {results.get('model', 'Model')}")
        print(line)
        metrics_order = [
            "RMSE", "MAE",
            f"MAP@{results.get('k', self.k)}",
            f"Precision@{results.get('k', self.k)}",
            f"Recall@{results.get('k', self.k)}",
            f"NDCG@{results.get('k', self.k)}",
            "n_users_evaluated",
            "rmse_eval_time_sec",
            "ranking_eval_time_sec",
        ]
        for key in metrics_order:
            val = results.get(key)
            if val is not None:
                print(f"  {key:<28s}: {val}")
        print(line)

    def comparison_table(self) -> pd.DataFrame:
        """
        Return a DataFrame comparing all evaluated models side by side.

        Returns
        -------
        pd.DataFrame  rows = models, columns = metrics
        """
        if not self._results_log:
            logger.warning("No evaluation results logged yet.")
            return pd.DataFrame()

        df = pd.DataFrame(self._results_log)
        k = self.k
        metric_cols = [
            "model", "RMSE", "MAE",
            f"MAP@{k}", f"Precision@{k}", f"Recall@{k}", f"NDCG@{k}",
            "n_users_evaluated", "rmse_eval_time_sec", "ranking_eval_time_sec",
        ]
        available = [c for c in metric_cols if c in df.columns]
        return df[available].reset_index(drop=True)

    def save_results(self, path: Optional[Path] = None) -> Path:
        """
        Save all logged evaluation results to a JSON file.

        Returns
        -------
        Path  where results were saved
        """
        if path is None:
            path = RESULTS_DIR / "evaluation_results.json"
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self._results_log, f, indent=2, default=str)
        logger.info("Evaluation results saved → %s", path)
        return path


# ─────────────────────────────────────────────
# 7. Convenience Standalone Functions
# ─────────────────────────────────────────────

def evaluate_rmse_only(
    model,
    model_type: str,
    test_df: pd.DataFrame,
    user2idx: Optional[Dict] = None,
    movie2idx: Optional[Dict] = None,
) -> Tuple[float, float]:
    """
    Quick RMSE + MAE evaluation without generating recommendations.
    Useful for fast validation during training.

    Returns
    -------
    (rmse, mae)
    """
    if model_type == "item_cf":
        valid_mask = (
            test_df["user_id"].isin(user2idx) &
            test_df["movie_id"].isin(movie2idx)
        )
        test_known = test_df[valid_mask]
        u_idx = test_known["user_id"].map(user2idx).values.astype(int)
        m_idx = test_known["movie_id"].map(movie2idx).values.astype(int)
        preds = model.predict_batch(u_idx, m_idx)
        actuals = test_known["rating"].values.astype(np.float32)
    else:
        preds = model.predict_batch(
            test_df["user_id"].values,
            test_df["movie_id"].values,
            show_progress=False,
        )
        actuals = test_df["rating"].values.astype(np.float32)

    return compute_rmse(actuals, preds), compute_mae(actuals, preds)


def per_user_metrics(
    recommendations: Dict,
    test_df: pd.DataFrame,
    k: int = DEFAULT_K,
    relevance_threshold: float = RELEVANCE_THRESHOLD,
) -> pd.DataFrame:
    """
    Compute per-user AP@K, P@K, R@K, NDCG@K.
    Useful for analysis of metric distributions.

    Returns
    -------
    pd.DataFrame  columns: [user_id, AP@K, Precision@K, Recall@K, NDCG@K,
                             n_relevant, n_recommended]
    """
    relevant_items: Dict = (
        test_df[test_df["rating"] >= relevance_threshold]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )

    rows = []
    for uid, recs in recommendations.items():
        relevant = relevant_items.get(uid, set())
        rec_ids = [item_id for item_id, _ in recs]
        rows.append({
            "user_id": uid,
            f"AP@{k}": _average_precision_at_k(rec_ids, relevant, k),
            f"Precision@{k}": _precision_at_k(rec_ids, relevant, k),
            f"Recall@{k}": _recall_at_k(rec_ids, relevant, k),
            f"NDCG@{k}": _ndcg_at_k(rec_ids, relevant, k),
            "n_relevant": len(relevant),
            "n_recommended": len(rec_ids),
        })

    return pd.DataFrame(rows)


def rating_bias_analysis(
    actuals: np.ndarray,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """
    Break down RMSE by actual rating bucket (1, 2, 3, 4, 5).
    Helps diagnose whether the model struggles with specific rating levels.

    Returns
    -------
    pd.DataFrame  columns: [actual_rating, count, RMSE, MAE, mean_pred, bias]
    """
    actuals = np.asarray(actuals, dtype=np.float32)
    predictions = np.asarray(predictions, dtype=np.float32)

    rows = []
    for rating in sorted(np.unique(actuals.astype(int))):
        mask = actuals.astype(int) == rating
        if mask.sum() == 0:
            continue
        a = actuals[mask]
        p = predictions[mask]
        rows.append({
            "actual_rating": rating,
            "count": int(mask.sum()),
            "RMSE": round(float(np.sqrt(np.mean((a - p) ** 2))), 4),
            "MAE": round(float(np.mean(np.abs(a - p))), 4),
            "mean_pred": round(float(p.mean()), 4),
            "bias": round(float((p - a).mean()), 4),  # positive = over-predicting
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 8. Main — smoke-test
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
    from src.collaborative_filtering import ItemBasedCF
    from src.svd_model import SVDModel

    # ── Load data ────────────────────────────────────────────────────
    loader = NetflixDataLoader(use_subset=True)
    loader.load(cache=True)
    loader.load_movies()
    train_df, val_df, test_df = load_splits()

    train_matrix = build_interaction_matrix(train_df, loader.user2idx, loader.movie2idx)

    # ── Fit models ───────────────────────────────────────────────────
    print("\nFitting Item-CF …")
    cf_model = ItemBasedCF(top_k_similar=50, use_adjusted=True)
    cf_model.fit(train_matrix, cache_path=PROCESSED_DIR / "item_cf_similarity.pkl")

    print("\nFitting SVD …")
    svd_model = SVDModel(params={"n_factors": 100, "n_epochs": 20})
    svd_model.fit(train_df)

    # ── Evaluate ─────────────────────────────────────────────────────
    evaluator = Evaluator(k=10, relevance_threshold=3.5, n_eval_users=500)

    print("\nEvaluating Item-CF …")
    cf_results = evaluator.evaluate_model(
        model=cf_model,
        model_type="item_cf",
        test_df=test_df,
        train_df=train_df,
        model_name="Item-CF (Adjusted Cosine)",
        idx2movie=loader.idx2movie,
        movie2idx=loader.movie2idx,
        user2idx=loader.user2idx,
    )
    evaluator.print_report(cf_results)

    print("\nEvaluating SVD …")
    svd_results = evaluator.evaluate_model(
        model=svd_model,
        model_type="svd",
        test_df=test_df,
        train_df=train_df,
        model_name="SVD (Surprise)",
        all_movie_ids=train_df["movie_id"].unique().tolist(),
    )
    evaluator.print_report(svd_results)

    # ── Comparison table ─────────────────────────────────────────────
    print("\n── Model Comparison ──")
    comparison = evaluator.comparison_table()
    print(comparison.to_string(index=False))

    # ── Save results ─────────────────────────────────────────────────
    evaluator.save_results()

    # ── Rating bias analysis (SVD) ───────────────────────────────────
    preds = svd_model.predict_batch(test_df["user_id"].values, test_df["movie_id"].values)
    bias_df = rating_bias_analysis(test_df["rating"].values, preds)
    print("\n── SVD Rating Bias Analysis ──")
    print(bias_df.to_string(index=False))

    print("\n✅  evaluation.py smoke-test complete.")

import os
import math
import pickle
import math
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve
)

from preprocessing_1 import Preprocessor, load_and_slice

from river import anomaly, drift, linear_model, optim
from river import preprocessing as river_prep


# ── Paths ───────────────────────────────────────────────────────────────────

DATA_PATH     = r"D:\FraudShield\data\raw\train_transaction.csv"
ARTIFACTS_DIR = r"D:\FraudShield\artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def arr_to_dicts(X: np.ndarray, feature_cols: list) -> list[dict]:
    """Convert numpy matrix rows → list of River-compatible feature dicts."""
    return [
        {feature_cols[j]: float(X[i, j]) for j in range(len(feature_cols))}
        for i in range(len(X))
    ]


def hst_squash(x: float, scale: float = 10.0) -> float:
    """
    Sigmoid squash of raw HalfSpaceTrees score.
    HST output is heavily right-skewed near 0. Squashing centres it
    around 0.5 so the meta-learner's StandardScaler gets clean gradients.
    """
    return 1.0 / (1.0 + math.exp(-scale * (x - 0.5)))


def print_metrics(label: str, y_true: np.ndarray, y_score: np.ndarray):
    auc_roc = roc_auc_score(y_true, y_score)
    auc_pr  = average_precision_score(y_true, y_score)
    print(f"  {label:35s}  AUC-ROC: {auc_roc:.4f}  |  AUC-PR: {auc_pr:.4f}")
    return auc_roc, auc_pr


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD + SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    df = load_and_slice(DATA_PATH)

    train_df, eval_df = train_test_split(
        df,
        test_size = 0.22,
        stratify  = df['isFraud'],
        random_state = 42,
    )
    print(f"  Train : {len(train_df):,}  |  fraud rate: {train_df['isFraud'].mean():.3%}")
    print(f"  Eval  : {len(eval_df):,}   |  fraud rate: {eval_df['isFraud'].mean():.3%}")
    return train_df, eval_df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def run_preprocessing(train_df: pd.DataFrame, eval_df: pd.DataFrame):
    prep = Preprocessor(sparse_threshold=0.70, clip=5.0)

    # X        — unscaled → LightGBM, SHAP, HST
    # X_scaled — ±5σ clipped StandardScaled → AE slot (reserved)
    X_train, _ = prep.fit_transform(train_df)
    X_eval,  _ = prep.transform(eval_df)

    y_train = train_df['isFraud'].values
    y_eval  = eval_df['isFraud'].values

    prep.summary()
    print(f"\n  X_train : {X_train.shape}  |  NaN: {np.isnan(X_train).sum()}")
    print(f"  X_eval  : {X_eval.shape}   |  NaN: {np.isnan(X_eval).sum()}")

    prep.save(os.path.join(ARTIFACTS_DIR, "preprocessor.pkl"))
    print("  ✅ Preprocessor saved")
    return prep, X_train, X_eval, y_train, y_eval


# ══════════════════════════════════════════════════════════════════════════════
# 3.  LIGHTGBM — TWO-PHASE TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_lgbm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_eval:  np.ndarray, y_eval:  np.ndarray,
    feature_cols: list,
):
    """
    Two-phase LightGBM training (coarse → refinement)
    """

    # ── Imbalance handling ─────────────────────────────
    fraud_count = int(y_train.sum())
    legit_count = len(y_train) - fraud_count
    spw         = legit_count / fraud_count

    # ── Base params ────────────────────────────────────
    base_params = dict(
        objective         = 'binary',
        metric            = ['auc', 'average_precision'],
        num_leaves        = 127,
        min_child_samples = 30,
        feature_fraction  = 0.8,
        bagging_fraction  = 0.8,
        bagging_freq      = 1,
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        scale_pos_weight  = spw,
        n_jobs            = -1,
        verbose           = -1,
        random_state      = 42,
    )

    # 🔥 FIX: prevent LightGBM from freeing raw data
    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=feature_cols,
        free_raw_data=False
    )

    deval = lgb.Dataset(
        X_eval,
        label=y_eval,
        reference=dtrain,
        free_raw_data=False
    )

    # ── Phase 1 ───────────────────────────────────────
    print("  Phase 1 — coarse pass  (lr=0.05, max 500 rounds)")

    model_p1 = lgb.train(
        {**base_params, "learning_rate": 0.05},
        dtrain,
        num_boost_round=500,
        valid_sets=[deval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    best_p1 = model_p1.best_iteration or 500
    print(f"  Phase 1 complete — best iteration: {best_p1}")

    p1_eval_scores = model_p1.predict(X_eval, num_iteration=best_p1)
    print_metrics("Phase 1 (eval)", y_eval, p1_eval_scores)

    # ── Phase 2 ───────────────────────────────────────
    print("\n  Phase 2 — refinement  (lr=0.01, max 1500 more rounds)")

    model = lgb.train(
        {**base_params, "learning_rate": 0.01},
        dtrain,
        num_boost_round=1500,
        valid_sets=[deval],
        init_model=model_p1,  # continue training
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )

    best_total = model.best_iteration or (best_p1 + 1500)
    print(f"  Phase 2 complete — total iterations: {best_total}")

    # ── Predictions (IMPORTANT: use best iteration) ────
    lgb_train_scores = model.predict(X_train, num_iteration=best_total)
    lgb_eval_scores  = model.predict(X_eval,  num_iteration=best_total)

    print_metrics("Final LightGBM (eval)", y_eval, lgb_eval_scores)

    # ── Save model ─────────────────────────────────────
    model.save_model(os.path.join(ARTIFACTS_DIR, "lgbm.txt"))
    print("  ✅ LightGBM saved")

    # ── Feature importance ─────────────────────────────
    importance = pd.DataFrame({
        "feature": feature_cols,
        "gain": model.feature_importance(importance_type="gain"),
        "split_count": model.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False).reset_index(drop=True)

    imp_path = os.path.join(ARTIFACTS_DIR, "lgbm_feature_importance.csv")
    importance.to_csv(imp_path, index=False)

    print("\n  Top-10 features by gain importance:")
    for _, row in importance.head(10).iterrows():
        print(f"    {row['feature']:35s}  gain={row['gain']:>12.1f}"
              f"  splits={row['split_count']:>6}")

    return model, lgb_train_scores, lgb_eval_scores


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SHAP
# ══════════════════════════════════════════════════════════════════════════════

def compute_and_save_shap(
    model,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    feature_cols: list,
    n_samples: int = 3000,
):
    """
    Exact SHAP values via TreeExplainer on a stratified eval sample.
    Stratified to guarantee fraud rows are represented (≥20% fraud).

    Saves:
      shap_values.npy       — (n_samples, n_features)  float32
      shap_X_sample.npy     — matching raw feature matrix
      shap_y_sample.npy     — matching labels (0/1)
      shap_base_value.npy   — model base value for waterfall / force plots
      shap_feature_cols.pkl — ordered feature name list
    """
    print(f"  Computing SHAP on {n_samples} stratified eval samples...")

    fraud_idx = np.where(y_eval == 1)[0]
    legit_idx = np.where(y_eval == 0)[0]
    n_fraud   = min(len(fraud_idx), n_samples // 5)
    n_legit   = n_samples - n_fraud

    rng           = np.random.default_rng(42)
    sampled_idx   = np.concatenate([
        rng.choice(fraud_idx, n_fraud, replace=False),
        rng.choice(legit_idx, n_legit, replace=False),
    ])

    X_sample = X_eval[sampled_idx]
    y_sample = y_eval[sampled_idx]

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)

    # LightGBM binary returns [neg_class_array, pos_class_array] — take fraud
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    shap_vals = shap_vals.astype(np.float32)

    base_value = (
        explainer.expected_value
        if isinstance(explainer.expected_value, float)
        else float(explainer.expected_value[1])
    )

    # ── Save ──────────────────────────────────────────────────────────────
    np.save(os.path.join(ARTIFACTS_DIR, "shap_values.npy"),    shap_vals)
    np.save(os.path.join(ARTIFACTS_DIR, "shap_X_sample.npy"),  X_sample.astype(np.float32))
    np.save(os.path.join(ARTIFACTS_DIR, "shap_y_sample.npy"),  y_sample)
    np.save(os.path.join(ARTIFACTS_DIR, "shap_base_value.npy"),
            np.array([base_value], dtype=np.float32))
    joblib.dump(feature_cols, os.path.join(ARTIFACTS_DIR, "shap_feature_cols.pkl"))

    # ── Global top-10 ─────────────────────────────────────────────────────
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx  = mean_abs.argsort()[::-1][:10]
    print("  Top-10 SHAP features (global mean |SHAP|):")
    for rank, i in enumerate(top_idx, 1):
        print(f"    {rank:2d}. {feature_cols[i]:35s}  {mean_abs[i]:.5f}")

    # ── Fraud vs legit mean SHAP per feature (useful for fingerprinter) ───
    fraud_mask = y_sample == 1
    fraud_mean_shap = shap_vals[fraud_mask].mean(axis=0)
    legit_mean_shap = shap_vals[~fraud_mask].mean(axis=0)
    shap_contrast   = pd.DataFrame({
        "feature"         : feature_cols,
        "fraud_mean_shap" : fraud_mean_shap,
        "legit_mean_shap" : legit_mean_shap,
        "contrast"        : fraud_mean_shap - legit_mean_shap,
    }).sort_values("contrast", ascending=False)

    shap_contrast.to_csv(
        os.path.join(ARTIFACTS_DIR, "shap_fraud_vs_legit_contrast.csv"),
        index=False
    )

    print("  ✅ SHAP artifacts saved")
    return shap_vals, X_sample, base_value


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ONLINE LAYER
# ══════════════════════════════════════════════════════════════════════════════



def safe_float(x):
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return 0.0
        return x
    except:
        return 0.0

class OnlineLayer:
    """
    Three-component real-time intelligence layer.

    HalfSpaceTrees  — online unsupervised anomaly detection.
                      Updates on every production transaction.
                      Catches fraud patterns LightGBM hasn't seen yet.

    ADWIN           — one drift detector per signal stream (lgb + hst).
                      Fires retrain_recommended when distribution shifts.

    Meta-LR         — adaptive score fusion.
                      Learns to blend lgb + hst from analyst feedback.
                      The moat: gets smarter with every confirmed case.

    Note on River Pipeline: deliberately NOT using river.compose.Pipeline.
    Pipeline.learn_one internally creates a VectorDict that causes a type
    conflict with LogisticRegression._eval_gradient_one in several River
    versions. Manual two-step (scaler → lr) is explicit and version-safe.
    """

    def __init__(self):
        self.hst = anomaly.HalfSpaceTrees(
            n_trees     = 50,
            height      = 8,
            window_size = 512,
            seed        = 42,
        )

        self.adwin_lgb = drift.ADWIN()
        self.adwin_hst = drift.ADWIN()

        # Manual scaler + LR — avoids River Pipeline VectorDict bug [FIX-1]
        self.meta_scaler = river_prep.StandardScaler()
        self.meta_lr     = linear_model.LogisticRegression(
            optimizer = optim.SGD(lr=0.01)
        )

        self.drift_log   : list[dict] = []
        self._tx_count   : int        = 0
        self._meta_warmed: bool       = False

    # ── warm-up ─────────────────────────────────────────────────────────────

    def warmup_hst(self, X_legit_dicts: list[dict]):
        """[FIX-2] Teach HST the legit distribution before any scoring."""
        for x in X_legit_dicts:
            self.hst.learn_one(x)
        print(f"  HST warmed on {len(X_legit_dicts):,} legit transactions")

    def warmup_meta(
        self,
        lgb_train_scores: np.ndarray,
        X_train_dicts: list[dict],
        y_train: np.ndarray,
    ):  
        print(f"  Warming meta-learner on {len(X_train_dicts):,} training samples...")

        for i, x_dict in enumerate(X_train_dicts):
            raw_hst = self.hst.score_one(x_dict)
            hst_sq = hst_squash(raw_hst)

            # 🔥 SAFE VALUES
            lgb_val = safe_float(lgb_train_scores[i])
            hst_val = safe_float(hst_sq)

            # 🔥 BOUND VALUES (optional but powerful)
            lgb_val = max(0.0, min(1.0, lgb_val))
            hst_val = max(0.0, min(1.0, hst_val))

            features = {
                "lgb_score": lgb_val,
                "hst_score": hst_val,
            }

            try:
                self.meta_lr.learn_one(features, bool(y_train[i]))
            except Exception as e:
                print(f"\n💥 Crash at index {i}")
                print("Features:", features)
                print("Label:", y_train[i])
                raise e

            if (i + 1) % 50_000 == 0:
                print(f"    {i+1:,} / {len(X_train_dicts):,}")

        self._meta_warmed = True
        print("  Meta-learner warm-up complete")



    # ── score ────────────────────────────────────────────────────────────────

    def score(self, x_dict: dict, lgb_score: float, learn: bool = True) -> dict:
        self._tx_count += 1

        raw_hst = self.hst.score_one(x_dict)
        hst_sq = hst_squash(raw_hst)

        if learn:
            self.hst.learn_one(x_dict)

        lgb_drift = self.adwin_lgb.update(lgb_score)
        hst_drift = self.adwin_hst.update(hst_sq)

        if lgb_drift or hst_drift:
            self.drift_log.append({
                "tx_index": self._tx_count,
                "signal": "lgb" if lgb_drift else "hst",
                "lgb_score_at_drift": round(lgb_score, 4),
                "hst_score_at_drift": round(hst_sq, 4),
                "retrain_recommended": True,
            })

        lgb_val = safe_float(lgb_score)
        hst_val = safe_float(hst_sq)

        lgb_val = max(0.0, min(1.0, lgb_val))
        hst_val = max(0.0, min(1.0, hst_val))

        features = {
            "lgb_score": lgb_val,
            "hst_score": hst_val,
        }

        try:
            meta_score = self.meta_lr.predict_proba_one(features).get(True, 0.5)
        except Exception as e:
            print(f"  [WARN] meta predict failed at tx={self._tx_count}: {e}")
            meta_score = (lgb_score + hst_sq) / 2.0

        return {
            "lgb_score": round(lgb_score, 4),
            "hst_raw": round(raw_hst, 4),
            "hst_squashed": round(hst_sq, 4),
            "meta_score": round(meta_score, 4),
            "drift_alert": bool(lgb_drift or hst_drift),
        }

    # ── analyst feedback — the moat ──────────────────────────────────────────

    def analyst_feedback(
        self,
        lgb_score: float,
        hst_squashed: float,
        is_fraud: bool,
    ):  
        x = {
            "lgb_score": safe_float(lgb_score),
            "hst_score": safe_float(hst_squashed),
        }
        self.meta_lr.learn_one(x, is_fraud)

    # ── diagnostics ──────────────────────────────────────────────────────────

    def drift_summary(self):
        n = len(self.drift_log)
        print(f"  Drift events detected: {n}")
        if n:
            print("  Last 5 drift events:")
            for ev in self.drift_log[-5:]:
                print(f"    tx={ev['tx_index']:>8,}  signal={ev['signal']}"
                      f"  lgb={ev['lgb_score_at_drift']:.4f}"
                      f"  hst={ev['hst_score_at_drift']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  EVAL
# ══════════════════════════════════════════════════════════════════════════════

def run_hybrid_eval(
    online_layer    : OnlineLayer,
    X_eval_dicts    : list[dict],
    lgb_eval_scores : np.ndarray,
    y_eval          : np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    [FIX-4] Offline eval with learn=False — HST does not absorb eval fraud.
    Returns (meta_scores, hst_scores) arrays.
    """
    n            = len(X_eval_dicts)
    meta_scores  = np.zeros(n, dtype=np.float32)
    hst_scores   = np.zeros(n, dtype=np.float32)

    for i, (x_dict, lgb_score) in enumerate(zip(X_eval_dicts, lgb_eval_scores)):
        result        = online_layer.score(x_dict, float(lgb_score), learn=False)
        meta_scores[i] = result["meta_score"]
        hst_scores[i]  = result["hst_squashed"]

        if (i + 1) % 25_000 == 0:
            print(f"  scored {i+1:,} / {n:,}")

    return meta_scores, hst_scores


def full_eval_report(
    y_eval      : np.ndarray,
    lgb_scores  : np.ndarray,
    hst_scores  : np.ndarray,
    meta_scores : np.ndarray,
):
    print("\n" + "=" * 65)
    print("  EVALUATION REPORT")
    print("=" * 65)
    print_metrics("LightGBM (primary)",      y_eval, lgb_scores)
    print_metrics("HalfSpaceTrees (online)", y_eval, hst_scores)
    print_metrics("Meta-Learner (hybrid)",   y_eval, meta_scores)
    print("-" * 65)

    # Precision @ fixed recall — the metric that matters for fraud ops
    prec, rec, thresh = precision_recall_curve(y_eval, meta_scores)
    print("  Meta-Learner precision at operational recall targets:")
    for target in [0.80, 0.90, 0.95]:
        idx = np.argmin(np.abs(rec - target))
        t   = float(thresh[idx]) if idx < len(thresh) else float("nan")
        print(f"    Recall={target:.0%}  →  "
              f"Precision={prec[idx]:.3f}  Threshold={t:.3f}")
    print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "=" * 65)
    print("  FRAUDSHEID — Hybrid Training Pipeline  v3  (FINAL)")
    print("=" * 65)

    # ── 1. Data ──────────────────────────────────────────────────────────────
    print("\n── 1. LOAD DATA ──────────────────────────────────────────────")
    train_df, eval_df = load_data()

    # ── 2. Preprocessing ─────────────────────────────────────────────────────
    print("\n── 2. PREPROCESSING ──────────────────────────────────────────")
    prep, X_train, X_eval, y_train, y_eval = run_preprocessing(train_df, eval_df)

    # ── 3. LightGBM ──────────────────────────────────────────────────────────
    print("\n── 3. LIGHTGBM (two-phase) ───────────────────────────────────")
    lgbm_model, lgb_train_scores, lgb_eval_scores = train_lgbm(
        X_train, y_train, X_eval, y_eval, prep.feature_cols
    )

    # ── 4. SHAP ──────────────────────────────────────────────────────────────
    print("\n── 4. SHAP ───────────────────────────────────────────────────")
    shap_vals, shap_X, base_value = compute_and_save_shap(
        lgbm_model, X_eval, y_eval, prep.feature_cols, n_samples=3000
    )

    # ── 5. Online Layer ───────────────────────────────────────────────────────
    print("\n── 5. ONLINE LAYER ───────────────────────────────────────────")
    online_layer  = OnlineLayer()
    X_train_dicts = arr_to_dicts(X_train, prep.feature_cols)
    X_eval_dicts  = arr_to_dicts(X_eval,  prep.feature_cols)

    # HST warm-up — legit only, 50k samples [FIX-2]
    legit_idx   = np.where(y_train == 0)[0]
    n_warmup    = min(50_000, len(legit_idx))
    legit_dicts = [X_train_dicts[i] for i in legit_idx[:n_warmup]]
    print(f"  Warming HST on {n_warmup:,} legit-only transactions...")
    online_layer.warmup_hst(legit_dicts)

    # Meta warm-up — full train with known labels [FIX-3]
    online_layer.warmup_meta(lgb_train_scores, X_train_dicts, y_train)

    # ── 6. Eval ───────────────────────────────────────────────────────────────
    print("\n── 6. HYBRID EVAL (learn=False) ──────────────────────────────")
    meta_scores, hst_eval_scores = run_hybrid_eval(
        online_layer, X_eval_dicts, lgb_eval_scores, y_eval
    )
    full_eval_report(y_eval, lgb_eval_scores, hst_eval_scores, meta_scores)
    online_layer.drift_summary()

    # ── 7. Save all artifacts ─────────────────────────────────────────────────
    print("\n── 7. SAVING ─────────────────────────────────────────────────")

    # Online layer (pickle — River objects are pickle-safe)
    online_path = os.path.join(ARTIFACTS_DIR, "online_layer.pkl")
    with open(online_path, "wb") as f:
        pickle.dump(online_layer, f)
    print(f"  ✅ Online layer      → {online_path}")

    # Eval score arrays — consumed by Pattern Fingerprinter (Layer 4)
    np.save(os.path.join(ARTIFACTS_DIR, "eval_lgb_scores.npy"),
            lgb_eval_scores.astype(np.float32))
    np.save(os.path.join(ARTIFACTS_DIR, "eval_hst_scores.npy"),  hst_eval_scores)
    np.save(os.path.join(ARTIFACTS_DIR, "eval_meta_scores.npy"), meta_scores)
    np.save(os.path.join(ARTIFACTS_DIR, "eval_labels.npy"),      y_eval.astype(np.int8))
    print("  ✅ Eval score arrays → eval_*.npy")

    # ── Artifact manifest ─────────────────────────────────────────────────────
    print("\n── ARTIFACT MANIFEST ─────────────────────────────────────────")
    for fname in sorted(os.listdir(ARTIFACTS_DIR)):
        fpath = os.path.join(ARTIFACTS_DIR, fname)
        size  = os.path.getsize(fpath) / 1024
        print(f"  {fname:45s}  {size:8.1f} KB")

    print("\n✅ Fraudsheid training pipeline v3 complete.")
    print(f"   Artifacts: {ARTIFACTS_DIR}")
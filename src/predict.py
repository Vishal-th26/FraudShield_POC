import os
import pandas as pd
import joblib
import lightgbm as lgb
import pickle
from HybridModel_v1 import OnlineLayer
from preprocessing_1 import Preprocessor, load_and_slice
from HybridModel_v1 import arr_to_dicts

# ── PATHS ─────────────────────────────────────────────
ARTIFACTS_DIR = "artifacts"

PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
LGBM_PATH         = os.path.join(ARTIFACTS_DIR, "lgbm.txt")
ONLINE_PATH       = os.path.join(ARTIFACTS_DIR, "online_layer.pkl")


# ── LOAD EVERYTHING ONCE (IMPORTANT) ──────────────────

print("🚀 Initializing FraudShield Engine...")

preprocessor = Preprocessor.load(PREPROCESSOR_PATH)
lgbm_model   = lgb.Booster(model_file=LGBM_PATH)

with open(ONLINE_PATH, "rb") as f:
    online_layer = pickle.load(f)

print("✅ All artifacts loaded successfully.\n")


# ── MAIN PIPELINE ─────────────────────────────────────

def run_pipeline(file_path: str):

    # 1. Load data
    df = load_and_slice(file_path)

    # 2. Preprocess
    X, _ = preprocessor.transform(df)

    # 3. Convert for River
    X_dicts = arr_to_dicts(X, preprocessor.feature_cols)

    # 4. LightGBM scores
    lgb_scores = lgbm_model.predict(X)

    results = []

    # 5. Hybrid scoring
    for i, (x_dict, lgb_score) in enumerate(zip(X_dicts, lgb_scores)):

        out = online_layer.score(
            x_dict,
            float(lgb_score),
            learn=False  # IMPORTANT for demo consistency
        )

        results.append({
            "transaction_id": int(df.iloc[i]["TransactionID"]) if "TransactionID" in df.columns else i,
            "lgb_score": out["lgb_score"],
            "hst_score": out["hst_squashed"],
            "final_score": out["meta_score"],
            "risk_label": "🔴 High Risk" if out["meta_score"] > 0.7 else "🟢 Low Risk",
            "drift_alert": out["drift_alert"]
        })

    return pd.DataFrame(results)


# ── CLI RUN ───────────────────────────────────────────

if __name__ == "__main__":

    INPUT_FILE = r"D:\FraudShield\data\raw\train_transaction.csv"

    output = run_pipeline(INPUT_FILE)

    print("\n=== FRAUD DETECTION RESULTS ===")
    print(output.head())

    output.to_csv("predictions.csv", index=False)

    print("\n✅ Predictions saved to predictions.csv")
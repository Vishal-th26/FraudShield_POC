import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
 
# ── Column definitions ─────────────────────────────────────────────────────
 
KEEP_COLS = (
    ['TransactionID', 'isFraud', 'card1']
    + ['TransactionAmt', 'ProductCD']
    + [f'card{i}' for i in range(2, 7)]
    + ['addr1', 'addr2']
    + [f'C{i}' for i in range(1, 15)]
    + [f'D{i}' for i in range(1, 16)]
    + [f'V{i}' for i in range(1, 51)]
)
 
CATEGORICAL_COLS = ['ProductCD', 'card4', 'card6']          # low-cardinality strings
ID_COLS          = ['TransactionID', 'isFraud', 'card1']    # never fed to model
 
SPARSE_THRESHOLD = 0.70          # drop V cols with >70% NaN
CLIP_SIGMA       = 5.0           # for AE only
 
 
# ── Step 1: load & slice ───────────────────────────────────────────────────
 
def load_and_slice(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[[c for c in KEEP_COLS if c in df.columns]]
    df = df.loc[:, ~df.columns.duplicated()]     # safety dedup
    return df
 
 
# ── Step 2: drop sparse V cols ─────────────────────────────────────────────
 
def drop_sparse(df: pd.DataFrame, ref: pd.DataFrame, threshold: float = SPARSE_THRESHOLD):
    """Fit on ref (legit train), apply to df. Returns (df, dropped_cols)."""
    v_cols   = [f'V{i}' for i in range(1, 51) if f'V{i}' in ref.columns]
    nan_rate = ref[v_cols].isnull().mean()
    to_drop  = nan_rate[nan_rate > threshold].index.tolist()
    return df.drop(columns=to_drop, errors='ignore'), to_drop
 
 
# ── Step 3: missing flags ──────────────────────────────────────────────────
 
def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag for every column that actually contains NaN."""
    for col in df.columns:
        if col in ID_COLS:
            continue
        if df[col].isna().any():
            df[f'{col}_missing'] = df[col].isna().astype(np.int8)
    return df
 
 
# ── Step 4: amount transform ───────────────────────────────────────────────
 
def log_amount(df: pd.DataFrame) -> pd.DataFrame:
    if 'TransactionAmt' in df.columns:
        df['TransactionAmt'] = np.log1p(df['TransactionAmt'].clip(lower=0))
    return df
 
 
# ── Main preprocessor ──────────────────────────────────────────────────────
 
class Preprocessor:
    """
    Fit on full labelled train split.
    Produces:
      - X      : float32 numpy array  (no scaling — for LightGBM)
      - X_scaled: float32 numpy array  (scaled+clipped — for Autoencoder)
      - feature_cols: list of column names (for SHAP)
    """
 
    def __init__(self, sparse_threshold: float = SPARSE_THRESHOLD, clip: float = CLIP_SIGMA):
        self.sparse_threshold = sparse_threshold
        self.clip             = clip
 
        self.dropped_v_cols : list[str]              = []
        self.cat_encoders   : dict[str, LabelEncoder] = {}
        self.num_imputer    = SimpleImputer(strategy='median')
        self.cat_imputer    = SimpleImputer(strategy='most_frequent')
        self.scaler         = StandardScaler()
        self.feature_cols   : list[str]              = []
        self._fitted        = False
 
    # ── fit ────────────────────────────────────────────────────────────────
 
    def fit(self, train_df: pd.DataFrame) -> 'Preprocessor':
        df = train_df.copy()
 
        # 1. Drop sparse V cols (fit reference = full train, not legit-only)
        df, self.dropped_v_cols = drop_sparse(df, df, self.sparse_threshold)
 
        # 2. Missing flags
        df = add_missing_flags(df)
 
        # 3. Log amount
        df = log_amount(df)
 
        # 4. Label-encode categoricals
        for col in CATEGORICAL_COLS:
            if col not in df.columns:
                continue
            le = LabelEncoder()
            le.fit(df[col].astype(str).fillna('Unknown'))
            self.cat_encoders[col] = le
 
        df = self._encode_cats(df)
 
        # 5. Resolve feature columns
        exclude = set(ID_COLS)
        self.feature_cols = [c for c in df.columns if c not in exclude]
 
        # Separate numeric vs flag cols for imputation
        flag_cols = [c for c in self.feature_cols if c.endswith('_missing')]
        num_cols  = [c for c in self.feature_cols if not c.endswith('_missing')]
 
        # 6. Fit imputers
        self.num_imputer.fit(df[num_cols])
        # flags have no NaN by construction — no imputer needed
 
        # 7. Fit scaler on assembled matrix (for AE)
        X_num   = self.num_imputer.transform(df[num_cols])
        X_flags = df[flag_cols].values.astype(np.float32)
        X_full  = np.hstack([X_num, X_flags]).astype(np.float32)
 
        # reorder feature_cols to match hstack
        self._num_cols  = num_cols
        self._flag_cols = flag_cols
        self.feature_cols = num_cols + flag_cols   # canonical order
 
        self.scaler.fit(X_full)
 
        self._fitted = True
        return self
 
    # ── transform ──────────────────────────────────────────────────────────
 
    def transform(self, df: pd.DataFrame):
        assert self._fitted, "Call .fit() first"
        df = df.copy()
 
        df = df.drop(columns=self.dropped_v_cols, errors='ignore')
        df = add_missing_flags(df)
        df = log_amount(df)
        df = self._encode_cats(df)
 
        # Ensure all expected columns exist (handles unseen splits)
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
 
        X_num   = self.num_imputer.transform(df[self._num_cols])
        X_flags = df[self._flag_cols].values.astype(np.float32)
        X       = np.hstack([X_num, X_flags]).astype(np.float32)
 
        # Scaled version for AE
        X_scaled = np.clip(self.scaler.transform(X), -self.clip, self.clip).astype(np.float32)
 
        return X, X_scaled
 
    def fit_transform(self, train_df: pd.DataFrame):
        self.fit(train_df)
        return self.transform(train_df)
 
    # ── internals ──────────────────────────────────────────────────────────
 
    def _encode_cats(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, le in self.cat_encoders.items():
            if col not in df.columns:
                continue
            known = set(le.classes_)
            series = df[col].astype(str).fillna('Unknown')
            df[col] = series.apply(lambda x: int(le.transform([x])[0]) if x in known else -1)
        return df
 
    # ── save/load ───────────────────────────────────────────────────────────
 
    def save(self, path: str):
        joblib.dump(self, path)
 
    @staticmethod
    def load(path: str) -> 'Preprocessor':
        return joblib.load(path)
 
    # ── diagnostics ─────────────────────────────────────────────────────────
 
    def summary(self):
        assert self._fitted
        print(f"Total features  : {len(self.feature_cols)}")
        print(f"  Numeric/cat   : {len(self._num_cols)}")
        print(f"  Missing flags : {len(self._flag_cols)}")
        print(f"  Dropped V cols: {len(self.dropped_v_cols)} → {self.dropped_v_cols}")
        print(f"  Clip sigma    : ±{self.clip}σ  (AE only)")
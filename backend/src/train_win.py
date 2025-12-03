# backend/src/train_win.py

import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from inspect import signature
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

from settings import FEAT_DIR, MODEL_DIR, TRAIN_END, VALID_END, SEED
from utils import season_split

OUT = MODEL_DIR / "win_clf_markets" / "v001"
OUT.mkdir(parents=True, exist_ok=True)


class IsoCal:
    """
    Simple wrapper that applies isotonic calibration to base.predict_proba(...).
    Stores a 2-column probability output to remain sklearn-like.
    """
    def __init__(self, base, iso):
        self.base = base
        self.iso = iso

    def predict_proba(self, X):
        p = self.base.predict_proba(X)[:, 1]
        p = self.iso.transform(p)
        p = np.clip(p, 1e-7, 1 - 1e-7)  # numerical safety
        return np.column_stack([1 - p, p])

def main():
    # ---------- load feature table ----------
    matchups = pd.read_parquet(FEAT_DIR / "matchups.parquet")

    diff_cols = [c for c in matchups.columns if c.startswith("diff_")]
    home_field = []
    market_feats = []
    feature_cols = sorted(set(diff_cols)) + home_field + market_feats
    print("[train_win] using features:", len(feature_cols))
    print("[train_win] diff:", len(diff_cols), "| home_field:", home_field, "| markets:", market_feats)

    # ---------- dataset & splits ----------
    mask = matchups[feature_cols].notnull().all(1) & matchups["home_win"].notnull()
    X = matchups.loc[mask, feature_cols].reset_index(drop=True)
    y = matchups.loc[mask, "home_win"].astype(int).reset_index(drop=True)
    seasons = matchups.loc[mask, "season"].reset_index(drop=True)

    tr, va, te = season_split(seasons, TRAIN_END, VALID_END)
    X_tr, X_va, X_te = X.iloc[tr], X.iloc[va], X.iloc[te]
    y_tr, y_va, y_te = y.iloc[tr], y.iloc[va], y.iloc[te]
    seasons_tr = seasons.iloc[tr]

    print(f"[train_win] split sizes — train {len(tr)}, val {len(va)}, test {len(te)}")

    min_season = seasons_tr.min()
    max_season = seasons_tr.max()
    print(f"[train_win] training seasons: {min_season} to {max_season} ")

    decay_rate = 0.65 
    sample_weights_tr = decay_rate ** (max_season - seasons_tr)
    
    print(f"[train_win] sample weight range: {sample_weights_tr.min():.2f} to {sample_weights_tr.max():.2f}")
    print(f"[train_win] split sizes — train {len(tr)}, val {len(va)}, test {len(te)}")

    # ---------- model ----------
    clf = xgb.XGBClassifier(
        max_depth=4,
        n_estimators=5000,
        learning_rate=0.02,
        subsample=0.8, #uses 80% of the training rows to prevent overfitting
        colsample_bytree=0.8, #uses 80% of the features for more diversity of feature combinations
        reg_lambda=2.0, #L2 regularization
        objective="binary:logistic", #binary classification with logistic loss
        eval_metric="logloss",
        n_jobs=-1, #use all CPU cores
        random_state=SEED,
        verbosity=0, #nothing is printed
    )

    fit_kwargs = {
        "eval_set": [(X_va, y_va)],
        "sample_weight": sample_weights_tr
    }
    clf.fit(X_tr, y_tr, verbose=False, **fit_kwargs)

    if hasattr(clf, "best_iteration"):
        print("[train_win] best_iteration:", clf.best_iteration)

    # ---------- manual isotonic calibration on validation ----------

    p_va_raw = clf.predict_proba(X_va)[:, 1] #Converts XGBOutputs 
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_va_raw, y_va)
    cal = IsoCal(clf, iso)

    # ---------- test metrics ----------
    p_te = cal.predict_proba(X_te)[:, 1]
    print("WIN — Test LogLoss:", round(log_loss(y_te, p_te), 4))
    print("WIN — Test Brier  :", round(brier_score_loss(y_te, p_te), 4))
    print("WIN — Test AUC    :", round(roc_auc_score(y_te, p_te), 4))

    # ---------- persist ----------
    joblib.dump(cal, OUT / "model.joblib")
    with open(OUT / "meta.json", "w") as f:
        json.dump({"features": feature_cols}, f)

    print(f"[train_win] saved model + features to: {OUT}")

    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(clf, max_num_features=30, importance_type='gain', ax=ax)
    plt.title("Top 30 Features by Gain")
    plt.tight_layout()
    plt.savefig(OUT / "feature_importance.png", dpi=150)
    print(f"[train_win] saved feature importance plot to: {OUT / 'feature_importance.png'}")
    plt.close()


if __name__ == "__main__":
    main()
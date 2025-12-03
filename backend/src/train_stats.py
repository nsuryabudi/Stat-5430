import os, json, joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from lightgbm import LGBMRegressor
from settings import FEAT_DIR, MODEL_DIR, TRAIN_END, VALID_END, SEED
from utils import season_split

STAT_COLS = [
    # **ORIGINAL STATS (keep these)**
    "home_passing_yards",
    "home_rushing_yards",
    "home_yards",
    "home_sacks",
    "home_sacks_allowed",
    
    # **PREVIOUSLY MISSING (add these)**
    "home_first_downs",
    "home_penalties",
    "home_penalty_yards",
    "home_fg_made",
    "home_fg_att",
    
    # **NEW HIGH-VALUE STATS (add these)**
    "home_turnovers",
    "home_turnovers_forced",
    "home_yards_per_play",
    "home_qb_hits_for",
    "home_qb_hits_allowed",
    "home_tackles_for_loss",
    "home_passes_defended",
    "home_defensive_tds",
    "home_completion_pct",
    "home_yards_per_carry",
    "home_int_return_yards",
    "home_two_pt_conversions",
]
def rmse(a,b): return root_mean_squared_error(a,b)

def fit_reg_and_q(Xtr, ytr, Xva, yva, sample_weights_tr=None, alpha=0.10):
    reg = LGBMRegressor(
        n_estimators=2000, learning_rate=0.02,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, n_jobs=-1, random_state=SEED, verbose=-1
    )
    
    fit_kwargs = {}
    if sample_weights_tr is not None:
        fit_kwargs["sample_weight"] = sample_weights_tr
    
    reg.fit(Xtr, ytr, **fit_kwargs)
    resid = yva - reg.predict(Xva)
    q = float(np.quantile(np.abs(resid), 1 - alpha))
    return reg, q

def main():
    matchups = pd.read_parquet(FEAT_DIR / "matchups.parquet")

    # rolling diffs
    home_cols = [c for c in matchups.columns
                 if c.startswith("home_") and c.endswith(("_r3","_r5","_r8","_r10","_exp"))]
    # diff_cols = [f"diff_{hc[len('home_'):]}" for hc in home_cols
    #              if f"away_{hc[len('home_'):]}" in matchups.columns]

    # include market features if present (often improves yard/sack models too)
    market_feats = [c for c in ["market_home_prob","spread_line","total_line"] if c in matchups.columns]
    market_feats = []
    #feature_cols = sorted(set(diff_cols)) + market_feats

    feature_cols = sorted(home_cols) + market_feats
    print("[train_stats] using features:", len(feature_cols))
    print("[train_stats] home features:", len(home_cols), "| markets:", market_feats)

    mask = matchups[feature_cols].notnull().all(1)
    X = matchups.loc[mask, feature_cols].reset_index(drop=True)
    seasons = matchups.loc[mask, "season"].reset_index(drop=True)

    tr, va, te = season_split(seasons, TRAIN_END, VALID_END)
    X_tr, X_va, X_te = X.iloc[tr], X.iloc[va], X.iloc[te]

    seasons_tr = seasons.iloc[tr]
    min_season = seasons_tr.min()
    max_season = seasons_tr.max()
    # sample_weights_tr = 1.0 + (seasons_tr - min_season) / (max_season - min_season)
    decay_rate = 0.65
    sample_weights_tr = decay_rate ** (max_season - seasons_tr)
    print(f"[train_stats] sample weight range: {sample_weights_tr.min():.2f} to {sample_weights_tr.max():.2f}")

    for stat_col in STAT_COLS:
        if stat_col not in matchups.columns:
            print(f"Skip {stat_col} — not found.")
            continue

        y = matchups.loc[mask, stat_col].reset_index(drop=True)
        y_tr, y_va, y_te = y.iloc[tr], y.iloc[va], y.iloc[te]

        reg, q = fit_reg_and_q(X_tr, y_tr, X_va, y_va, sample_weights_tr=sample_weights_tr, alpha=0.10)
        y_hat = reg.predict(X_te)
        print(f"{stat_col} — MAE {mean_absolute_error(y_te, y_hat):.1f}  RMSE {rmse(y_te, y_hat):.1f}  q@90% {q:.1f}")

        name = stat_col.replace("home_","")
        outdir = MODEL_DIR / name / "v001"
        outdir.mkdir(parents=True, exist_ok=True)
        joblib.dump(reg, outdir / "model.joblib")
        json.dump({"q": q, "features": feature_cols}, open(outdir / "meta.json","w"))

if __name__ == "__main__":
    main()
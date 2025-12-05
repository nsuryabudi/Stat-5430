import numpy as np
import pandas as pd

# ---------- splits ----------
def season_split(season: pd.Series, train_end: int, valid_end: int):
    # Train: everything up to train_end EXCEPT valid_year
    s = season.values
    
    # Train: everything up to and including train_end
    tr = np.where(s <= train_end)[0]

    # Validation: everything between train_end and valid_end (exclusive of train_end)
    va = np.where((s > train_end) & (s <= valid_end))[0]
    
    # Test: everything after valid_end
    te = np.where(s > valid_end)[0]
    return tr, va, te

# ---------- rolling (leakage-safe) ----------
def add_rolling_means(df, cols, windows=(3,5,8,10), ewm_halflife=3, use_ewm=True):
    """
    Leakage-safe rolling features per team:
      - sort by team, season, week
      - shift(1) so current game is excluded
      - rolling(w, min_periods=1).mean() for each window
      - optional exponential-weighted mean with halflife
    """
    print(f"[utils.add_rolling_means] windows={windows} use_ewm={use_ewm} halflife={ewm_halflife}")

    out = df.sort_values(["team", "season", "week"]).copy()
    g = out.groupby("team", group_keys=False) 

    # Coerce once
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Fixed windows
    for w in windows:
        for c in cols:
            if c not in out.columns:
                continue
            if out[c].isna().all():
                continue
            out[f"{c}_r{w}"] = g[c].shift(1).rolling(w, min_periods=1).mean()

    # Exponential window
    if use_ewm:
        for c in cols:
            if c not in out.columns:
                continue
            if out[c].isna().all():
                continue
            ewm_series = g[c].apply(lambda s: s.shift(1).ewm(halflife=ewm_halflife, adjust=False).mean())
            out[f"{c}_exp"] = ewm_series

    return out

# ---------- team code normalization (optional, handy in infer) ----------
TEAM_ALIASES = {"LAR":"LA","STL":"LA","SD":"LAC","OAK":"LV"}
def normalize_team(code: str) -> str:
    return TEAM_ALIASES.get(code, code)
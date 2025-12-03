# backend/app/main.py

# ---------------- IsoCal shim for legacy pickles (MUST run before joblib.load) ----------------
import sys, types, numpy as np
class IsoCal:  # compatible interface with what was saved
    def __init__(self, base, iso):
        self.base = base
        self.iso = iso
    def predict_proba(self, X):
        p = self.base.predict_proba(X)[:, 1]
        p = self.iso.transform(p)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.column_stack([1 - p, p])

# Register IsoCal on all likely module paths the pickle might reference
for name in ("backend.src.infer", "__main__", "__mp_main__"):
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    setattr(mod, "IsoCal", IsoCal)
# ---------------------------------------------------------------------------------------------

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import json, joblib
import pandas as pd
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
FEAT_DIR = BACKEND / "data" / "features"
MODEL_DIR = BACKEND / "models"

# ---------- helpers ----------
TEAM_ALIASES = {"LAR": "LA", "STL": "LA", "SD": "LAC", "OAK": "LV"}
def normalize_team(code: str) -> str:
    return TEAM_ALIASES.get(code, code)

def american_to_prob(odds: float) -> float:
    if odds is None: return np.nan
    o = float(odds)
    if np.isnan(o): return np.nan
    return (100.0 / (o + 100.0)) if o > 0 else (-o / (-o + 100.0))

def remove_vig(p_home_raw: float, p_away_raw: float):
    if np.isnan(p_home_raw) or np.isnan(p_away_raw): return (np.nan, np.nan, np.nan)
    s = p_home_raw + p_away_raw
    if s <= 0: return (np.nan, np.nan, np.nan)
    return p_home_raw / s, p_away_raw / s, s - 1.0

def _load_features(dirpath: Path) -> list[str]:
    with open(dirpath / "meta.json") as f:
        return json.load(f)["features"]

def ensure_features_numeric_ordered(X: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    for c in features:
        if c not in X.columns:
            X[c] = np.nan
    X = X[features]
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.astype("float32")

# ---------- load data at startup ----------
TEAM_PREGAME = pd.read_parquet(FEAT_DIR / "team_pregame.parquet").copy()

def latest_row(team: str) -> pd.Series:
    team = normalize_team(team)
    sub = TEAM_PREGAME.loc[TEAM_PREGAME["team"] == team]
    if sub.empty:
        raise HTTPException(400, f"No features found for team '{team}'.")
    return sub.sort_values(["season", "week"]).iloc[-1]

def build_diff_row(home: str, away: str, include_absolute: bool = False) -> pd.DataFrame:
    """Home-away diff features using same windows as training (_r3,_r5,_r8,_r10,_exp)."""
    h = latest_row(home); a = latest_row(away)
    roll_cols = [c for c in TEAM_PREGAME.columns if c.endswith(("_r3","_r5","_r8","_r10","_exp"))]
    home_abs = {f"home_{c}": h.get(c, np.nan) for c in roll_cols}
    away_abs = {f"away_{c}": a.get(c, np.nan) for c in roll_cols}
    diffs    = {f"diff_{c}": home_abs[f"home_{c}"] - away_abs[f"away_{c}"] for c in roll_cols}
    row = {}
    if include_absolute:
        row.update(home_abs); row.update(away_abs)
    row.update(diffs)
    return pd.DataFrame([row])

def add_markets(X: pd.DataFrame,
                spread_line: float|None,
                total_line: float|None,
                home_moneyline: int|None,
                away_moneyline: int|None) -> pd.DataFrame:
    extra = {}
    if spread_line is not None:
        extra["spread_line"] = float(spread_line)
    if total_line is not None:
        extra["total_line"] = float(total_line)
    if (home_moneyline is not None) and (away_moneyline is not None):
        ph = american_to_prob(home_moneyline)
        pa = american_to_prob(away_moneyline)
        ph_nv, _, _ = remove_vig(ph, pa)
        extra["market_home_prob"] = ph_nv
    return X.assign(**extra) if extra else X

# ---------- load models ----------
def _load_win():
    """
    Prefer robust artifacts (base_model + iso). If not present, fall back to legacy
    single pickle (which will now unpickle thanks to the shim above).
    """
    d = MODEL_DIR / "win_clf_markets" / "v001"
    if not d.exists():
        d = MODEL_DIR / "win_clf" / "v001"

    feats = _load_features(d)

    base_path = d / "base_model.joblib"
    iso_path  = d / "iso.joblib"
    if base_path.exists() and iso_path.exists():
        base = joblib.load(base_path)
        iso  = joblib.load(iso_path)
        class _Cal:
            def __init__(self, base, iso): self.base, self.iso = base, iso
            def predict_proba(self, X):
                p = self.base.predict_proba(X)[:, 1]
                p = self.iso.transform(p)
                p = np.clip(p, 1e-7, 1 - 1e-7)
                return np.column_stack([1 - p, p])
        return _Cal(base, iso), feats, d

    # Legacy: contains IsoCal inside the pickle
    cal = joblib.load(d / "model.joblib")
    return cal, feats, d

def _load_stat_models():
    #names = ["passing_yards", "rushing_yards", "sacks", "sacks_allowed", "yards"]
    names = [
        "passing_yards", "rushing_yards", "yards", "sacks", "sacks_allowed",
        "first_downs", "penalties", "penalty_yards", "fg_made", "fg_att",
        "turnovers", "turnovers_forced", "yards_per_play",
        "qb_hits_for", "qb_hits_allowed",
        "tackles_for_loss", "passes_defended", "defensive_tds",
        "completion_pct", "yards_per_carry",
        "int_return_yards", "two_pt_conversions",
    ]
    stat_models, stat_features, qmap = {}, {}, {}
    for n in names:
        d = MODEL_DIR / n / "v001"
        if (d / "model.joblib").exists() and (d / "meta.json").exists():
            stat_models[n]   = joblib.load(d / "model.joblib")
            stat_features[n] = _load_features(d)
            qmap[n]          = float(json.load(open(d / "meta.json")).get("q", 0.0))
    return stat_models, stat_features, qmap

CAL, WIN_FEATURES, WIN_DIR = _load_win()
STAT_MODELS, STAT_FEATURES, CONFORMAL_Q = _load_stat_models()

TEAM_CODES = [
  "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB",
  "HOU","IND","JAX","KC","LAC","LA","LV","MIA","MIN","NE","NO","NYG","NYJ",
  "PHI","PIT","SEA","SF","TB","TEN","WAS"
]

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NFL Matchup Predictor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

from pydantic import BaseModel, field_validator

class PredictIn(BaseModel):
    home: str
    away: str
    spread_line: float | None = None
    total_line: float | None = None
    home_moneyline: int | None = None
    away_moneyline: int | None = None

    @field_validator("home","away")
    @classmethod
    def v_team(cls, v):
        if v not in TEAM_CODES:
            raise ValueError(f"Team '{v}' not in supported list.")
        return v

@app.get("/teams")
def teams():
    return {"teams": TEAM_CODES}

@app.get("/model_info")
def model_info():
    return {
        "win_model_dir": str(WIN_DIR),
        "win_num_features": len(WIN_FEATURES),
        "win_uses_markets": any(c in WIN_FEATURES for c in ["market_home_prob","spread_line","total_line"]),
        "stat_models": {k: len(v) for k, v in STAT_FEATURES.items()}
    }

def predict_two(home: str, away: str,
                spread_line: float|None = None,
                total_line: float|None = None,
                home_moneyline: int|None = None,
                away_moneyline: int|None = None):
    
    # ---------- BUILD FEATURES ----------
    # Get absolute home and away features
    h_row = latest_row(home)
    a_row = latest_row(away)
    
    roll_cols = [c for c in TEAM_PREGAME.columns if c.endswith(("_r3","_r5","_r8","_r10","_exp"))]
    
    # Build home_* and away_* features (for stat models)
    home_abs = {f"home_{c}": h_row.get(c, np.nan) for c in roll_cols}
    away_abs = {f"away_{c}": a_row.get(c, np.nan) for c in roll_cols}
    
    # Build diff_* features (for win model)
    diffs = {f"diff_{c}": home_abs[f"home_{c}"] - away_abs[f"away_{c}"] for c in roll_cols}
    
    # Combine all features
    all_features = {**home_abs, **away_abs, **diffs}
    all_features["is_home"] = 0.3  # Add home field indicator
    
    X_all = pd.DataFrame([all_features])
    X_all = add_markets(X_all, spread_line, total_line, home_moneyline, away_moneyline)
    
    # ---------- WIN PROBABILITY ----------
    X_win = ensure_features_numeric_ordered(X_all.copy(), WIN_FEATURES)
    p_home = float(CAL.predict_proba(X_win)[:, 1])
    
    # ---------- HOME STATS ----------
    home_stats = {}
    for name, reg in STAT_MODELS.items():
        feats = STAT_FEATURES[name]
        Xh = ensure_features_numeric_ordered(X_all.copy(), feats)
        yhat = float(reg.predict(Xh)[0])
        q = float(CONFORMAL_Q.get(name, 0.0))
        home_stats[f"home_{name}"] = {"pred": yhat, "lo": yhat - q, "hi": yhat + q}
    
    # ---------- AWAY STATS (flip perspective) ----------
    # Rebuild features from away perspective
    away_abs_flip = {f"home_{c}": a_row.get(c, np.nan) for c in roll_cols}
    home_abs_flip = {f"away_{c}": h_row.get(c, np.nan) for c in roll_cols}
    diffs_flip = {f"diff_{c}": away_abs_flip[f"home_{c}"] - home_abs_flip[f"away_{c}"] for c in roll_cols}
    
    all_features_flip = {**away_abs_flip, **home_abs_flip, **diffs_flip}
    all_features_flip["is_home"] = 0.3
    
    X_all_flip = pd.DataFrame([all_features_flip])
    X_all_flip = add_markets(X_all_flip, spread_line, total_line, away_moneyline, home_moneyline)
    
    away_stats = {}
    for name, reg in STAT_MODELS.items():
        feats = STAT_FEATURES[name]
        Xa = ensure_features_numeric_ordered(X_all_flip.copy(), feats)
        yhat = float(reg.predict(Xa)[0])
        q = float(CONFORMAL_Q.get(name, 0.0))
        away_stats[f"away_{name}"] = {"pred": yhat, "lo": yhat - q, "hi": yhat + q}
    
    return {
        "home_team": normalize_team(home),
        "away_team": normalize_team(away),
        "context": "latest available pregame features",
        "model_used": str(WIN_DIR),
        "win_prob_home": p_home,
        "win_prob_away": 1.0 - p_home,
        "stats": {"home": home_stats, "away": away_stats}
    }

@app.post("/predict")
def predict(inp: PredictIn):
    if inp.home == inp.away:
        raise HTTPException(400, "home and away cannot be the same team.")
    return predict_two(
        inp.home, inp.away,
        spread_line=inp.spread_line,
        total_line=inp.total_line,
        home_moneyline=inp.home_moneyline,
        away_moneyline=inp.away_moneyline,
    )

@app.get("/health")
def health():
    return {"ok": True}
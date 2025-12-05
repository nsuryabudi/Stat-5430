# ============================================================================
# IsoCal shim for legacy pickles
# This MUST run before joblib.load() to prevent unpickling errors.
# Provides a fallback IsoCal class definition in case Python can't find
# the original class used during model training. This prevents unpickling
# errors when loading models trained in different modules/contexts.
# ============================================================================
import sys
import types
import numpy as np


class IsoCal:
    """Fallback interface for pickled models that reference IsoCal."""
    def __init__(self, base, iso):
        self.base = base
        self.iso = iso
    
    def predict_proba(self, X):
        """Apply isotonic calibration to base model predictions."""
        p = self.base.predict_proba(X)[:, 1]
        p = self.iso.transform(p)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.column_stack([1 - p, p])


# Register IsoCal as a fallback in all module paths the pickle might reference
for name in ("backend.src.infer", "__main__", "__mp_main__"):
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    setattr(mod, "IsoCal", IsoCal)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import json
import joblib
import pandas as pd
from pathlib import Path


# ============================================================================
# Path Configuration
# ============================================================================
ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
FEAT_DIR = BACKEND / "data" / "features"
MODEL_DIR = BACKEND / "models"


# ============================================================================
# Team Code Normalization
# Handle NFL team relocations (e.g., St. Louis → LA, Oakland → Las Vegas)
# ============================================================================
TEAM_ALIASES = {
    "LAR": "LA",   # Rams relocation
    "STL": "LA",   # Legacy Rams code
    "SD": "LAC",   # Chargers relocation
    "OAK": "LV",   # Raiders relocation
}


def normalize_team(code: str) -> str:
    """Normalize team codes to handle relocations."""
    return TEAM_ALIASES.get(code, code)


# ============================================================================
# Model Loading Helpers
# ============================================================================

def _load_features(dirpath: Path) -> list[str]:
    """Load feature list from meta.json."""
    with open(dirpath / "meta.json") as f:
        return json.load(f)["features"]


def ensure_features_numeric_ordered(X: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Ensure DataFrame has all required features in correct order and numeric dtypes.
    
    Models expect features in exact order they were trained on. This function:
    1. Adds missing features as NaN
    2. Reorders columns to match training
    3. Converts all columns to numeric (coercing errors to NaN)
    4. Casts to float32 for efficiency
    """
    for c in features:
        if c not in X.columns:
            X[c] = np.nan
    X = X[features]
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.astype("float32")


# ============================================================================
# Load Team Features
# Contains rolling averages of team stats (points, yards, etc.) by season/week
# ============================================================================
TEAM_PREGAME = pd.read_parquet(FEAT_DIR / "team_pregame.parquet").copy()


def latest_row(team: str) -> pd.Series:
    """
    Get most recent pregame features for a team.
    
    Returns the latest available season/week row for the specified team,
    containing rolling averages of their performance metrics.
    """
    team = normalize_team(team)
    sub = TEAM_PREGAME.loc[TEAM_PREGAME["team"] == team]
    if sub.empty:
        raise HTTPException(400, f"No features found for team '{team}'.")
    return sub.sort_values(["season", "week"]).iloc[-1]


def _load_win():
    """
    Load win probability model (XGBoost + isotonic calibration).
    
    Returns:
        - Calibrated model (base XGBoost + isotonic regression)
        - List of feature names
        - Model directory path
    """
    d = MODEL_DIR / "win_clf" / "v001"
    if not d.exists():
        raise FileNotFoundError(f"Win model not found at {d}")
    
    feats = _load_features(d)
    
    # Try loading separate files (preferred structure)
    base_path = d / "base_model.joblib"
    iso_path = d / "iso.joblib"
    
    if base_path.exists() and iso_path.exists():
        base = joblib.load(base_path)
        iso = joblib.load(iso_path)
        
        class _Cal:
            """Wrapper combining base model and isotonic calibration."""
            def __init__(self, base, iso):
                self.base = base
                self.iso = iso
            
            def predict_proba(self, X):
                p = self.base.predict_proba(X)[:, 1]
                p = self.iso.transform(p)
                p = np.clip(p, 1e-7, 1 - 1e-7)
                return np.column_stack([1 - p, p])
        
        return _Cal(base, iso), feats, d
    
    # Fallback: legacy single pickle
    cal = joblib.load(d / "model.joblib")
    return cal, feats, d


def _load_stat_models():
    """
    Load all individual stat prediction models (LightGBM regressors).
    
    Returns:
        - Dictionary of {stat_name: model}
        - Dictionary of {stat_name: feature_list}
        - Dictionary of {stat_name: conformal_q} for prediction intervals
    """
    stat_names = [
        "passing_yards", "rushing_yards", "yards",
        "sacks", "sacks_allowed",
        "first_downs", "penalties", "penalty_yards",
        "fg_made", "fg_att",
        "turnovers", "turnovers_forced", "yards_per_play",
        "qb_hits_for", "qb_hits_allowed",
        "tackles_for_loss", "passes_defended", "defensive_tds",
        "completion_pct", "yards_per_carry",
        "int_return_yards", "two_pt_conversions",
    ]
    
    stat_models = {}
    stat_features = {}
    conformal_q = {}
    
    for name in stat_names:
        model_dir = MODEL_DIR / name / "v001"
        model_path = model_dir / "model.joblib"
        meta_path = model_dir / "meta.json"
        
        if model_path.exists() and meta_path.exists():
            stat_models[name] = joblib.load(model_path)
            stat_features[name] = _load_features(model_dir)
            
            with open(meta_path) as f:
                meta = json.load(f)
                # q is the 90th percentile of absolute residuals from validation
                conformal_q[name] = float(meta.get("q", 0.0))
    
    return stat_models, stat_features, conformal_q


# ============================================================================
# Load Models at Startup
# ============================================================================
CAL, WIN_FEATURES, WIN_DIR = _load_win()
STAT_MODELS, STAT_FEATURES, CONFORMAL_Q = _load_stat_models()


# ============================================================================
# Supported NFL Teams
# ============================================================================
TEAM_CODES = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LAC", "LA", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"
]


# ============================================================================
# FastAPI App Setup
# ============================================================================
app = FastAPI(title="NFL Matchup Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (tighten for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request Validation
# ============================================================================

class PredictIn(BaseModel):
    """Input validation for prediction requests."""
    home: str
    away: str
    
    @field_validator("home", "away")
    @classmethod
    def validate_team(cls, v):
        """Ensure team code is valid."""
        if v not in TEAM_CODES:
            raise ValueError(f"Team '{v}' not in supported list: {TEAM_CODES}")
        return v


# ============================================================================
# Prediction Logic
# ============================================================================

def predict_matchup(home: str, away: str):
    """
    Generate win probability and stat predictions for a matchup.
    
    Process:
    1. Load latest rolling features for both teams
    2. Build home_*, away_*, and diff_* features
    3. Predict home win probability using calibrated XGBoost
    4. Predict home team stats using LightGBM regressors
    5. Predict away team stats by applying models to away team's features
    6. Return nested JSON with win probs and stat predictions (with intervals)
    
    Args:
        home: Home team code
        away: Away team code
    
    Returns:
        Dictionary with win probabilities and stat predictions
    """
    # Load latest features for both teams
    h_row = latest_row(home)
    a_row = latest_row(away)
    
    # Get all rolling feature column names (e.g., points_r3, yards_exp)
    roll_cols = [
        c for c in TEAM_PREGAME.columns 
        if c.endswith(("_r3", "_r5", "_r8", "_r10", "_exp"))
    ]
    
    # Build feature dictionaries for win model
    # Win model uses home_*, away_*, and diff_* (home - away) features
    home_features = {f"home_{c}": h_row.get(c, np.nan) for c in roll_cols}
    away_features = {f"away_{c}": a_row.get(c, np.nan) for c in roll_cols}
    diff_features = {
        f"diff_{c}": home_features[f"home_{c}"] - away_features[f"away_{c}"]
        for c in roll_cols
    }
    
    # Combine all features into single DataFrame
    all_features = {**home_features, **away_features, **diff_features}
    X_all = pd.DataFrame([all_features])
    
    # ========================================================================
    # WIN PROBABILITY PREDICTION
    # ========================================================================
    X_win = ensure_features_numeric_ordered(X_all.copy(), WIN_FEATURES)
    p_home = float(CAL.predict_proba(X_win)[:, 1])
    
    # ========================================================================
    # HOME TEAM STAT PREDICTIONS
    # Stat models use only home_* features (trained on home team performance)
    # ========================================================================
    home_stats = {}
    for stat_name, model in STAT_MODELS.items():
        features = STAT_FEATURES[stat_name]
        
        # Map home team's rolling features to home_* feature names
        stat_features = {feat: h_row.get(feat.replace("home_", ""), np.nan) 
                        for feat in features if feat.startswith("home_")}
        X_stat = pd.DataFrame([stat_features])
        X_stat = ensure_features_numeric_ordered(X_stat, features)
        
        prediction = float(model.predict(X_stat)[0])
        uncertainty = float(CONFORMAL_Q.get(stat_name, 0.0))
        
        # Store prediction with ~90% prediction interval
        home_stats[f"home_{stat_name}"] = {
            "pred": prediction,
            "lo": prediction - uncertainty,
            "hi": prediction + uncertainty
        }
    
    # ========================================================================
    # AWAY TEAM STAT PREDICTIONS
    # Apply same models to away team's features (NOT flipped perspectives)
    # ========================================================================
    away_stats = {}
    for stat_name, model in STAT_MODELS.items():
        features = STAT_FEATURES[stat_name]
        
        # Map away team's rolling features to home_* feature names
        # (Models expect home_* column names, but we feed away team's data)
        stat_features = {feat: a_row.get(feat.replace("home_", ""), np.nan) 
                        for feat in features if feat.startswith("home_")}
        X_stat = pd.DataFrame([stat_features])
        X_stat = ensure_features_numeric_ordered(X_stat, features)
        
        prediction = float(model.predict(X_stat)[0])
        uncertainty = float(CONFORMAL_Q.get(stat_name, 0.0))
        
        away_stats[f"away_{stat_name}"] = {
            "pred": prediction,
            "lo": prediction - uncertainty,
            "hi": prediction + uncertainty
        }
    
    return {
        "home_team": normalize_team(home),
        "away_team": normalize_team(away),
        "context": "latest available pregame features",
        "model_used": str(WIN_DIR),
        "win_prob_home": p_home,
        "win_prob_away": 1.0 - p_home,
        "stats": {
            "home": home_stats,
            "away": away_stats
        }
    }


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"ok": True}


@app.get("/teams")
def get_teams():
    """Get list of supported team codes."""
    return {"teams": TEAM_CODES}


@app.get("/model_info")
def model_info():
    """Get information about loaded models."""
    return {
        "win_model_dir": str(WIN_DIR),
        "win_num_features": len(WIN_FEATURES),
        "stat_models": {name: len(feats) for name, feats in STAT_FEATURES.items()}
    }


@app.post("/predict")
def predict(inp: PredictIn):
    """
    Predict matchup outcome and stats.
    
    Args:
        inp: PredictIn object with home and away team codes
    
    Returns:
        Win probabilities and stat predictions
    """
    if inp.home == inp.away:
        raise HTTPException(400, "home and away cannot be the same team.")
    
    return predict_matchup(inp.home, inp.away)


@app.get("/debug/team/{team_code}")
def debug_team_features(team_code: str):
    """Debug endpoint to inspect team features."""
    if team_code not in TEAM_CODES:
        raise HTTPException(400, f"Invalid team: {team_code}")
    
    team = normalize_team(team_code)
    row = latest_row(team)
    
    roll_cols = [c for c in TEAM_PREGAME.columns if c.endswith(("_r3", "_r5", "_r8", "_r10", "_exp"))]
    
    features = {}
    for col in roll_cols[:20]:
        val = row.get(col, np.nan)
        features[col] = float(val) if not np.isnan(val) else None
    
    return {
        "team": team,
        "season": int(row.get("season", 0)),
        "week": int(row.get("week", 0)),
        "total_features": len(roll_cols),
        "sample_features": features,
        "nan_count": sum(1 for col in roll_cols if np.isnan(row.get(col, np.nan)))
    }
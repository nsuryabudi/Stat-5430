import numpy as np
import pandas as pd
from settings import RAW_DIR, FEAT_DIR
from utils import add_rolling_means, american_to_prob, remove_vig

FEAT_DIR.mkdir(parents=True, exist_ok=True)

def build_base(team_week: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    games = schedules.sort_values(["season","week"]).copy()

    h = games[["season","week","home_team","home_score"]].rename(
        columns={"home_team":"team","home_score":"points"})
    a = games[["season","week","away_team","away_score"]].rename(
        columns={"away_team":"team","away_score":"points"})
    points_by_team = pd.concat([h,a], ignore_index=True)

    ha = games[["season","week","home_team","away_team"]]
    aa = ha.rename(columns={"home_team":"away_team","away_team":"home_team"})
    pair_long = pd.concat([ha, aa], ignore_index=True).rename(
        columns={"home_team":"team","away_team":"opp_team"}).drop_duplicates()

    base = team_week[["season","week","team"]].drop_duplicates().copy()
    base = base.merge(points_by_team, on=["season","week","team"], how="left")

    tw = team_week
    
    # **SACKS**
    if "def_sacks" in tw.columns:
        sacks_for = tw[["season","week","team","def_sacks"]].rename(columns={"def_sacks":"sacks_for"})
        base = base.merge(sacks_for, on=["season","week","team"], how="left")
    else:
        base["sacks_for"] = np.nan
    
    if "sacks_suffered" in tw.columns:
        sacks_allowed = tw[["season","week","team","sacks_suffered"]].rename(columns={"sacks_suffered":"sacks_allowed"})
        base = base.merge(sacks_allowed, on=["season","week","team"], how="left")
    else:
        base["sacks_allowed"] = np.nan

    # **YARDS**
    if "passing_yards" in tw.columns:
        base = base.merge(tw[["season","week","team","passing_yards"]], on=["season","week","team"], how="left")
    else:
        base["passing_yards"] = np.nan
    
    if "rushing_yards" in tw.columns:
        base = base.merge(tw[["season","week","team","rushing_yards"]], on=["season","week","team"], how="left")
    else:
        base["rushing_yards"] = np.nan
    
    if {"passing_yards","rushing_yards"}.issubset(base.columns):
        base["yards"] = base["passing_yards"].fillna(0) + base["rushing_yards"].fillna(0)
    else:
        base["yards"] = np.nan

    # **FIRST DOWNS**
    base["first_downs"] = 0
    for col in ["passing_first_downs", "rushing_first_downs", "receiving_first_downs"]:
        if col in tw.columns:
            fd_data = tw[["season","week","team",col]]
            fd_data = fd_data.groupby(["season","week","team"])[col].sum().reset_index()
            base = base.merge(fd_data, on=["season","week","team"], how="left", suffixes=('', '_temp'))
            base["first_downs"] = base["first_downs"].fillna(0) + base[f"{col}_temp" if f"{col}_temp" in base.columns else col].fillna(0)
            if f"{col}_temp" in base.columns:
                base = base.drop(columns=[f"{col}_temp"])
    print(f"first_downs: non-null = {base['first_downs'].notnull().sum()}, non-zero = {(base['first_downs'] != 0).sum()}")

    # **OTHER BASIC STATS**
    other_stats = ["penalties", "penalty_yards", "fg_made", "fg_att"]
    for stat in other_stats:
        if stat in tw.columns:
            base = base.merge(tw[["season","week","team",stat]], on=["season","week","team"], how="left")
            print(f"{stat}: non-null = {base[stat].notnull().sum()}, non-zero = {(base[stat] != 0).sum()}")
        else:
            base[stat] = np.nan

    turnovers_cols = ["passing_interceptions", "rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost"]
    base["turnovers"] = 0
    for col in turnovers_cols:
        if col in tw.columns:
            turnovers_data = tw[["season","week","team",col]]
            base = base.merge(turnovers_data, on=["season","week","team"], how="left", suffixes=('', '_temp'))
            base["turnovers"] = base["turnovers"].fillna(0) + base[f"{col}_temp" if f"{col}_temp" in base.columns else col].fillna(0)
            if f"{col}_temp" in base.columns:
                base = base.drop(columns=[f"{col}_temp"])
    print(f"turnovers: non-null = {base['turnovers'].notnull().sum()}, non-zero = {(base['turnovers'] != 0).sum()}")

    turnovers_forced_cols = ["def_interceptions", "def_fumbles"]
    base["turnovers_forced"] = 0
    for col in turnovers_forced_cols:
        if col in tw.columns:
            tf_data = tw[["season","week","team",col]]
            base = base.merge(tf_data, on=["season","week","team"], how="left", suffixes=('', '_temp'))
            base["turnovers_forced"] = base["turnovers_forced"].fillna(0) + base[f"{col}_temp" if f"{col}_temp" in base.columns else col].fillna(0)
            if f"{col}_temp" in base.columns:
                base = base.drop(columns=[f"{col}_temp"])
    print(f"turnovers_forced: non-null = {base['turnovers_forced'].notnull().sum()}, non-zero = {(base['turnovers_forced'] != 0).sum()}")

    if "attempts" in tw.columns and "carries" in tw.columns:
        plays_data = tw[["season","week","team","attempts","carries"]].copy()
        plays_data["plays"] = plays_data["attempts"].fillna(0) + plays_data["carries"].fillna(0)
        base = base.merge(plays_data[["season","week","team","plays"]], on=["season","week","team"], how="left")
        
        if "yards" in base.columns and "plays" in base.columns:
            base["yards_per_play"] = base["yards"] / base["plays"].replace(0, np.nan)
            print(f"yards_per_play: non-null = {base['yards_per_play'].notnull().sum()}, non-zero = {(base['yards_per_play'] != 0).sum()}")
        else:
            base["yards_per_play"] = np.nan
    else:
        base["yards_per_play"] = np.nan

    if "def_qb_hits" in tw.columns:
        base = base.merge(
            tw[["season","week","team","def_qb_hits"]].rename(columns={"def_qb_hits":"qb_hits_for"}),
            on=["season","week","team"], how="left"
        )
        
        opp_qb_hits = tw[["season","week","team","def_qb_hits"]].rename(
            columns={"team":"opp_team", "def_qb_hits":"qb_hits_allowed"})
        base_temp = base.merge(pair_long, on=["season","week","team"], how="left")
        base_temp = base_temp.merge(opp_qb_hits, on=["season","week","opp_team"], how="left")
        base["qb_hits_allowed"] = base_temp["qb_hits_allowed"].values
        
        print(f"qb_hits_for: non-null = {base['qb_hits_for'].notnull().sum()}, non-zero = {(base['qb_hits_for'] != 0).sum()}")
        print(f"qb_hits_allowed: non-null = {base['qb_hits_allowed'].notnull().sum()}, non-zero = {(base['qb_hits_allowed'] != 0).sum()}")
    else:
        base["qb_hits_for"] = np.nan
        base["qb_hits_allowed"] = np.nan

    if "def_tackles_for_loss" in tw.columns:
        base = base.merge(
            tw[["season","week","team","def_tackles_for_loss"]].rename(
                columns={"def_tackles_for_loss":"tackles_for_loss"}),
            on=["season","week","team"], how="left"
        )
        print(f"tackles_for_loss: non-null = {base['tackles_for_loss'].notnull().sum()}, non-zero = {(base['tackles_for_loss'] != 0).sum()}")
    else:
        base["tackles_for_loss"] = np.nan

    if "def_pass_defended" in tw.columns:
        base = base.merge(
            tw[["season","week","team","def_pass_defended"]].rename(
                columns={"def_pass_defended":"passes_defended"}),
            on=["season","week","team"], how="left"
        )
        print(f"passes_defended: non-null = {base['passes_defended'].notnull().sum()}, non-zero = {(base['passes_defended'] != 0).sum()}")
    else:
        base["passes_defended"] = np.nan

    if "def_tds" in tw.columns:
        base = base.merge(
            tw[["season","week","team","def_tds"]].rename(columns={"def_tds":"defensive_tds"}),
            on=["season","week","team"], how="left"
        )
        print(f"defensive_tds: non-null = {base['defensive_tds'].notnull().sum()}, non-zero = {(base['defensive_tds'] != 0).sum()}")
    else:
        base["defensive_tds"] = np.nan

    if "completions" in tw.columns and "attempts" in tw.columns:
        comp_data = tw[["season","week","team","completions","attempts"]].copy()
        comp_data["completion_pct"] = comp_data["completions"] / comp_data["attempts"].replace(0, np.nan)
        base = base.merge(
            comp_data[["season","week","team","completion_pct"]],
            on=["season","week","team"], how="left"
        )
        print(f"completion_pct: non-null = {base['completion_pct'].notnull().sum()}, non-zero = {(base['completion_pct'] != 0).sum()}")
    else:
        base["completion_pct"] = np.nan

    if "rushing_yards" in tw.columns and "carries" in tw.columns:
        rush_data = tw[["season","week","team","rushing_yards","carries"]].copy()
        rush_data["yards_per_carry"] = rush_data["rushing_yards"] / rush_data["carries"].replace(0, np.nan)
        base = base.merge(
            rush_data[["season","week","team","yards_per_carry"]],
            on=["season","week","team"], how="left"
        )
        print(f"yards_per_carry: non-null = {base['yards_per_carry'].notnull().sum()}, non-zero = {(base['yards_per_carry'] != 0).sum()}")
    else:
        base["yards_per_carry"] = np.nan

    if "def_interception_yards" in tw.columns:
        base = base.merge(
            tw[["season","week","team","def_interception_yards"]].rename(
                columns={"def_interception_yards":"int_return_yards"}),
            on=["season","week","team"], how="left"
        )
        print(f"int_return_yards: non-null = {base['int_return_yards'].notnull().sum()}, non-zero = {(base['int_return_yards'] != 0).sum()}")
    else:
        base["int_return_yards"] = np.nan

    if "passing_2pt_conversions" in tw.columns and "rushing_2pt_conversions" in tw.columns:
        twopts_data = tw[["season","week","team","passing_2pt_conversions","rushing_2pt_conversions"]].copy()
        twopts_data["two_pt_conversions"] = twopts_data["passing_2pt_conversions"].fillna(0) + twopts_data["rushing_2pt_conversions"].fillna(0)
        base = base.merge(
            twopts_data[["season","week","team","two_pt_conversions"]],
            on=["season","week","team"], how="left"
        )
        print(f"two_pt_conversions: non-null = {base['two_pt_conversions'].notnull().sum()}, non-zero = {(base['two_pt_conversions'] != 0).sum()}")
    else:
        base["two_pt_conversions"] = np.nan

    base["third_down_pct"] = np.nan
    base["red_zone_td_pct"] = np.nan
    base["time_of_possession"] = np.nan

    return base

def build_labels(games: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    labels = games.copy()
    labels["home_win"] = (labels["home_score"] > labels["away_score"]).astype("int8")

    cols = [
        "season", "week", "team",
        "points", "yards", "passing_yards", "rushing_yards", 
        "sacks_for", "sacks_allowed",
        "first_downs", "penalties", "penalty_yards", 
        "fg_made", "fg_att",
        "turnovers", "turnovers_forced", "yards_per_play",
        "qb_hits_for", "qb_hits_allowed",
        "tackles_for_loss", "passes_defended", "defensive_tds",
        "completion_pct", "yards_per_carry", 
        "int_return_yards", "two_pt_conversions",
    ]
    cols = [c for c in cols if c in base.columns or c in ["season", "week", "team"]]
    team_idx = base[cols].set_index(["season","week","team"])

    def map_stat(stat, hn=None, an=None):
        if stat not in team_idx.columns: return
        hn = hn or f"home_{stat}"
        an = an or f"away_{stat}"
        labels[hn] = labels.set_index(["season","week","home_team"]).index.map(team_idx[stat])
        labels[an] = labels.set_index(["season","week","away_team"]).index.map(team_idx[stat])

    if "points" in team_idx.columns:
        map_stat("points","home_points","away_points")
    else:
        labels["home_points"] = labels["home_score"].astype("float32")
        labels["away_points"] = labels["away_score"].astype("float32")

    for stat in ["yards", "passing_yards", "rushing_yards"]:
        map_stat(stat)
    
    map_stat("sacks_for", "home_sacks", "away_sacks")
    map_stat("sacks_allowed", "home_sacks_allowed", "away_sacks_allowed")

    for stat in ["first_downs", "penalties", "penalty_yards", "fg_made", "fg_att"]:
        map_stat(stat)

    for stat in [
        "turnovers", "turnovers_forced", "yards_per_play",
        "qb_hits_for", "qb_hits_allowed",
        "tackles_for_loss", "passes_defended", "defensive_tds",
        "completion_pct", "yards_per_carry",
        "int_return_yards", "two_pt_conversions"
    ]:
        map_stat(stat)

    keep = [
        "game_id", "season", "week", "home_team", "away_team", "home_win",
        "home_points", "away_points",
        "home_yards", "away_yards",
        "home_passing_yards", "away_passing_yards",
        "home_rushing_yards", "away_rushing_yards",
        "home_sacks", "away_sacks",
        "home_sacks_allowed", "away_sacks_allowed",
        "home_first_downs", "away_first_downs",
        "home_penalties", "away_penalties",
        "home_penalty_yards", "away_penalty_yards",
        "home_fg_made", "away_fg_made",
        "home_fg_att", "away_fg_att",
        "home_turnovers", "away_turnovers",
        "home_turnovers_forced", "away_turnovers_forced",
        "home_yards_per_play", "away_yards_per_play",
        "home_qb_hits_for", "away_qb_hits_for",
        "home_qb_hits_allowed", "away_qb_hits_allowed",
        "home_tackles_for_loss", "away_tackles_for_loss",
        "home_passes_defended", "away_passes_defended",
        "home_defensive_tds", "away_defensive_tds",
        "home_completion_pct", "away_completion_pct",
        "home_yards_per_carry", "away_yards_per_carry",
        "home_int_return_yards", "away_int_return_yards",
        "home_two_pt_conversions", "away_two_pt_conversions",
        "spread_line", "total_line",
        "home_moneyline", "away_moneyline",
        "market_home_prob", "market_away_prob",
    ]
    
    return labels[[c for c in keep if c in labels.columns]]

def build_matchups(team_pregame: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    home_feat = team_pregame.add_prefix("home_").rename(
        columns={"home_season":"season","home_week":"week","home_team":"home_team"})
    away_feat = team_pregame.add_prefix("away_").rename(
        columns={"away_season":"season","away_week":"week","away_team":"away_team"})

    tab = labels.merge(home_feat, on=["season","week","home_team"], how="left") \
                .merge(away_feat, on=["season","week","away_team"], how="left")

    # build diffs for rolling stats
    home_cols = [c for c in tab.columns
                 if c.startswith("home_") and c.endswith(("_r3","_r5","_r8","_r10","_exp"))]
    for hc in home_cols:
        bn = hc[len("home_"):]     # e.g., passing_yards_r8
        ac = f"away_{bn}"
        if ac in tab.columns:
            tab[f"diff_{bn}"] = tab[hc] - tab[ac]
    
    return tab

def main():
    games      = pd.read_parquet(RAW_DIR / "schedules.parquet").sort_values(["season","week"])
    team_week  = pd.read_parquet(RAW_DIR / "team_week.parquet")

    base = build_base(team_week, games)
    base.to_parquet(FEAT_DIR / "base.parquet", index=False)

    final_stats = [
        "points", "yards", "passing_yards", "rushing_yards", 
        "sacks_for", "sacks_allowed",
        "first_downs", "penalties", "penalty_yards", 
        "fg_made", "fg_att",
        "turnovers", "turnovers_forced", "yards_per_play",
        "qb_hits_for", "qb_hits_allowed",           # QB pressure
        "tackles_for_loss",                          # Defensive penetration
        "passes_defended",                           # Pass defense
        "defensive_tds",                             # Defensive scoring
        "completion_pct",                            # QB accuracy
        "yards_per_carry",                           # Rush efficiency
        "int_return_yards",                          # Big play defense
        "two_pt_conversions",                        # Aggression
    ]
    roll_cols   = [c for c in final_stats if c in base.columns]

    team_pregame = add_rolling_means(base, roll_cols, windows=(3,5,8,10), ewm_halflife=3, use_ewm=True)
    rolled = [c for c in team_pregame.columns if c.endswith(("_r3","_r5","_r8","_r10","_exp"))]
    team_pregame = team_pregame[["season","week","team"] + rolled]
    team_pregame.to_parquet(FEAT_DIR / "team_pregame.parquet", index=False)

    labels = build_labels(games, base)
    labels.to_parquet(FEAT_DIR / "labels.parquet", index=False)

    matchups = build_matchups(team_pregame, labels)
    matchups.to_parquet(FEAT_DIR / "matchups.parquet", index=False)

    print("Built base/team_pregame/labels/matchups under", FEAT_DIR)
    # quick sanity
    for c in ["spread_line","total_line","market_home_prob"]:
        print(f"[build_features] matchups has {c}? {c in matchups.columns}")

if __name__ == "__main__":
    main()
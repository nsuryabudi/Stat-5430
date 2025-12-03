import nflreadpy as nfl
import polars as pl
from io_utils import save_raw

from settings import SEASONS

def main():
    # Play-by-play (optional: big)
    pbp: pl.DataFrame = nfl.load_pbp(SEASONS)
    save_raw(pbp.to_pandas(), "pbp")

    # Team weekly stats
    team_week: pl.DataFrame = nfl.load_team_stats(SEASONS)
    save_raw(team_week.to_pandas(), "team_week")

    # Schedules
    schedules: pl.DataFrame = nfl.load_schedules(SEASONS)
    save_raw(schedules.to_pandas(), "schedules")

    print("Saved raw nflverse tables as Parquet (+ CSV).")

if __name__ == "__main__":
    main()
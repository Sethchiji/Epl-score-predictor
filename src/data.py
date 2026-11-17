"""Load Premier League match results and betting odds from football-data.co.uk.

Each season is one CSV file, e.g. 2025/26 -> https://www.football-data.co.uk/mmz4281/2526/E0.csv
"""
from __future__ import annotations

import os
from datetime import date

import pandas as pd

# You can point this at a different location (e.g. a local folder) with an env variable.
BASE_URL = os.environ.get("FOOTBALL_DATA_BASE_URL", "https://www.football-data.co.uk/mmz4281")
LEAGUE = "E0"  # E0 = English Premier League

BIG_SIX = ["Arsenal", "Chelsea", "Liverpool", "Man City", "Man United", "Tottenham"]

# Columns we keep. Odds columns are optional (not every season has every bookmaker).
CORE_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
ODDS_PREFIXES = ["PS", "B365", "Avg"]  # Pinnacle, Bet365, market average (match result)
# Over/under 2.5 goals odds: Pinnacle uses the prefix "P" here (not "PS")
OU_COLS = ["P>2.5", "P<2.5", "B365>2.5", "B365<2.5", "Avg>2.5", "Avg<2.5"]


def season_code(start_year: int) -> str:
    """2025 -> '2526' (the 2025/26 season)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def current_season_start(today: date | None = None) -> int:
    """The EPL season starts in August, so Jan-Jul belong to the season that began last year."""
    today = today or date.today()
    return today.year if today.month >= 8 else today.year - 1


def load_season(start_year: int) -> pd.DataFrame:
    """Download one season and return clean, played matches only."""
    code = season_code(start_year)
    url = f"{BASE_URL}/{code}/{LEAGUE}.csv"
    raw = pd.read_csv(url, encoding="latin1", on_bad_lines="skip")

    odds_cols = [f"{p}{o}" for p in ODDS_PREFIXES for o in "HDA" if f"{p}{o}" in raw.columns]
    odds_cols += [c for c in OU_COLS if c in raw.columns]
    df = raw[[c for c in CORE_COLS if c in raw.columns] + odds_cols].copy()

    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])  # drop blank / unplayed rows
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")  # dd/mm/yy or dd/mm/yyyy
    df = df.dropna(subset=["Date"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df["HomeTeam"] = df["HomeTeam"].str.strip()
    df["AwayTeam"] = df["AwayTeam"].str.strip()
    df["season"] = code
    return df.reset_index(drop=True)


def load_seasons(start_years: list[int]) -> pd.DataFrame:
    """Load several seasons and stack them, oldest first. Seasons that fail to download are skipped."""
    frames = []
    for y in start_years:
        try:
            frames.append(load_season(y))
        except Exception as exc:  # e.g. season file not published yet
            print(f"Skipping {season_code(y)}: {exc}")
    if not frames:
        raise RuntimeError("No seasons could be loaded. Check your internet connection or season years.")
    return pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)


def load_recent(n_seasons: int = 4, today: date | None = None) -> pd.DataFrame:
    """The current season plus the previous (n_seasons - 1) seasons."""
    current = current_season_start(today)
    return load_seasons(list(range(current - n_seasons + 1, current + 1)))


# ---------------------------------------------------------------- upcoming fixtures
FIXTURES_URL = os.environ.get("FOOTBALL_DATA_FIXTURES_URL", "https://www.football-data.co.uk/fixtures.csv")


def load_fixtures() -> pd.DataFrame:
    """Upcoming Premier League fixtures (the file only covers the next few days of matches).

    The file starts with a hidden BOM character, so we read it as 'utf-8-sig' to keep the
    first column named 'Div' (with latin1 it becomes 'ï»¿Div').
    """
    fx = pd.read_csv(FIXTURES_URL, encoding="utf-8-sig", on_bad_lines="skip")
    fx.columns = [c.strip().lstrip("﻿") for c in fx.columns]
    if "Div" not in fx.columns:
        raise ValueError(f"Unexpected fixtures file format. Columns: {fx.columns.tolist()[:8]}")
    fx = fx[fx["Div"] == LEAGUE].copy()
    keep = ["Date", "Time", "HomeTeam", "AwayTeam"] + [
        c for c in ["B365H", "B365D", "B365A", "B365>2.5", "B365<2.5"] if c in fx.columns]
    fx = fx[[c for c in keep if c in fx.columns]]
    fx["Date"] = pd.to_datetime(fx["Date"], dayfirst=True, errors="coerce")
    fx["HomeTeam"] = fx["HomeTeam"].str.strip()
    fx["AwayTeam"] = fx["AwayTeam"].str.strip()
    return fx.dropna(subset=["Date"]).sort_values(["Date", "Time" if "Time" in fx.columns else "Date"]).reset_index(drop=True)

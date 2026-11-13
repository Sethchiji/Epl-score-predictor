"""Honest evaluation: walk forward through a season, refitting before each matchweek,
and compare the model with simple baselines and with the bookmakers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import DEFAULT_XI, fit_poisson

OUTCOMES = ["H", "D", "A"]


def implied_probs(df: pd.DataFrame, prefix: str) -> pd.DataFrame | None:
    """Turn decimal odds into probabilities and remove the bookmaker's margin (overround)."""
    cols = [f"{prefix}{o}" for o in "HDA"]
    if not all(c in df.columns for c in cols):
        return None
    inv = 1.0 / df[cols].astype(float)
    probs = inv.div(inv.sum(axis=1), axis=0)
    probs.columns = ["p_home", "p_draw", "p_away"]
    return probs


def _metrics(probs: pd.DataFrame, results: pd.Series) -> dict:
    """Log loss (lower = better), Brier score (lower = better), accuracy of the top pick."""
    P = probs[["p_home", "p_draw", "p_away"]].to_numpy()
    y = np.zeros_like(P)
    idx = results.map({"H": 0, "D": 1, "A": 2}).to_numpy()
    y[np.arange(len(idx)), idx] = 1
    eps = 1e-12
    return {
        "log_loss": float(-np.mean(np.log(np.clip((P * y).sum(axis=1), eps, 1)))),
        "brier": float(np.mean(((P - y) ** 2).sum(axis=1))),
        "accuracy": float(np.mean(P.argmax(axis=1) == idx)),
        "n_matches": int(len(idx)),
    }


def backtest(matches: pd.DataFrame, test_season: str, xi: float = DEFAULT_XI,
             refit_every_days: int = 7) -> pd.DataFrame:
    """Walk-forward test: for each week of `test_season`, fit only on earlier matches, then predict that week.

    Returns one row per predicted match with the model's probabilities and the actual result.
    """
    test = matches[matches["season"] == test_season].sort_values("Date")
    if test.empty:
        raise ValueError(f"No matches found for season {test_season}")

    rows = []
    start = test["Date"].min()
    end = test["Date"].max()
    window_start = start
    while window_start <= end:
        window_end = window_start + pd.Timedelta(days=refit_every_days)
        week = test[(test["Date"] >= window_start) & (test["Date"] < window_end)]
        if not week.empty:
            model = fit_poisson(matches, as_of=window_start, xi=xi)
            for _, m in week.iterrows():
                if m["HomeTeam"] not in model.teams or m["AwayTeam"] not in model.teams:
                    continue  # e.g. a promoted team with no prior data in the window
                pred = model.predict_match(m["HomeTeam"], m["AwayTeam"])
                pred.update({
                    "Date": m["Date"], "FTHG": m["FTHG"], "FTAG": m["FTAG"], "FTR": m["FTR"],
                    "actual_score": f"{m['FTHG']}-{m['FTAG']}",
                })
                for col in m.index:
                    if col[:-1] in ("PS", "B365", "Avg") and col[-1] in "HDA":
                        pred[col] = m[col]
                rows.append(pred)
        window_start = window_end
    return pd.DataFrame(rows)


def compare(bt: pd.DataFrame, train_matches: pd.DataFrame | None = None) -> pd.DataFrame:
    """Score the model against baselines and bookmakers on the same matches."""
    out = {}
    out["Poisson model"] = _metrics(bt, bt["FTR"])

    # Baseline 1: always the same probabilities = historical home/draw/away rates
    if train_matches is not None and not train_matches.empty:
        rates = train_matches["FTR"].value_counts(normalize=True).reindex(OUTCOMES).fillna(0)
    else:
        rates = pd.Series({"H": 0.45, "D": 0.25, "A": 0.30})
    base = pd.DataFrame({"p_home": rates["H"], "p_draw": rates["D"], "p_away": rates["A"]}, index=bt.index)
    out["Baseline (league avg rates)"] = _metrics(base, bt["FTR"])

    # Bookmakers (the real benchmark)
    for prefix, name in [("PS", "Pinnacle odds"), ("B365", "Bet365 odds"), ("Avg", "Market average odds")]:
        probs = implied_probs(bt, prefix)
        if probs is not None:
            ok = probs.notna().all(axis=1)
            if ok.any():
                out[name] = _metrics(probs[ok], bt.loc[ok, "FTR"])

    table = pd.DataFrame(out).T
    table["exact_score_hit_rate"] = np.nan
    table.loc["Poisson model", "exact_score_hit_rate"] = float((bt["most_likely_score"] == bt["actual_score"]).mean())
    return table.round(4)

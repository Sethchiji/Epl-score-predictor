"""Poisson goals model for football score prediction.

Idea: each team has an ATTACK strength and a DEFENSE strength, plus there is a HOME advantage.

    expected home goals = exp(intercept + home + attack[home_team] + defense[away_team])
    expected away goals = exp(intercept +        attack[away_team] + defense[home_team])

Goals are then treated as Poisson-distributed, which gives a probability for every scoreline.
Recent matches count more than old ones (time decay), so the model tracks current form.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson

# Dixon & Coles (1997) used ~0.0065 per half-week, which is ~0.0019 per day.
# A match 1 year old gets weight exp(-0.0019 * 365) ~ 0.50.
DEFAULT_XI = 0.0019


def to_long(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match: who scored, against whom, and whether at home."""
    home = pd.DataFrame({
        "team": matches["HomeTeam"], "opponent": matches["AwayTeam"],
        "goals": matches["FTHG"], "home": 1, "Date": matches["Date"],
    })
    away = pd.DataFrame({
        "team": matches["AwayTeam"], "opponent": matches["HomeTeam"],
        "goals": matches["FTAG"], "home": 0, "Date": matches["Date"],
    })
    return pd.concat([home, away], ignore_index=True)


@dataclass
class PoissonModel:
    result: object          # fitted statsmodels GLM
    teams: list[str]        # teams the model knows
    as_of: pd.Timestamp     # date of the most recent match used
    xi: float

    # ---------- predictions ----------
    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        for t in (home_team, away_team):
            if t not in self.teams:
                raise ValueError(f"'{t}' is not in the training data. Known teams: {', '.join(self.teams)}")
        X = pd.DataFrame({
            "team": [home_team, away_team], "opponent": [away_team, home_team], "home": [1, 0],
        })
        mu = self.result.predict(X)
        return float(mu.iloc[0]), float(mu.iloc[1])

    def score_matrix(self, home_team: str, away_team: str, max_goals: int = 10) -> pd.DataFrame:
        """P(home scores i, away scores j). Rows = home goals, columns = away goals."""
        lam_h, lam_a = self.expected_goals(home_team, away_team)
        goals = np.arange(max_goals + 1)
        m = np.outer(poisson.pmf(goals, lam_h), poisson.pmf(goals, lam_a))
        return pd.DataFrame(m, index=goals, columns=goals)

    def predict_match(self, home_team: str, away_team: str) -> dict:
        m = self.score_matrix(home_team, away_team).values
        lam_h, lam_a = self.expected_goals(home_team, away_team)
        i, j = np.unravel_index(m.argmax(), m.shape)
        return {
            "home_team": home_team,
            "away_team": away_team,
            "exp_home_goals": lam_h,
            "exp_away_goals": lam_a,
            "p_home": float(np.tril(m, -1).sum()),   # home goals > away goals
            "p_draw": float(np.trace(m)),
            "p_away": float(np.triu(m, 1).sum()),
            "most_likely_score": f"{i}-{j}",
            "p_most_likely_score": float(m[i, j]),
        }

    def predict_fixtures(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """fixtures needs HomeTeam and AwayTeam columns."""
        rows = [self.predict_match(h, a) for h, a in zip(fixtures["HomeTeam"], fixtures["AwayTeam"])]
        return pd.DataFrame(rows)

    # ---------- interpretation ----------
    def team_ratings(self) -> pd.DataFrame:
        """Attack: higher = scores more. Defense: LOWER = concedes less (better).

        Values are multipliers vs. an average team (1.00). Attack 1.30 = scores 30% more than average.
        """
        p = self.result.params
        att = {t: p.get(f"team[T.{t}]", 0.0) for t in self.teams}
        dfn = {t: p.get(f"opponent[T.{t}]", 0.0) for t in self.teams}
        att = pd.Series(att); dfn = pd.Series(dfn)
        att -= att.mean(); dfn -= dfn.mean()   # centre so the average team = 0 on the log scale
        out = pd.DataFrame({"attack": np.exp(att), "defense": np.exp(dfn)})
        out["net_rating"] = out["attack"] / out["defense"]
        return out.sort_values("net_rating", ascending=False).round(3)

    @property
    def home_advantage(self) -> float:
        """Multiplier on goals for playing at home (e.g. 1.25 = 25% more goals)."""
        return float(np.exp(self.result.params["home"]))


def fit_poisson(matches: pd.DataFrame, as_of: pd.Timestamp | None = None, xi: float = DEFAULT_XI) -> PoissonModel:
    """Fit the model on all matches played strictly before `as_of` (default: all matches)."""
    if as_of is not None:
        matches = matches[matches["Date"] < as_of]
    if matches.empty:
        raise ValueError("No matches to train on.")
    ref_date = pd.Timestamp(as_of) if as_of is not None else matches["Date"].max()

    long = to_long(matches)
    days_ago = (ref_date - long["Date"]).dt.days.clip(lower=0)
    weights = np.exp(-xi * days_ago)

    result = smf.glm(
        "goals ~ home + team + opponent",
        data=long,
        family=sm.families.Poisson(),
        var_weights=weights,
    ).fit()

    teams = sorted(set(long["team"]))
    return PoissonModel(result=result, teams=teams, as_of=matches["Date"].max(), xi=xi)

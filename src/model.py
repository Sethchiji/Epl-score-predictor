"""Poisson goals model for football score prediction, with shrinkage.

Idea: each team has an ATTACK strength and a DEFENSE strength, plus there is a HOME advantage.

    expected home goals = exp(intercept + home + attack[home_team] + defense[away_team])
    expected away goals = exp(intercept +        attack[away_team] + defense[home_team])

Goals are treated as Poisson-distributed, which gives a probability for every scoreline.

Two things keep the ratings realistic:
- Time decay: recent matches count more than old ones (weight = exp(-xi * days_ago)).
- Shrinkage: a ridge (L2) penalty pulls every rating toward the league average. Teams with lots
  of data barely move; teams with only a few matches (e.g. newly promoted) stay near average
  instead of getting extreme ratings. Without it, a team that kept a clean sheet in its only
  match would be rated as impossible to score against.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

# Dixon & Coles (1997) used ~0.0065 per half-week, which is ~0.0019 per day.
# A match 1 year old gets weight exp(-0.0019 * 365) ~ 0.50.
DEFAULT_XI = 0.0019
# Ridge penalty strength. Roughly: how many "average matches" of evidence a team starts with.
# Tune it with the backtest (see notebook 01, section 7).
DEFAULT_SHRINKAGE = 5.0


@dataclass
class PoissonModel:
    teams: list[str]
    intercept: float
    home: float
    attack: pd.Series     # log scale, index = team
    defense: pd.Series    # log scale, index = team (higher = concedes more)
    as_of: pd.Timestamp   # date of the most recent match used
    xi: float
    shrinkage: float

    # ---------- predictions ----------
    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        for t in (home_team, away_team):
            if t not in self.attack.index:
                raise ValueError(f"'{t}' is not in the training data. Known teams: {', '.join(self.teams)}")
        lam_h = np.exp(self.intercept + self.home + self.attack[home_team] + self.defense[away_team])
        lam_a = np.exp(self.intercept + self.attack[away_team] + self.defense[home_team])
        return float(lam_h), float(lam_a)

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
        att = self.attack - self.attack.mean()
        dfn = self.defense - self.defense.mean()
        out = pd.DataFrame({"attack": np.exp(att), "defense": np.exp(dfn)})
        out["net_rating"] = out["attack"] / out["defense"]
        return out.sort_values("net_rating", ascending=False).round(3)

    @property
    def home_advantage(self) -> float:
        """Multiplier on goals for playing at home (e.g. 1.25 = 25% more goals)."""
        return float(np.exp(self.home))


def fit_poisson(matches: pd.DataFrame, as_of: pd.Timestamp | None = None,
                xi: float = DEFAULT_XI, shrinkage: float = DEFAULT_SHRINKAGE) -> PoissonModel:
    """Fit on all matches played strictly before `as_of` (default: all matches).

    Maximises the time-weighted Poisson log-likelihood minus a ridge penalty
    0.5 * shrinkage * (sum(attack^2) + sum(defense^2)).
    """
    if as_of is not None:
        matches = matches[matches["Date"] < as_of]
    if matches.empty:
        raise ValueError("No matches to train on.")
    if shrinkage <= 0:
        raise ValueError("shrinkage must be > 0 (it also keeps the ratings identifiable).")
    ref_date = pd.Timestamp(as_of) if as_of is not None else matches["Date"].max()

    teams = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))
    pos = {t: k for k, t in enumerate(teams)}
    n = len(teams)
    hi = matches["HomeTeam"].map(pos).to_numpy()
    ai = matches["AwayTeam"].map(pos).to_numpy()
    hg = matches["FTHG"].to_numpy(dtype=float)
    ag = matches["FTAG"].to_numpy(dtype=float)
    w = np.exp(-xi * (ref_date - matches["Date"]).dt.days.clip(lower=0).to_numpy())

    def objective(theta):
        c, h = theta[0], theta[1]
        att, dfn = theta[2:2 + n], theta[2 + n:]
        eta_h = c + h + att[hi] + dfn[ai]
        eta_a = c + att[ai] + dfn[hi]
        mu_h, mu_a = np.exp(eta_h), np.exp(eta_a)
        # negative log-likelihood (constant log(y!) terms dropped) + ridge penalty
        loss = np.sum(w * (mu_h - hg * eta_h)) + np.sum(w * (mu_a - ag * eta_a))
        loss += 0.5 * shrinkage * (att @ att + dfn @ dfn)
        # gradient
        rh, ra = w * (mu_h - hg), w * (mu_a - ag)
        g_att = np.bincount(hi, rh, n) + np.bincount(ai, ra, n) + shrinkage * att
        g_dfn = np.bincount(ai, rh, n) + np.bincount(hi, ra, n) + shrinkage * dfn
        grad = np.concatenate([[rh.sum() + ra.sum(), rh.sum()], g_att, g_dfn])
        return loss, grad

    theta0 = np.zeros(2 + 2 * n)
    theta0[0] = np.log(max((hg.mean() + ag.mean()) / 2, 0.1))
    res = minimize(objective, theta0, jac=True, method="L-BFGS-B")
    if not res.success:
        print(f"Warning: optimiser did not fully converge ({res.message})")

    th = res.x
    return PoissonModel(
        teams=teams, intercept=float(th[0]), home=float(th[1]),
        attack=pd.Series(th[2:2 + n], index=teams), defense=pd.Series(th[2 + n:], index=teams),
        as_of=matches["Date"].max(), xi=xi, shrinkage=shrinkage,
    )

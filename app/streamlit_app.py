"""EPL Score Predictor: Streamlit app (Phase 2 version: reads data straight from football-data.co.uk).

In a later phase this will read predictions from BigQuery instead.
Run locally/Colab:  streamlit run app/streamlit_app.py
"""
import sys
from itertools import permutations
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))  # so "src" can be imported
from src.data import BIG_SIX, load_fixtures, load_recent  # noqa: E402
from src.model import fit_poisson  # noqa: E402

st.set_page_config(page_title="EPL Score Predictor", page_icon="⚽", layout="wide")


@st.cache_data(ttl=6 * 60 * 60, show_spinner="Downloading match data...")
def get_matches(n_seasons: int) -> pd.DataFrame:
    return load_recent(n_seasons)


@st.cache_data(ttl=3 * 60 * 60, show_spinner="Checking upcoming fixtures...")
def get_fixtures() -> pd.DataFrame:
    return load_fixtures()


@st.cache_resource(show_spinner="Fitting the model...")
def get_model(n_seasons: int, xi: float, shrinkage: float):
    return fit_poisson(get_matches(n_seasons), xi=xi, shrinkage=shrinkage)


st.title("⚽ Premier League Score Predictor")
st.caption("Poisson goals model with time-weighted, shrinkage-regularised team attack/defense strengths. "
           "Trained on all Premier League matches, focused on the Big Six.")

with st.sidebar:
    st.header("Model settings")
    n_seasons = st.slider("Seasons of history", 2, 6, 4)
    half_life = st.slider("Form half-life (days)", 90, 730, 365, step=15,
                          help="How fast old matches lose influence. 365 = a match from a year ago counts half.")
    xi = 0.6931 / half_life
    shrinkage = st.slider("Shrinkage", 0.5, 30.0, 5.0, step=0.5,
                          help="Pulls teams with little data (e.g. newly promoted) toward average. Higher = more cautious.")

matches = get_matches(n_seasons)
model = get_model(n_seasons, xi, shrinkage)
st.sidebar.write(f"Trained on **{len(matches):,}** matches up to **{model.as_of:%d %b %Y}**")
st.sidebar.write(f"Home advantage: **{model.home_advantage:.2f}×** goals")

big_six = [t for t in BIG_SIX if t in model.teams]
tab0, tab1, tab2, tab3 = st.tabs(["Upcoming fixtures", "Big Six head-to-head", "Match predictor", "Team ratings"])

with tab0:
    st.subheader("Next Premier League fixtures")
    try:
        fx = get_fixtures()
    except Exception as exc:  # site down or format changed: don't break the rest of the app
        fx = None
        st.warning(f"Couldn't load the fixtures file right now ({exc}). The other tabs still work.")
    if fx is not None and fx.empty:
        st.info("No Premier League fixtures in the upcoming-fixtures file yet. It only covers the next few days "
                "of matches, so the next round appears shortly before matchday. Meanwhile, use the "
                "**Match predictor** tab for any matchup.")
    elif fx is not None:
        known = fx["HomeTeam"].isin(model.teams) & fx["AwayTeam"].isin(model.teams)
        preds = model.predict_fixtures(fx[known]).set_index(fx[known].index)
        table = pd.concat([fx[known][["Date"] + (["Time"] if "Time" in fx.columns else [])], preds], axis=1)
        if {"B365H", "B365D", "B365A"}.issubset(fx.columns):
            inv = 1 / fx.loc[known, ["B365H", "B365D", "B365A"]].astype(float)
            table["bookie_home"] = (inv["B365H"] / inv.sum(axis=1)).values
        if {"B365>2.5", "B365<2.5"}.issubset(fx.columns):
            inv_o = 1 / fx.loc[known, "B365>2.5"].astype(float)
            inv_u = 1 / fx.loc[known, "B365<2.5"].astype(float)
            table["bookie_over"] = (inv_o / (inv_o + inv_u)).values
        table["big_six"] = table["home_team"].isin(BIG_SIX) | table["away_team"].isin(BIG_SIX)
        only_big6 = st.toggle("Big Six matches only", value=False)
        if only_big6:
            table = table[table["big_six"]]
        cols = ["Date"] + (["Time"] if "Time" in table.columns else []) + [
            "home_team", "away_team", "p_home", "p_draw", "p_away", "p_over_2_5", "p_btts", "most_likely_score"]
        labels = {"home_team": "Home", "away_team": "Away", "p_home": "Home win", "p_draw": "Draw",
                  "p_away": "Away win", "p_over_2_5": "Over 2.5", "p_btts": "BTTS",
                  "most_likely_score": "Most likely score"}
        for extra in ("bookie_home", "bookie_over"):
            if extra in table.columns:
                cols.append(extra)
        fmt = {c: "{:.0%}" for c in ["p_home", "p_draw", "p_away", "p_over_2_5", "p_btts",
                                     "bookie_home", "bookie_over"] if c in cols}
        fmt["Date"] = lambda d: d.strftime("%a %d %b")
        st.dataframe(
            table[cols].style.format(fmt).apply(
                lambda r: ["font-weight: bold" if table.loc[r.name, "big_six"] else "" for _ in r], axis=1),
            width="stretch", hide_index=True,
            column_config={**{k: st.column_config.Column(v) for k, v in labels.items()},
                           "bookie_home": st.column_config.Column("Bet365 home %",
                           help="Bet365's home-win probability (margin removed), for comparison"),
                           "bookie_over": st.column_config.Column("Bet365 over 2.5 %",
                           help="Bet365's over 2.5 goals probability (margin removed), for comparison")},
        )
        st.caption("**Bold** = Big Six involved. Probabilities are model predictions; "
                   "the Bet365 columns show the bookmaker's implied chances (margin removed) for comparison.")
        if (~known).any():
            st.caption("Skipped (team not in training data): " +
                       ", ".join(f"{h} vs {a}" for h, a in zip(fx.loc[~known, "HomeTeam"], fx.loc[~known, "AwayTeam"])))

with tab1:
    st.subheader("Every Big Six fixture, predicted")
    fixtures = pd.DataFrame(list(permutations(big_six, 2)), columns=["HomeTeam", "AwayTeam"])
    preds = model.predict_fixtures(fixtures)
    show = preds[["home_team", "away_team", "p_home", "p_draw", "p_away", "p_over_2_5", "p_btts",
                  "exp_home_goals", "exp_away_goals", "most_likely_score"]].copy()
    show.columns = ["Home", "Away", "Home win", "Draw", "Away win", "Over 2.5", "BTTS",
                    "Exp. home goals", "Exp. away goals", "Most likely score"]
    st.dataframe(
        show.style.format({c: "{:.0%}" for c in ["Home win", "Draw", "Away win", "Over 2.5", "BTTS"]}
                          | {c: "{:.2f}" for c in ["Exp. home goals", "Exp. away goals"]}),
        width="stretch", hide_index=True,
    )
    pivot = preds.pivot(index="home_team", columns="away_team", values="p_home")
    fig = px.imshow(pivot, text_auto=".0%", color_continuous_scale="Blues", aspect="auto",
                    labels={"x": "Away team", "y": "Home team", "color": "P(home win)"})
    st.plotly_chart(fig, width="stretch")

with tab2:
    st.subheader("Predict any match")
    c1, c2 = st.columns(2)
    default_home = model.teams.index(big_six[0]) if big_six else 0
    home = c1.selectbox("Home team", model.teams, index=default_home)
    away_options = [t for t in model.teams if t != home]
    away = c2.selectbox("Away team", away_options,
                        index=away_options.index(big_six[1]) if len(big_six) > 1 and big_six[1] in away_options else 0)
    p = model.predict_match(home, away)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{home} win", f"{p['p_home']:.0%}")
    m2.metric("Draw", f"{p['p_draw']:.0%}")
    m3.metric(f"{away} win", f"{p['p_away']:.0%}")
    m4.metric("Most likely score", p["most_likely_score"], f"{p['p_most_likely_score']:.0%} chance", delta_color="off")

    mat = model.score_matrix(home, away, max_goals=5)
    fig = px.imshow(mat, text_auto=".1%", color_continuous_scale="Greens", aspect="auto",
                    labels={"x": f"{away} goals", "y": f"{home} goals", "color": "Probability"})
    st.plotly_chart(fig, width="stretch")

    st.markdown("##### Goals markets")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Both teams score", f"{p['p_btts']:.0%}")
    g2.metric("Over 2.5 goals", f"{p['p_over_2_5']:.0%}")
    g3.metric("Expected total goals", f"{p['exp_total_goals']:.2f}")
    g4.metric("Double chance (home or draw)", f"{p['p_home_or_draw']:.0%}")
    markets = pd.DataFrame({
        "Market": ["Over 1.5 goals", "Over 2.5 goals", "Over 3.5 goals", "Both teams to score",
                   f"{home} to score", f"{home} 2+ goals", f"{home} clean sheet",
                   f"{away} to score", f"{away} 2+ goals", f"{away} clean sheet",
                   f"{home} or draw", f"{away} or draw"],
        "Probability": [p["p_over_1_5"], p["p_over_2_5"], p["p_over_3_5"], p["p_btts"],
                        p["p_home_over_0_5"], p["p_home_over_1_5"], p["p_home_clean_sheet"],
                        p["p_away_over_0_5"], p["p_away_over_1_5"], p["p_away_clean_sheet"],
                        p["p_home_or_draw"], p["p_away_or_draw"]],
    })
    markets["Fair odds"] = 1 / markets["Probability"]
    st.dataframe(markets.style.format({"Probability": "{:.0%}", "Fair odds": "{:.2f}"}),
                 width="stretch", hide_index=True)
    st.caption("Fair odds = 1 / probability, with no bookmaker margin. These are model estimates, not betting advice: "
               "see the README for how the model compares with bookmaker odds.")

with tab3:
    st.subheader("Team strength ratings")
    st.caption("Attack: goals scored vs. an average team (higher is better). "
               "Defense: goals conceded vs. average (lower is better).")
    ratings = model.team_ratings()
    st.dataframe(ratings.rename(columns={"attack": "Attack", "defense": "Defense", "net_rating": "Net rating"}),
                 width="stretch")
    fig = px.scatter(ratings.reset_index(names="team"), x="attack", y="defense", text="team",
                     labels={"attack": "Attack (higher = better)", "defense": "Defense (lower = better)"})
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, width="stretch")

st.caption("Data: football-data.co.uk · Model: time-weighted Poisson with ridge shrinkage")

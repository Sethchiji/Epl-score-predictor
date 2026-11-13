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
from src.data import BIG_SIX, load_recent  # noqa: E402
from src.model import fit_poisson  # noqa: E402

st.set_page_config(page_title="EPL Score Predictor", page_icon="⚽", layout="wide")


@st.cache_data(ttl=6 * 60 * 60, show_spinner="Downloading match data...")
def get_matches(n_seasons: int) -> pd.DataFrame:
    return load_recent(n_seasons)


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
tab1, tab2, tab3 = st.tabs(["Big Six head-to-head", "Match predictor", "Team ratings"])

with tab1:
    st.subheader("Every Big Six fixture, predicted")
    fixtures = pd.DataFrame(list(permutations(big_six, 2)), columns=["HomeTeam", "AwayTeam"])
    preds = model.predict_fixtures(fixtures)
    show = preds[["home_team", "away_team", "p_home", "p_draw", "p_away",
                  "exp_home_goals", "exp_away_goals", "most_likely_score"]].copy()
    st.dataframe(
        show.style.format({c: "{:.0%}" for c in ["p_home", "p_draw", "p_away"]}
                          | {c: "{:.2f}" for c in ["exp_home_goals", "exp_away_goals"]}),
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

with tab3:
    st.subheader("Team strength ratings")
    st.caption("Attack: goals scored vs. an average team (higher is better). "
               "Defense: goals conceded vs. average (lower is better).")
    ratings = model.team_ratings()
    st.dataframe(ratings, width="stretch")
    fig = px.scatter(ratings.reset_index(names="team"), x="attack", y="defense", text="team",
                     labels={"attack": "Attack (higher = better)", "defense": "Defense (lower = better)"})
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, width="stretch")

st.caption("Data: football-data.co.uk · Model: time-weighted Poisson with ridge shrinkage")

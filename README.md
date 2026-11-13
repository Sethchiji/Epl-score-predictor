# ⚽ EPL Score Predictor

Predicts Premier League scorelines and win/draw/loss probabilities with a **time-weighted Poisson goals model**, with a focus on the **Big Six** (Arsenal, Chelsea, Liverpool, Man City, Man United, Tottenham). Every prediction is checked with a walk-forward backtest against bookmaker odds.

> 🔗 **Live app:** _add your Streamlit link here after Phase 2_
> 📸 _Add a screenshot of the app here_

## How the model works

Every team gets an **attack** strength and a **defense** strength, and the league has one **home advantage**:

```
expected home goals = exp(intercept + home + attack[home] + defense[away])
expected away goals = exp(intercept +        attack[away] + defense[home])
```

Goals are modeled as Poisson-distributed, which gives a probability for every scoreline (0-0, 1-0, 2-1, ...). Summing those probabilities gives the home win, draw, and away win chances. Older matches are down-weighted with exponential time decay (Dixon & Coles, 1997), so the ratings follow current form.

**Why train on all 20 teams if the focus is the Big Six?** The Big Six play each other only 30 times a season, which is too few matches to fit a model. Their games against the other 14 teams are what reveal how strong they really are.

## Results

_Fill this in after the Phase 1 backtest. Example layout:_

| Model (season 20XX/XX, walk-forward) | Log loss ↓ | Brier ↓ | Accuracy |
|---|---|---|---|
| Baseline (league-average rates) | | | |
| **Poisson model** | | | |
| Pinnacle closing odds (benchmark) | | | |

Key insights:
1. ...
2. ...
3. ...

## Project structure

```
src/data.py            download & clean match data (football-data.co.uk)
src/model.py           Poisson model: fit, scoreline matrix, predictions, team ratings
src/evaluate.py        walk-forward backtest, log loss / Brier, bookmaker comparison
app/streamlit_app.py   interactive web app
notebooks/00_...       one-time setup: push this repo to GitHub from Colab
notebooks/01_...       explore data, fit the model, backtest
```

## Roadmap

- [ ] **Phase 1: Model (Colab).** Run `notebooks/01_first_model.ipynb`, backtest, and write the insights above
- [ ] **Phase 2: Deploy the app.** Streamlit Community Cloud gives you a live link
- [ ] **Phase 3: Cloud warehouse.** Load the raw match data into the BigQuery sandbox
- [ ] **Phase 4: dbt.** Build staging and mart models with data tests
- [ ] **Phase 5: Automation.** A GitHub Actions weekly job runs load → dbt → refit → write predictions
- [ ] **Phase 6: App v2.** The app reads predictions from BigQuery, with model-vs-bookmaker tracking over time
- [ ] **Stretch:** Dixon-Coles low-score correction, xG-based features, an expected-points league table simulation

## Run it

Everything runs in **Google Colab**, so nothing needs to be installed locally. Open a notebook from GitHub in Colab (File → Open notebook → GitHub).

App (Phase 2): deploy at [share.streamlit.io](https://share.streamlit.io) → New app → this repo → main file `app/streamlit_app.py`.

**Data:** [football-data.co.uk](https://www.football-data.co.uk/), free historical results and odds, used for educational purposes.

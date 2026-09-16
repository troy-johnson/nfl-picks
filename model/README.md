# Prediction model

V1 builds pregame rolling team-strength features from nflverse play-by-play, trains a logistic regression with chronological validation, then blends its probability with the de-vigged market probability using the prior season's Brier score to choose the blend weight.

## Historical backtest

Run a local expanding-window backtest over completed games:

```bash
uv run --project model python model/predict.py --backtest
```

The default tests the latest three completed seasons. Each week trains on earlier completed weeks only. It scores the statistical model, Elo, QB, combined Elo-QB, and opponent-adjusted team challengers, standalone Elo, recorded market odds, and selected market blends with accuracy, Brier score, and log loss. Results are written to `model/artifacts/backtest.json`.

Use a fixed period when comparing a model change:

```bash
uv run --project model python model/predict.py --backtest --from-season 2023 --to-season 2025
```

The historical nflverse moneyline may not match odds available at the original prediction time. Treat market results as a useful recorded-odds baseline, not proof of live betting value.

The Elo, QB, combined Elo-QB, and opponent-adjusted logistic models are backtest challengers. The QB challenger uses each team's primary passer from its latest completed game, not the target game's recorded starter. None is used for live picks until it improves the market-blended result on a held-out period.

## Experiment queue

`model/experiments.py` compares a registry of candidate models on identical games without changing `predict.py`. Each candidate is retrained every week on an expanding window. Hyperparameters, Platt calibration for tree models, and the market blend weight are all chosen on the latest completed prior season only.

```bash
uv run --project model python model/experiments.py --list
uv run --project model python model/experiments.py --from-season 2023 --to-season 2025
uv run --project model python model/experiments.py --candidates elo_qb,boosted_trees,random_forest
```

The first run builds features and caches them in `model/artifacts/features-v5-2021-<season>.pkl`. Besides the live features, the builder adds these Elo columns: `elo_diff_warm` and `elo_diff_mov` come from a second Elo walk that starts in 2010, lets postseason games update ratings, and (for `elo_diff_mov`) applies FiveThirtyEight's margin-of-victory multiplier with the margin capped at 24 points; `elo_diff_mov2` is the same walk with franchise aliases (`OAK` to `LV`, `SD` to `LAC`, `STL` to `LA`) and the signed winner-minus-loser gap in the multiplier; and the `elo_grid_k{k}_h{h}_r{r}` columns sweep K in (16, 20, 24), home advantage in (48, 65), and season carryover in (0.5, 0.67, 0.8) for the per-week tuned candidate. Pass `--refresh-features` after a data update. Results, including per-game probabilities, per-week hyperparameters, calibration bins, paired week-block differences against the baseline and the market, and per-season metrics (`bySeason` and `marketBySeason`), are written to `model/artifacts/experiments.json`. A summary table is printed, one row per candidate plus a per-season Brier line.

Default queue: `baseline`, `elo`, `qb`, `elo_qb`, `elo_injury`, `elo_warm`, `elo_mov`, `elo_mov_injury`, `elo_mov2`, `elo_mov2_injury`, `elo_tuned_injury`, `market_stack`, `regularized_logistic`, `boosted_trees`. `random_forest` is registered but opt-in; in the 2023-2025 replay it correlated 0.93 with boosted trees and scored worse. Before each target week, `elo_tuned_injury` picks the grid column whose pure Elo probability has the lowest log loss over all completed games before that week and uses it as `elo_diff_tuned`.

Injury features come from the nflverse weekly injury reports (`injuries_{season}.csv`). A row counts only when its `date_modified` UTC date is strictly before the game's `gameday`; a season file published without `date_modified` (the 2025 file) is accepted as the whole weekly report without a date check, because 99.7% of weighted rows in the 2021-2024 files are dated at least one day before gameday.

2023-2025 replay result: `elo`, `elo_qb`, `regularized_logistic`, `elo_injury`, and the Elo-v2 candidates improve the baseline (Brier 0.2203-0.2249 versus 0.2277); `elo_tuned_injury` (per-week-tuned signed-margin Elo plus injury load) is the best raw candidate at 0.22026 Brier and 0.63110 log loss, with `elo_mov2_injury` and `elo_mov_injury` within 0.00005 Brier of it. `market_stack` improves the baseline but stays within one standard error of the recorded market. Tree models are rejected because they raise accuracy but worsen Brier score and log loss. No candidate beats the recorded market (Brier 0.2103). Every candidate without market input blends to 100% market, except `elo_mov_injury`, `elo_mov2_injury`, and `elo_tuned_injury`, which take a 0.95 weight in a third of the weeks and blend to about 0.21037, within noise of the market. See [docs/nfl-game-outcome-models.md](../docs/nfl-game-outcome-models.md) for the full table.

To add a candidate, append a `Candidate(...)` to `CANDIDATES` in `model/experiments.py`. Give it a feature tuple, an estimator builder, an optional small hyperparameter grid, and `calibrate=True` for models with uncalibrated probabilities. Then run the tests, which check that no candidate uses outcome or target-starter columns.

The Wednesday run creates the weekly snapshot. The final run occurs 60-120 minutes before the first kickoff. It refreshes market, quarterback, weather, and team news data. Injury headlines are intentionally not assigned arbitrary point values; they are shown for review beside market movement before the first kickoff.

## Future odds snapshots

Create a The Odds API account and set `THE_ODDS_API_KEY` locally or as a GitHub Actions repository secret. The scheduled capture workflow checks every in-season day and records one snapshot 60-120 minutes before the first kickoff.

```bash
THE_ODDS_API_KEY=... uv run --project model python model/predict.py --capture-odds
```

Snapshots are written to `data/odds-snapshots/<season>/<game-day>.json`. They preserve each available US sportsbook's de-vigged moneyline probability and their equal-weight average. Recording does not change live picks.

## Crowd pick shares

`model/crowd.py` records the public pick shares from ESPN's NFL Pick'em game. The shares tell how the pool field picks each game. They do not change live picks.

```bash
uv run --project model python model/crowd.py --archive --from-season 2021 --to-season 2025
uv run --project model python model/crowd.py --capture
uv run --project model python model/crowd.py --evaluate --from-season 2021 --to-season 2025
uv run --project model python model/crowd.py --report
```

- `--archive` downloads the final straight-up shares for past seasons into `data/crowd-picks/<season>/week-NN.json`.
- `--capture` runs in the capture workflow 0-120 minutes before the first kickoff of the next game day. Locked games keep their earlier shares. Open games take the fresh shares.
- `--evaluate` joins the archive with nflverse results and closing moneylines and writes `model/artifacts/crowd-evaluation.json`.
- `--report` reads `public/data/current.json` and lists, for each game, the market probability, the crowd share, and the contrarian value (market probability minus crowd share) of the less-picked team.

2021-2025 result (1,355 games): the crowd favorite wins 65.2% of games and the market favorite wins 66.5%. The crowd share is not a probability: used as one it scores Brier 0.2459 against 0.2117 for the market. The field is much more extreme than the market. When the market favorite sits at 60-70%, about 78-83% of entries pick it. Coin-flip games (market favorite below 55%) are where a pick against the field costs the least expected points.

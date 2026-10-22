# Prediction model

V1 builds pregame rolling team-strength features from nflverse play-by-play, trains a logistic regression with chronological validation, then blends its probability with the de-vigged market probability using the prior season's Brier score to choose the blend weight.

## Live market inputs

`scheduled_runs.py` calls `live_picks.py` for publication.
It uses the statistical functions from the frozen `predict.py` source.
`live_odds.py` supplies the current market input from The Odds API.

- Each bookmaker contributes its de-vigged moneyline probability with equal weight.
- A consensus requires two distinct bookmakers with prices updated within 30 minutes.
- The adapter rejects future timestamps, invalid prices, duplicate books, and mismatched kickoff times.
- Games within 180 minutes of kickoff must have fresh consensus odds.
- Later games can use labeled schedule odds with unknown age when consensus is unavailable.
- Missing current-season play-by-play for completed prior weeks stops publication.
- Network calls and validation finish before the published file changes.
- Started games retain their previous predictions and timestamps.

The refresh workflow supplies `THE_ODDS_API_KEY` from the repository secret.
For a local run, export that variable before running `scheduled_runs.py`.
The site shows each game's market source, bookmaker count, price timestamp, and age at generation.
The displayed spread still comes from the schedule and has its own label.

Each refresh archives its exact market inputs with its published forecasts.
Files use `data/odds-snapshots/<season>/<day>/<timestamp>.json` and never replace an earlier capture.
The Wednesday snapshot and later kickoff batches can therefore have separate records.
Operational changes do not require edits to the frozen statistical sources.

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

The first run builds features and caches them in `model/artifacts/features-v6-2021-<season>.pkl`. Besides the live features, the builder adds these Elo columns: `elo_diff_warm` and `elo_diff_mov` come from a second Elo walk that starts in 2010, lets postseason games update ratings, and (for `elo_diff_mov`) applies FiveThirtyEight's margin-of-victory multiplier with the margin capped at 24 points; `elo_diff_mov2` is the same walk with franchise aliases (`OAK` to `LV`, `SD` to `LAC`, `STL` to `LA`) and the signed winner-minus-loser gap in the multiplier; and the `elo_grid_k{k}_h{h}_r{r}` columns sweep K in (16, 20, 24), home advantage in (48, 65), and season carryover in (0.5, 0.67, 0.8) for the per-week tuned candidate. It also adds two market-prior columns: `market_rating_diff` is a spread-based power rating, fit per game day by weighted least squares to the closing spreads of completed games strictly before the target game's day (current season weight 1.0, previous season 0.5, ridge 0.1, no home-advantage term in the feature), and `elo_diff_market_seed` is the seeded Elo walk (K=16, home advantage 48, carryover 0.67, signed factor, aliases), whose season-start ratings regress toward 1500 + 25 x market rating fit from the previous season's spreads. Pass `--refresh-features` after a data update. Results, including per-game probabilities, per-week hyperparameters, calibration bins, paired week-block differences against the baseline, the market, and `elo_tuned_injury` (`rawVersusEloTunedInjury`), and per-season metrics (`bySeason` and `marketBySeason`), are written to `model/artifacts/experiments.json`. A summary table is printed, one row per candidate plus a per-season Brier line.

Default queue: `baseline`, `elo`, `qb`, `elo_qb`, `elo_injury`, `elo_warm`, `elo_mov`, `elo_mov_injury`, `elo_mov2`, `elo_mov2_injury`, `elo_tuned_injury`, `market_rating_injury`, `elo_seed_injury`, `elo_seed_market_rating_injury`, `market_stack`, `regularized_logistic`, `boosted_trees`. `random_forest` is registered but opt-in; in the 2023-2025 replay it correlated 0.93 with boosted trees and scored worse. Before each target week, `elo_tuned_injury` picks the grid column whose pure Elo probability has the lowest log loss over all completed games before that week and uses it as `elo_diff_tuned`.

Injury features come from the nflverse weekly injury reports (`injuries_{season}.csv`). A row counts only when its `date_modified` UTC date is strictly before the game's `gameday`; a season file published without `date_modified` (the 2025 file) is accepted as the whole weekly report without a date check, because 99.7% of weighted rows in the 2021-2024 files are dated at least one day before gameday.

2023-2025 replay result: `elo`, `elo_qb`, `regularized_logistic`, `elo_injury`, and the Elo-v2 candidates improve the baseline (Brier 0.2203-0.2249 versus 0.2277); `elo_tuned_injury` (per-week-tuned signed-margin Elo plus injury load) is the best raw candidate at 0.22026 Brier and 0.63110 log loss, with `elo_mov2_injury` and `elo_mov_injury` within 0.00005 Brier of it. The market-prior candidates (`market_rating_injury`, `elo_seed_injury`, `elo_seed_market_rating_injury`) improve the plain `elo` candidate but land 0.2211-0.2227 Brier, behind `elo_tuned_injury`, and their pooled deficits are under two standard errors. `market_stack` improves the baseline but stays within one standard error of the recorded market. Tree models are rejected because they raise accuracy but worsen Brier score and log loss. No candidate beats the recorded market (Brier 0.2103). Every candidate without market input blends to 100% market, except `elo_mov_injury`, `elo_mov2_injury`, `elo_tuned_injury`, and `market_rating_injury`, which take a 0.95 weight in a third of the weeks and blend to 0.21035-0.21038, within noise of the market. See [docs/nfl-game-outcome-models.md](../docs/nfl-game-outcome-models.md) for the full table.

To add a candidate, append a `Candidate(...)` to `CANDIDATES` in `model/experiments.py`. Give it a feature tuple, an estimator builder, an optional small hyperparameter grid, and `calibrate=True` for models with uncalibrated probabilities. Then run the tests, which check that no candidate uses outcome or target-starter columns.

The Wednesday run creates the weekly snapshot.
Final runs occur 45-180 minutes before each kickoff batch.
They refresh market, quarterback, weather, and team news data.
Injury headlines appear beside market movement for review.

## Data audit

The capture and refresh jobs exit 0 when they skip work. `model/audit.py` checks the saved files for each completed game day instead. It fails when a day lacks any of these:

- An odds capture for every game before kickoff.
- A pre-kickoff ESPN pick share for every regular-season game.
- A recency candidate forecast for every regular-season game after the freeze.
- A change to `public/data/current.json` in the 180 minutes before the first regular-season kickoff.

For days with the new capture format, every game needs odds within 180 minutes of its own kickoff.
Every regular-season game also needs a forecast paired with those odds.

```bash
uv run --project model python model/audit.py            # the whole season
uv run --project model python model/audit.py --days 2   # the last two days
```

The `Audit NFL data` workflow runs it daily with `--days 2`. A failed run is the alert. The refresh check reads the git history, so the workflow checks out all commits.

## Future odds snapshots

Set `THE_ODDS_API_KEY` locally and as a GitHub Actions repository secret.
The capture workflow records odds 45-180 minutes before the next kickoff that has not started.
A morning capture does not prevent another capture before afternoon or evening games.
If a morning window is missed, the workflow can still capture later games.

```bash
THE_ODDS_API_KEY=... uv run --project model python model/scheduled_runs.py --capture-odds
```

Captures use `data/odds-snapshots/<season>/<game-day>/<timestamp>.json`.
They preserve eligible US bookmakers' prices and the consensus probability.
Live generation includes a `published` block with forecasts that use that exact capture.
Independent capture jobs save market data without copying an older published forecast.
The reader also supports the earlier `<game-day>.json` archive.

### Same-time evaluation

`model/snapshots.py` selects each game's latest valid pregame paired forecast.
It joins that forecast to the result and recorded closing line.
It scores each source on identical games.

```bash
uv run --project model python model/snapshots.py
uv run --project model python model/snapshots.py --experiments model/artifacts/experiments.json
```

- Sources: `sameTimeMarket` (snapshot average), `closingMarket` (nflverse recorded line), `publishedPick` (blended probability users saw), `publishedStatistical` (raw model), and `candidate:<name>` for each challenger when `--experiments` points at an experiment queue output.
- Output `model/artifacts/snapshot-evaluation.json` holds per-source accuracy, Brier, log loss, and calibration; paired Brier and log-loss differences with week-block standard errors; per-season metrics; the same-time-versus-closing move; and every scored game.
- A comparison reports `status`. A market-beating claim needs at least 250 completed games (`--min-games`) and a difference beyond two standard errors on both Brier and log loss. `gamesToDetect` extrapolates from the observed week-block standard error how many games a 0.005 or 0.010 Brier difference would need.
- Unplayed games, ties, and captures without a valid paired forecast are excluded.
- New captures must identify the exact market capture used by the forecast.
- Earlier archives qualify only when publication preceded capture by at most five minutes.
- A later unpaired capture cannot replace an earlier paired forecast in the comparison.
- Captures saved at or after kickoff are excluded.

Until the archive holds a season of completed games, the report says `insufficient games`. That is the intended answer.

## Crowd pick shares

`model/crowd.py` records the public pick shares from ESPN's NFL Pick'em game.
The shares describe national entries, rather than this group's actual picks. They do not change live picks.

```bash
uv run --project model python model/crowd.py --archive --from-season 2021 --to-season 2025
uv run --project model python model/crowd.py --capture
uv run --project model python model/crowd.py --evaluate --from-season 2021 --to-season 2025
uv run --project model python model/crowd.py --report
```

- `--archive` downloads the final straight-up shares for past seasons into `data/crowd-picks/<season>/week-NN.json`.
- `--capture` runs in the capture workflow 0-180 minutes before the first kickoff of the next game day. Locked games keep their earlier shares. Open games take the fresh shares.
- `--evaluate` joins the archive with nflverse results and closing moneylines and writes `model/artifacts/crowd-evaluation.json`.
- `--report` reads `public/data/current.json` and lists, for each game, the market probability, the crowd share, and the contrarian value (market probability minus crowd share) of the less-picked team.

2021-2025 result (1,355 games): the crowd favorite wins 65.2% of games and the market favorite wins 66.5%. The crowd share is not a probability: used as one it scores Brier 0.2459 against 0.2117 for the market. The field is much more extreme than the market. When the market favorite sits at 60-70%, about 78-83% of entries pick it. Coin-flip games (market favorite below 55%) are where a pick against the field costs the least expected points.

### Historical pool challenger

The original pool challenger used national crowd shares to select occasional underdogs.
It remains available for historical experiments through `crowd.pool_pick`.
Live publication now sets `poolPick` to the standard `pick` for the accuracy objective.
It still displays crowd shares as context.

The rule (`crowd.pool_pick`) takes the underdog only when both conditions hold:

- the favorite's expected point edge `2p - 1` is at most `POOL_MAX_COST = 0.04` (market favorite at 52% or less), and
- at least `POOL_MIN_CROWD_SHARE = 0.60` of the field is on the favorite.

Simulate the rule grid on the archived seasons:

```bash
uv run --project model python model/crowd.py --simulate --from-season 2021 --to-season 2025
```

The simulation now defaults to nine entrants and excludes each entry's lowest-scoring week.
Use `--count-every-week` for the earlier scoring control.
Rules share the same simulated outcomes and opponents to reduce comparison noise.
Outcomes use recorded closing probabilities; simulated opponents use national crowd shares or always select the market favorite.
Ties split first-place credit. The simulation does not model ESPN tiebreak answers or individual opponents.
Historical ten-entry results do not establish an advantage in this group.
The simulator does not change live recommendations.

### Submitted-pick check

`data/pool-config.json` identifies the season, challenge, group, and entry.
`espn_pool.py` reads the group's current standings and actual rules.
The site displays the entrant count, points, rank, dropped week, tiebreak setting, and gap to first place.
It also displays the number of remaining regular-season games.
Accuracy remains the objective even when the entry trails the leader.

```bash
uv run --project model python model/espn_pool.py
```

The check uses these statuses:

- `matched`: a visible submitted pick matches the recommendation.
- `different`: a visible submitted pick differs from the recommendation.
- `missing`: the locked game has no visible pick, or a complete entry export confirms an open game has no pick.
- `unavailable`: the public feed does not expose enough information to check the pick.

ESPN can hide picks before kickoff. The public response cannot confirm that those picks are missing.
To check open picks, copy every submitted pick for the week from your signed-in ESPN entry into a local export.
Use an ignored path such as `model/artifacts/submitted-picks.json`:

```json
{
  "entryId": "3a361ee0-9107-11f1-88ce-316fbfbdb402",
  "season": 2026,
  "week": 7,
  "capturedAt": "2026-10-22T22:45:00Z",
  "complete": true,
  "picks": {
    "2026_07_NE_CHI": "CHI"
  }
}
```

Set `capturedAt` to the export time. Include every pick already submitted for that week.
An omitted game means no submitted pick when `complete` is true.
The checker rejects exports older than 30 minutes and exports for another entry or week.

```bash
uv run --project model python model/espn_pool.py --entry-file model/artifacts/submitted-picks.json
```

Set `ESPN_ENTRY_FILE` to the export path to include that check in local live generation.
The check reads picks; it does not submit or change them on ESPN.

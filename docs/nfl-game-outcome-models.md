# NFL Game-Outcome Prediction Models

## Scope

This note concerns pre-kickoff, straight-up win probabilities, not point-spread or margin forecasts. The app already uses chronological pregame features, logistic regression, and a de-vigged market blend. [Repository model note](../model/README.md).

## Established approaches

| Model | NFL precedent or primary source | Strengths for this app | Main limits | Use |
| --- | --- | --- | --- | --- |
| Elo | FiveThirtyEight publishes game-level NFL Elo forecasts from 1920 and separate QB-adjusted ratings. Its data include pregame ratings, home/neutral status, and win probabilities. [Source repository](https://github.com/fivethirtyeight/data/tree/master/nfl-elo) | Small, deterministic, sequential, and easy to explain. It is a strong no-feature baseline and a useful strength feature. | One latent team score needs hand-set update, home-field, season-regression, and QB rules. | Implement as the first challenger and keep it as a production fallback. |
| Bradley-Terry / Glicko | Bradley and Terry model paired-comparison probabilities from latent strengths. [Original paper](https://doi.org/10.1093/biomet/39.3-4.324). Glicko adds estimated rating uncertainty in a dynamic paired-comparison model. [Glickman, 1999](https://www.glicko.net/research/glicko.pdf) | Bradley-Terry makes the Elo probability model explicit. Glicko's rating deviation gives new teams, quarterbacks, and sparse observations less certainty. | Glicko is designed for rating periods and adds state choices. It still omits matchup features unless they are added separately. | Use a dynamic Bradley-Terry model before Glicko unless uncertainty intervals drive product decisions. |
| Logistic regression | Binomial logistic regression is the direct probability model for a binary winner label. [Cox, 1958](https://doi.org/10.1111/j.2517-6161.1958.tb00292.x) | Coefficients, diagnostics, regularization, and calibration are simple. It can combine rolling EPA, rest, venue, weather, starting QB, and market features. | Linear effects on log-odds can miss interactions and thresholds. Correlated team statistics need regularization. | Retain as the main statistical baseline. Test regularized versions before more complex models. |
| Random forest / gradient boosting | Random forests average randomized trees. [Breiman, 2001](https://doi.org/10.1023/A:1010933404324). Gradient boosting builds an additive model by fitting successive trees to the loss gradient. [Friedman, 2001](https://doi.org/10.1214/aos/1013203451) | Both learn nonlinear effects and interactions, such as weather by pass rate or QB status by team strength. Boosting is usually the smaller first tree-model experiment. | Small NFL samples make hyperparameter selection noisy. Raw tree probabilities can be poorly calibrated. Feature importance does not establish cause. | Run boosted trees and a random forest as offline challengers. Calibrate only on an earlier validation period. |
| Bayesian dynamic model | Glickman and Stern published a state-space model for NFL scores. [Author-hosted paper](https://www.glicko.net/research/nfl.pdf), cited on [Glickman's research page](https://www.glicko.net/research.html). | Partial pooling can stabilize early-season and weak-sample estimates. A dynamic offense/defense or team-strength state can report uncertainty intervals. | It requires prior, state-evolution, and inference choices. It is more difficult to operate and explain than logistic regression. | Consider after the feature data and evaluation set are stable, especially if uncertainty should change presentation or pick confidence. |

### Recommendation

1. Benchmark the existing logistic-plus-market model against market-only, Elo, and logistic without market inputs.
2. Add regularized logistic regression and Elo features before adding tree models.
3. Promote a boosted-tree or Bayesian model only when it improves out-of-time log loss and Brier score, and remains calibrated.

The market-only comparison is required. A model that does not beat a same-time de-vigged moneyline probability has not shown value beyond the market.

## Current Baseline Result

A local replay of the 2023-2025 regular seasons scored 815 games. The statistical logistic model had 62.3% accuracy, 0.2277 Brier score, and 0.6465 log loss. The recorded de-vigged market probability had 68.2% accuracy, 0.2103 Brier score, and 0.6081 log loss.

The prior-season blend selection chose a 100% market weight throughout this replay. Therefore, the current production blend matched the market baseline for this period. This result uses the moneyline timing stored by nflverse. It does not establish performance against odds available at the app's original prediction time.

The initial Elo challenger used 1500 starting ratings, 65 points of home-field advantage, a 20-point game update, and 67% rating carryover between seasons. It scored 60.5% accuracy, 0.2300 Brier score, and 0.6515 log loss. It did not beat the statistical model, so it should remain a baseline and not replace production predictions.

Adding the pregame Elo rating difference to the logistic model improved raw probabilities in the same replay: 63.7% accuracy, 0.2247 Brier score, and 0.6401 log loss. This improves on the original statistical model's 62.3%, 0.2277, and 0.6465. The market blend still selected 100% market weight, so the live model remains unchanged until a later evaluation can show value beyond the market baseline.

The QB challenger selects each team's primary passer from its latest completed game, then uses that passer's prior eight-game EPA per dropback, CPOE, and experience. It does not use the target game's recorded starter. It improved the raw model to 62.9% accuracy, 0.2274 Brier score, and 0.6462 log loss. This is a small gain over the baseline but is weaker than the Elo feature. The recorded market blend again selected a 100% market weight, so this challenger does not affect live picks.

The combined Elo-and-QB challenger scored 63.3% accuracy, 0.22465 Brier score, and 0.64012 log loss. It improved Brier score by 0.00006 over Elo alone, but accuracy was lower and the log-loss difference was negligible. Its market blend also selected a 100% market weight. Keep it as an offline challenger until a held-out replay shows a material market-blended gain.

The opponent-adjusted team challenger subtracts each opponent's pregame rolling offense or defense from a team's weekly EPA and success-rate values. It scored 62.2% accuracy, 0.2287 Brier score, and 0.6488 log loss. It lost to the unadjusted statistical model, so retain the result only as a rejected backtest challenger.

## Experiment Queue Results

`model/experiments.py` runs a registry of candidates through one expanding weekly replay. Every candidate is scored on the identical games. For each target week, the runner:

1. Trains on all completed games before the target week.
2. Chooses hyperparameters on the latest completed prior season, using a fit to the seasons before it.
3. Fits Platt calibration for tree models on that same prior-season validation set.
4. Chooses the market blend weight on that prior season with the Brier score.
5. Refits the chosen configuration on the full history and predicts the target week.

Outcome columns and the target game's recorded starters are never candidate features. Tests in `tests/test_experiments.py` check this rule, check that changing later outcomes does not change earlier predictions, and check the output shape.

### 2023-2025 replay (815 games, 54 weeks, all with recorded odds)

| Candidate | Accuracy | Brier | Log loss | Brier vs baseline (week-block SE) | Calibration slope | ECE | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Recorded market | 68.2% | 0.21033 | 0.60807 | -0.01732 (0.00355) | 1.09 | 0.031 | Benchmark |
| baseline (live features) | 62.3% | 0.22765 | 0.64650 | 0 | 0.99 | 0.016 | Live model |
| elo | 63.7% | 0.22471 | 0.64012 | -0.00294 (0.00197) | 0.94 | 0.027 | Improves baseline |
| qb | 62.9% | 0.22741 | 0.64618 | -0.00024 (0.00088) | 0.96 | 0.025 | Improves baseline, negligible |
| elo_qb | 63.3% | 0.22465 | 0.64012 | -0.00300 (0.00213) | 0.92 | 0.018 | Improves baseline |
| elo_injury (Elo + injury load) | 65.5% | 0.22159 | 0.63363 | -0.00606 (0.00270) | 0.96 | 0.043 | Improves baseline and elo |
| elo_warm (Elo warm-up from 2010) | 63.3% | 0.22421 | 0.63913 | -0.00344 (0.00234) | 0.96 | 0.033 | Improves baseline and elo |
| elo_mov (warm-up Elo + margin multiplier) | 63.9% | 0.22366 | 0.63806 | -0.00400 (0.00259) | 0.96 | 0.025 | Improves baseline and elo |
| elo_mov_injury (warm-up margin Elo + injury load) | 65.2% | 0.22038 | 0.63136 | -0.00727 (0.00335) | 0.97 | 0.042 | Improves baseline, elo, and elo_injury |
| elo_mov2 (aliased warm-up Elo, signed margin) | 63.8% | 0.22350 | 0.63775 | -0.00416 (0.00258) | 0.96 | 0.035 | Improves baseline and elo |
| elo_mov2_injury (signed-margin Elo + injury load) | 65.2% | 0.22030 | 0.63116 | -0.00735 (0.00333) | 0.97 | 0.041 | Improves baseline, elo, and elo_injury |
| elo_tuned_injury (per-week-tuned signed Elo + injury load) | 65.4% | 0.22026 | 0.63110 | -0.00739 (0.00334) | 0.97 | 0.038 | Improves baseline, elo, and elo_injury; best raw candidate |
| market_stack (Elo + injury + market logit) | 67.6% | 0.21231 | 0.61290 | -0.01534 (0.00350) | 1.00 | 0.031 | Improves baseline, does not beat the market |
| regularized_logistic (Elo+QB, C searched) | 62.6% | 0.22489 | 0.64011 | -0.00277 (0.00215) | 1.00 | 0.022 | Improves baseline |
| boosted_trees (depth 2, 100 trees, Platt) | 62.8% | 0.23088 | 0.65340 | +0.00323 (0.00301) | 0.97 | 0.061 | Rejected: accuracy gain without Brier or log-loss gain |
| random_forest (depth 3, Platt) | 62.7% | 0.23257 | 0.65956 | +0.00491 (0.00341) | 0.74 | 0.070 | Rejected: accuracy gain without Brier or log-loss gain |

Observations:

- No candidate beats the recorded market. The best raw log loss without market input (0.63110, elo_tuned_injury) is 0.023 worse than the market. market_stack closes most of the gap (0.00482 worse, week-block standard error 0.00467) but is still not better.
- Every logistic candidate without market input except elo_mov_injury, elo_mov2_injury, and elo_tuned_injury selected a 100% market weight in all 54 weeks, so its blended probability equals the market probability. Each of those three selected a 0.95 weight in 18 of 54 weeks (mean 0.98); their blends scored 0.21038, 0.21037, and 0.21037 Brier, each about 0.00004 Brier (SE 0.00009) worse than the market's 0.21033. market_stack averaged a 0.78 market weight and its blend scored 0.21073 Brier, marginally worse than the market. Live picks are unchanged.

The injury challenger (`elo_injury`) adds two pregame features to the Elo feature set: the home-minus-away difference in injury-report load (Out 1.0, Doubtful 0.75, Questionable 0.25, summed over each team's non-QB report rows) and the same difference over QB rows. The leakage rule counts an injury row from a season file that carries `date_modified` only when the UTC date of that timestamp is strictly earlier than the game's `gameday` date; rows modified on game day or later are excluded. The 2025 injury file has no `date_modified` column, so its rows are accepted without a date check. For 2021-2024, 99.7% of weighted report rows carry a `date_modified` at least one day before `gameday` and about 90% two days before (the Friday report), so the file is a final practice-week snapshot, not a game-day inactives list, and the timestamp check discards almost nothing. It scored 65.5% accuracy, 0.22159 Brier, and 0.63363 log loss: 0.00606 better than baseline (week-block SE 0.00270) and 0.00313 Brier and 0.00649 log loss better than elo alone (SE 0.00165 and 0.00352). Brier versus elo is -0.00038 in 2023 (SE 0.00217, 272 games), -0.00627 in 2024 (SE 0.00202, 272 games), and -0.00273 in 2025 (SE 0.00396, 271 games). The 2023 difference is near zero and the 2025 difference is within one standard error of zero, so two of three seasons support the gain but 2024 dominates it. Treat the result as supportive, not a confirmed gain. The candidate stays in the default queue.

The Elo-v2 challengers (`elo_warm`, `elo_mov`) replace the live Elo input with a warmer walk. Both start ratings in 2010 instead of 2021 and let postseason games update ratings (playoff weeks 19-22 sort after the regular season and never match a regular-season feature row), using the same 20-point K, 65-point home advantage, and 67% carryover as `predict.pregame_elo_differences`; with the margin term disabled the walk reproduces that function exactly. `elo_warm` keeps the plain win-loss update. `elo_mov` multiplies each update by FiveThirtyEight's margin-of-victory factor, ln(min(|margin|, 24) + 1) * 2.2 / (0.001 * |pregame difference from the winner's view| + 2.2), so a larger margin moves ratings more and a large pregame rating gap damps the update; a tie moves nothing under that factor. This differs from FiveThirtyEight, which uses the signed winner-minus-loser gap so that an upset moves ratings more than an expected win. The signed variant now has a tested counterpart: `elo_mov2` below uses it. `elo_warm` scored 63.3% accuracy, 0.22421 Brier, and 0.63913 log loss: 0.00344 better than baseline (week-block SE 0.00234) and 0.00050 Brier better than elo (SE 0.00059). `elo_mov` scored 63.9%, 0.22366, and 0.63806: 0.00400 better than baseline (SE 0.00259) and 0.00106 better than elo (SE 0.00099). The warm-up and the margin term each add a small, consistent gain over the cold-start Elo, and the margin term is slightly stronger.

Two Elo fixes came next: franchise aliasing and the signed margin factor.

Franchise aliasing: the 2010-2025 schedule from nflverse carries 35 team codes, not 32. The extra codes are `OAK` (Raiders before the 2020 move; the schedule uses `LV` after), `SD` (Chargers before 2017; the schedule uses `LAC` after), and `STL` (Rams before 2016; the schedule uses `LA` after, never `LAR`). The plain warm-up walk treated each old code as a new team, so those three franchises restarted at 1500 in their first season under the new code. `ELO_TEAM_ALIASES` maps `OAK` to `LV`, `SD` to `LAC`, and `STL` to `LA`. The mapping applies inside the Elo walk only, on a copy of the schedule frame, so feature rows keep the codes the schedule publishes. After the mapping the walk sees exactly 32 codes, and `load_features` asserts that count on every build.

The signed margin factor: `elo_mov2` runs the warm-up walk with FiveThirtyEight's signed gap, ln(min(|margin|, 24) + 1) * 2.2 / (0.001 * signed pregame difference from the winner's view + 2.2). An expected win keeps the same factor as before, because the winner's gap is positive. An upset gets a larger factor, so underdog wins move ratings more than favourite wins of the same margin. The default walk keeps the absolute gap, so the existing `elo_diff_mov` column and its tests are unchanged. `elo_mov2` uses the aliases, the signed factor, K=20, home advantage 65, carryover 0.67, and the margin cap of 24. It scored 63.8% accuracy, 0.22350 Brier, and 0.63775 log loss: 0.00416 better than baseline (week-block SE 0.00258) and 0.00016 Brier and 0.00031 log loss better than `elo_mov` (SE 0.00020 and 0.00045), a noise-level gain. The aliasing and the signed factor together add almost nothing over the unsigned walk on top of the injury features, as the next paragraph shows.

Per-week Elo parameter selection: the grid walk also builds 18 grid columns, `elo_grid_k{k}_h{h}_r{r}`, for K in (16, 20, 24), home advantage in (48, 65), and carryover in (0.5, 0.67, 0.8), all with the aliases, the signed factor, and the cap. Before fitting a target week, `elo_tuned_injury` picks the grid column whose pure Elo probability, 1 / (1 + 10 ** (-difference / 400)), has the lowest log loss over all completed rows strictly before that week. It writes that column into `elo_diff_tuned` for the history and the target week, and records the chosen parameters per week. Selection never sees the target week and never uses the validation season alone. The choice was stable: K=16, home advantage 48, carryover 0.67 won 46 of 54 weeks; K=20, home advantage 48, carryover 0.67 won 6; and K=20, home advantage 48, carryover 0.5 won 2. The replay prefers a smaller K and a smaller home advantage than the live settings.

`elo_mov_injury` stacks the same two injury features as `elo_injury` on the margin-of-victory Elo. It scored 65.2% accuracy, 0.22038 Brier, and 0.63136 log loss, 0.00727 better than baseline (week-block SE 0.00335) and 0.00433 Brier and 0.00877 log loss better than elo (SE 0.00214 and 0.00451). Against elo_injury it is 0.00121 Brier (SE 0.00101) and 0.00227 log loss (SE 0.00219) better, so it beats elo_injury on both scores pooled. Per season, its Brier versus elo_injury is -0.00086 in 2023 (SE 0.00202), -0.00317 in 2024 (SE 0.00163), and +0.00042 in 2025 (SE 0.00154): two of three seasons favor it and the 2025 difference is noise-level. The stack keeps the injury gain on top of the improved Elo, but the total is still 0.023 log loss short of the recorded market, so live picks stay unchanged.

The new round adds three challengers on top of that stack: `elo_mov2` (signed-margin Elo alone, table row above), `elo_mov2_injury` (the signed-margin Elo plus the two injury features), and `elo_tuned_injury` (the per-week-tuned grid column plus the two injury features). Against `elo_mov_injury`, pooled over 54 week blocks: `elo_mov2` is 0.00312 Brier (SE 0.00168) and 0.00639 log loss (SE 0.00362) worse, because it lacks the injury features; the like-for-like comparison is `elo_mov2` against `elo_mov`, a 0.00016 Brier gain (SE 0.00020). `elo_mov2_injury` is 0.00008 Brier (SE 0.00020) and 0.00019 log loss (SE 0.00044) better, and `elo_tuned_injury` is 0.00012 Brier (SE 0.00014) and 0.00026 log loss (SE 0.00030) better. Both differences are under one standard error, so the aliasing, the signed factor, and the parameter search do not produce a confirmed gain over the existing best. Per season, `elo_tuned_injury` versus `elo_mov_injury` is -0.00003 Brier in 2023 (SE 0.00014), +0.00017 in 2024 (SE 0.00021), and -0.00049 in 2025 (SE 0.00030); the 2025 Brier and log-loss differences (-0.00049 and -0.00120, SE 0.00030 and 0.00064) are the only ones near two standard errors, and one season out of three does not confirm a gain. `elo_tuned_injury` is the best raw candidate at 65.4% accuracy, 0.22026 Brier, and 0.63110 log loss, but the gain over `elo_mov_injury` is noise-level. Its market blend scored 0.21037 Brier and 0.60821 log loss, 0.00004 Brier (SE 0.00009) and 0.00013 log loss (SE 0.00022) worse than the recorded market. Nothing in this round changes live picks.

The market-stacking challenger (`market_stack`) trains the same Elo-plus-injury features together with `market_logit`, the recorded market probability on the logit scale. It is flagged `uses_market`: it trains only on games with a recorded market probability, never imputes the logit, and records its fitted coefficient each week. It scored 67.6% accuracy, 0.21231 Brier, and 0.61290 log loss. Against the recorded market it is 0.00198 Brier and 0.00482 log loss worse (week-block SE 0.00209 and 0.00467), which is within one standard error of the market. Per season, its Brier versus the recorded market is -0.00010 in 2023 (SE 0.00299), +0.00318 in 2024 (SE 0.00413), and +0.00287 in 2025 (SE 0.00381). The weekly `marketLogitCoefficient` ranged from 1.16 to 1.30 with a median of 1.22, so the fit consistently gives the market logit a weight at or above 1. The recorded moneylines are closing lines, so this replay cannot show market-beating value, and the candidate does not affect live picks.
- The Elo difference supplies almost all of the gain over the baseline. The QB features add nothing measurable after Elo.
- The regularization search is unstable. It chose C=0.01, 0.03, and 3.0 in different seasons and did not improve on the fixed C=0.6 used by `elo_qb`.
- Tree models lose to logistic regression. Their raw probabilities correlate only 0.79-0.84 with the baseline, so they are diverse, but their errors are larger. The blend selector gave them weights as low as 0.7-0.8 in some weeks, and the blended result (Brier 0.21074 and 0.21169) was worse than the market.
- The random forest correlates 0.93 with boosted trees and scores worse on every metric. It stays in the registry as an opt-in candidate but is not in the default queue.

### Per-season Brier score

Raw Brier score on the market-scored games of each season (pooled over 2023-2025). The summary's `bySeason` key in `model/artifacts/experiments.json` holds the full per-season metrics.

| Candidate | 2023 | 2024 | 2025 | Pooled |
| --- | --- | --- | --- | --- |
| Recorded market | 0.21871 | 0.20016 | 0.21214 | 0.21033 |
| baseline (live features) | 0.23341 | 0.22145 | 0.22811 | 0.22765 |
| elo | 0.23377 | 0.21397 | 0.22641 | 0.22471 |
| qb | 0.23423 | 0.22126 | 0.22675 | 0.22741 |
| elo_qb | 0.23464 | 0.21382 | 0.22550 | 0.22465 |
| elo_injury (Elo + injury load) | 0.23339 | 0.20770 | 0.22368 | 0.22159 |
| elo_warm (Elo warm-up from 2010) | 0.23319 | 0.21240 | 0.22706 | 0.22421 |
| elo_mov (warm-up Elo + margin multiplier) | 0.23283 | 0.21116 | 0.22700 | 0.22366 |
| elo_mov_injury (warm-up margin Elo + injury load) | 0.23253 | 0.20453 | 0.22409 | 0.22038 |
| elo_mov2 (aliased warm-up Elo, signed margin) | 0.23303 | 0.21107 | 0.22641 | 0.22350 |
| elo_mov2_injury (signed-margin Elo + injury load) | 0.23268 | 0.20467 | 0.22357 | 0.22030 |
| elo_tuned_injury (per-week-tuned signed Elo + injury load) | 0.23250 | 0.20470 | 0.22360 | 0.22026 |
| market_stack (Elo + injury + market logit) | 0.21861 | 0.20334 | 0.21501 | 0.21231 |
| regularized_logistic (Elo+QB, C searched) | 0.23504 | 0.21479 | 0.22483 | 0.22489 |
| boosted_trees (depth 2, 100 trees, Platt) | 0.23834 | 0.22624 | 0.22806 | 0.23088 |

Per-season spreads are large: 2024 was the most predictable season (market Brier 0.20016) and every candidate's biggest gain over the baseline came there. A pooled number can hide a single-season result, so candidate claims should quote the per-season differences, not only the pooled one.

These results use recorded nflverse closing moneylines. They can show that a candidate does or does not improve on the live model. They cannot show market-beating value. That claim needs the same-time multi-book snapshots in `data/odds-snapshots/` and enough completed games to give a week-block standard error smaller than the observed difference.

## Data and feature rules

Use one immutable row per game with an `as_of` timestamp. Each field must have been observable at that timestamp. This prevents future game results, final injury status, revised weather, and later odds from leaking into training.

- Use nflverse schedules for fixtures and results. The maintained release describes the schedules data as NFL game/schedule data and exposes it through `nflreadr` and `nflreadpy`. [Release source](https://github.com/nflverse/nflverse-data/releases/tag/schedules).
- Use nflverse play-by-play for rolling EPA and success-rate features. Freeze each feature before the target kickoff; do not calculate a season aggregate with the target game included.
- Use Pro-Football-Reference only as a historical results cross-check or supplemental archive. Record retrieval time, source URL, and field definitions. Do not silently mix its team identifiers or corrected results with nflverse data.
- Store an opening and prediction-time odds snapshot. Historical closing or current odds cannot stand in for the line available when a past prediction would have run.
- Keep regular season, playoffs, neutral-site games, and season openers identifiable. Report them separately when sample size permits.

## Time-aware evaluation

Random train/test splits are invalid for this use because they let later games affect earlier predictions. An empirical study of time-series evaluation found that out-of-sample methods preserving temporal order gave the most accurate estimates when non-stationary changes exist. [Cerqueira, Torgo, and Mozetic, 2020](https://doi.org/10.1007/s10994-020-05910-7).

1. Build features and odds snapshots as they existed before each kickoff.
2. Use expanding-window backtests: train through week *t-1*, predict week *t*, then advance. Refit all preprocessing and calibration inside each training window.
3. Select features, hyperparameters, and blend weights only on earlier seasons or folds. Keep one latest completed season untouched for final model selection.
4. Re-run the selected model over all completed history before live use. Keep the original per-game forecasts and later outcomes for audit.
5. Compare models on the identical games. Use paired score differences and season or week blocks for uncertainty intervals, rather than treating games as independent draws.

## Metrics

Report probabilities, not only picks. Proper scoring rules reward accurate and honest probabilities; the formal treatment is [Gneiting and Raftery, 2007](https://doi.org/10.1198/016214506000001437).

| Metric | Role | Interpretation |
| --- | --- | --- |
| Log loss | Primary selection metric | Penalizes confident incorrect probabilities strongly. Lower is better. |
| Brier score | Primary calibration metric | Mean squared probability error. Lower is better. It matches the app's current blend-selection metric. |
| Calibration curve and intercept/slope | Required diagnostic | Compare predicted probability bins with observed win rates. An overconfident model needs recalibration, not a new pick threshold. |
| Accuracy | Secondary display metric | Easy to understand, but it ignores probability quality and changes with the chosen 50% cutoff. |
| Market delta | Practical decision metric | Show log-loss and Brier-score differences from the same-time de-vigged market baseline. |

Do not choose a winner model by accuracy alone. A model can have the same picks as another model but materially better probability estimates.

## Source notes

- Primary sources were preferred: author-hosted papers, original algorithm papers, and source repositories.
- FiveThirtyEight's published NFL Elo dataset is an implementation reference, not evidence that its parameter values are optimal for this app.
- Glickman and Stern's score model supports the Bayesian/dynamic-model option. It does not by itself validate every modern feature set or odds source.

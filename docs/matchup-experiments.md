# Matchup, recency, and coaching tests

Scored 815 games from 2023–2025 with expanding weekly training.
Each week trains on earlier weeks. The prior season selects the market blend.
These tests are retrospective. The 2025 results are a temporal check, not a new untouched holdout.

## Probability scores

Lower Brier and log loss are better.

| Candidate | Accuracy | Raw Brier | Raw log loss | Blended Brier |
| --- | ---: | ---: | ---: | ---: |
| Recorded closing market | 68.2% | 0.21033 | 0.60807 | 0.21033 |
| baseline | 62.3% | 0.22765 | 0.64650 | 0.21033 |
| elo_tuned_injury | 65.4% | 0.22026 | 0.63110 | 0.21037 |
| team8 | 65.4% | 0.22130 | 0.63386 | 0.21033 |
| team4 | 65.3% | 0.22044 | 0.63227 | 0.21014 |
| team16 | 65.4% | 0.22119 | 0.63369 | 0.21033 |
| weighted3 | 66.9% | 0.22060 | 0.63280 | 0.21033 |
| offseason25 | 66.9% | 0.22063 | 0.63281 | 0.21021 |
| matchup | 64.8% | 0.22181 | 0.63541 | 0.21033 |
| coaching | 64.3% | 0.22189 | 0.63540 | 0.21033 |
| weighted_coaching | 65.8% | 0.22205 | 0.63678 | 0.21031 |

## Findings

Raw improvements beyond two week-block standard errors: none.

## Blends against the market

Negative differences favor the blend. A small gain relative to its SE does not establish an improvement.

| Candidate | Brier difference | SE | Log-loss difference | SE |
| --- | ---: | ---: | ---: | ---: |
| baseline | +0.00000 | 0.00000 | +0.00000 | 0.00000 |
| elo_tuned_injury | +0.00004 | 0.00009 | +0.00013 | 0.00022 |
| team8 | +0.00000 | 0.00000 | +0.00000 | 0.00000 |
| team4 | -0.00019 | 0.00022 | -0.00033 | 0.00052 |
| team16 | +0.00000 | 0.00000 | +0.00000 | 0.00000 |
| weighted3 | +0.00000 | 0.00000 | +0.00000 | 0.00000 |
| offseason25 | -0.00013 | 0.00024 | -0.00025 | 0.00054 |
| matchup | +0.00000 | 0.00000 | +0.00000 | 0.00000 |
| coaching | +0.00000 | 0.00000 | +0.00000 | 0.00000 |
| weighted_coaching | -0.00003 | 0.00009 | -0.00005 | 0.00021 |

## Added-feature comparisons

Negative differences favor the candidate. SE is the standard error across week blocks.

| Candidate vs control | Period | Brier difference | SE | Log-loss difference | SE |
| --- | --- | ---: | ---: | ---: | ---: |
| team8 vs elo_tuned_injury | pooled | +0.00103 | 0.00157 | +0.00276 | 0.00356 |
| team8 vs elo_tuned_injury | 2023 | +0.00393 | 0.00295 | +0.00898 | 0.00670 |
| team8 vs elo_tuned_injury | 2024 | +0.00300 | 0.00262 | +0.00762 | 0.00564 |
| team8 vs elo_tuned_injury | 2025 | -0.00385 | 0.00233 | -0.00837 | 0.00559 |
| team4 vs team8 | pooled | -0.00085 | 0.00175 | -0.00159 | 0.00388 |
| team4 vs team8 | 2023 | -0.00098 | 0.00351 | -0.00147 | 0.00782 |
| team4 vs team8 | 2024 | -0.00126 | 0.00298 | -0.00200 | 0.00692 |
| team4 vs team8 | 2025 | -0.00032 | 0.00269 | -0.00130 | 0.00564 |
| team16 vs team8 | pooled | -0.00011 | 0.00143 | -0.00017 | 0.00322 |
| team16 vs team8 | 2023 | -0.00132 | 0.00207 | -0.00313 | 0.00475 |
| team16 vs team8 | 2024 | -0.00111 | 0.00315 | -0.00257 | 0.00688 |
| team16 vs team8 | 2025 | +0.00211 | 0.00212 | +0.00520 | 0.00495 |
| weighted3 vs team8 | pooled | -0.00069 | 0.00129 | -0.00106 | 0.00289 |
| weighted3 vs team8 | 2023 | -0.00022 | 0.00304 | -0.00042 | 0.00684 |
| weighted3 vs team8 | 2024 | -0.00038 | 0.00174 | +0.00040 | 0.00398 |
| weighted3 vs team8 | 2025 | -0.00150 | 0.00178 | -0.00316 | 0.00393 |
| offseason25 vs weighted3 | pooled | +0.00003 | 0.00082 | +0.00001 | 0.00191 |
| offseason25 vs weighted3 | 2023 | +0.00027 | 0.00133 | +0.00027 | 0.00326 |
| offseason25 vs weighted3 | 2024 | -0.00205 | 0.00137 | -0.00386 | 0.00349 |
| offseason25 vs weighted3 | 2025 | +0.00188 | 0.00147 | +0.00363 | 0.00311 |
| matchup vs team8 | pooled | +0.00051 | 0.00048 | +0.00155 | 0.00122 |
| matchup vs team8 | 2023 | +0.00142 | 0.00107 | +0.00397 | 0.00280 |
| matchup vs team8 | 2024 | +0.00044 | 0.00080 | +0.00130 | 0.00204 |
| matchup vs team8 | 2025 | -0.00033 | 0.00054 | -0.00062 | 0.00114 |
| coaching vs elo_tuned_injury | pooled | +0.00162 | 0.00125 | +0.00430 | 0.00304 |
| coaching vs elo_tuned_injury | 2023 | +0.00094 | 0.00341 | +0.00390 | 0.00830 |
| coaching vs elo_tuned_injury | 2024 | +0.00292 | 0.00123 | +0.00652 | 0.00316 |
| coaching vs elo_tuned_injury | 2025 | +0.00101 | 0.00113 | +0.00246 | 0.00270 |
| weighted_coaching vs weighted3 | pooled | +0.00145 | 0.00134 | +0.00398 | 0.00331 |
| weighted_coaching vs weighted3 | 2023 | +0.00019 | 0.00360 | +0.00221 | 0.00876 |
| weighted_coaching vs weighted3 | 2024 | +0.00342 | 0.00148 | +0.00788 | 0.00424 |
| weighted_coaching vs weighted3 | 2025 | +0.00072 | 0.00116 | +0.00183 | 0.00260 |

## Blended comparisons

| Candidate vs control | Brier difference | SE | Log-loss difference | SE |
| --- | ---: | ---: | ---: | ---: |
| team8 vs elo_tuned_injury | -0.00004 | 0.00009 | -0.00013 | 0.00022 |
| team4 vs team8 | -0.00019 | 0.00022 | -0.00033 | 0.00052 |
| team16 vs team8 | +0.00000 | 0.00000 | +0.00000 | 0.00000 |
| weighted3 vs team8 | +0.00000 | 0.00000 | +0.00000 | 0.00000 |
| offseason25 vs weighted3 | -0.00013 | 0.00024 | -0.00025 | 0.00054 |
| matchup vs team8 | +0.00000 | 0.00000 | +0.00000 | 0.00000 |
| coaching vs elo_tuned_injury | -0.00004 | 0.00009 | -0.00013 | 0.00022 |
| weighted_coaching vs weighted3 | -0.00003 | 0.00009 | -0.00005 | 0.00021 |

## Feature definitions

- Team rates use pass EPA, rush EPA, sacks, and turnovers on offense and defense.
- Each rate divides weighted total counts by weighted total opportunities.
- The windows contain four, eight, or sixteen completed regular-season games.
- `weighted3` uses sixteen games with a three-game half-life.
- `offseason25` adds a 0.25 multiplier for games before the current season.
- Matchup features multiply each offense's rates by the opposing defense's rates and take the home-minus-away difference.
- Coaching features use log tenure, a recent observed change, and a change first observed this season.

## Limits

- Coaching features use the coach from the last completed regular-season game. They cannot detect a change before the new coach's first game.
- Coach tenure counts games in the available history, which begins in 2021. It excludes postseason games.
- These are coaching-continuity features, not estimates of each coach's skill.
- Team rates come from weekly box-score aggregates. Their definitions differ from the live play-by-play averages.
- Window comparisons use matching team-rate features, so they compare recency rather than different stat definitions.
- No feature uses target-week statistics or the target game's recorded coach.
- All specifications were fixed before this replay. Multiple exploratory tests can still produce false findings.
- Closing odds cannot establish an advantage against odds available when a live pick was published.

## Sources and reproduction

- [nflverse team stats](https://nflreadr.nflverse.com/reference/load_team_stats.html)
- [Team-stat definitions](https://nflreadr.nflverse.com/articles/dictionary_team_stats.html)
- [nflverse schedule](https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv)

Reuse the existing `features-v6-2021-2025.pkl` cache.
The JSON report records source hashes and per-game predictions.

```bash
uv run --project model python model/matchup_experiments.py --download-inputs
```

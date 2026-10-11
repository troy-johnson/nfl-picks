# Core recency tests

Scored 2184 games from 2018–2026 with recorded market odds.
Each candidate replaces the eight rolling strength features. No additional team-rate features enter the model.
Rest, neutral-site context, estimator, regularization, Elo, and injury definitions stay fixed within each control pair.
Each week trains on earlier weeks. The prior season selects the market blend.

## Combined scores

Lower Brier and log loss are better.

| Candidate | Raw accuracy | Raw Brier | Raw log loss | Blended Brier | Difference vs market | 95% Brier interval |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Recorded closing market | 66.3% | 0.21138 | 0.61044 | 0.21138 | 0 | — |
| baseline | 62.4% | 0.22861 | 0.64872 | 0.21178 | +0.00040 | [-0.00004, +0.00084] |
| elo_tuned_injury | 64.0% | 0.22283 | 0.63699 | 0.21162 | +0.00024 | [-0.00001, +0.00049] |
| baseline_core4 | 61.7% | 0.22985 | 0.65123 | 0.21159 | +0.00021 | [-0.00052, +0.00100] |
| strong_core4 | 64.2% | 0.22127 | 0.63333 | 0.21187 | +0.00049 | [-0.00013, +0.00117] |
| baseline_core12 | 62.6% | 0.22728 | 0.64602 | 0.21179 | +0.00041 | [-0.00016, +0.00095] |
| strong_core12 | 64.0% | 0.22289 | 0.63701 | 0.21166 | +0.00028 | [-0.00002, +0.00060] |
| baseline_core_exp4 | 63.0% | 0.22475 | 0.64029 | 0.21195 | +0.00057 | [-0.00015, +0.00129] |
| strong_core_exp4 | 64.4% | 0.22160 | 0.63398 | 0.21182 | +0.00044 | [-0.00002, +0.00092] |

## Adjustment for six variants

These Bonferroni-adjusted bootstrap intervals use 99.17% coverage for each raw-score comparison.
They allow for the six new variants in this experiment. They do not allow for all earlier model exploration.

| Candidate | Raw Brier difference | Adjusted interval | Raw log-loss difference | Adjusted interval |
| --- | ---: | --- | ---: | --- |
| baseline_core4 | +0.00124 | [-0.00410, +0.00657] | +0.00250 | [-0.00922, +0.01396] |
| strong_core4 | -0.00156 | [-0.00421, +0.00106] | -0.00366 | [-0.00957, +0.00221] |
| baseline_core12 | -0.00133 | [-0.00493, +0.00227] | -0.00270 | [-0.01052, +0.00508] |
| strong_core12 | +0.00006 | [-0.00175, +0.00185] | +0.00001 | [-0.00409, +0.00400] |
| baseline_core_exp4 | -0.00386 | [-0.00673, -0.00117] | -0.00843 | [-0.01457, -0.00271] |
| strong_core_exp4 | -0.00123 | [-0.00263, +0.00011] | -0.00302 | [-0.00612, -0.00000] |

## Comparisons with the eight-game control

Negative differences favor the candidate. Intervals use 10,000 paired samples of complete season-week blocks.

| Period | Candidate | Raw Brier difference | 95% interval | Raw log-loss difference | 95% interval |
| --- | --- | ---: | --- | ---: | --- |
| 2018–2022 | baseline_core4 | +0.00122 | [-0.00393, +0.00652] | +0.00264 | [-0.00870, +0.01432] |
| 2018–2022 | strong_core4 | -0.00194 | [-0.00465, +0.00076] | -0.00481 | [-0.01091, +0.00132] |
| 2018–2022 | baseline_core12 | -0.00182 | [-0.00519, +0.00144] | -0.00358 | [-0.01100, +0.00351] |
| 2018–2022 | strong_core12 | -0.00020 | [-0.00192, +0.00149] | -0.00076 | [-0.00459, +0.00306] |
| 2018–2022 | baseline_core_exp4 | -0.00407 | [-0.00680, -0.00147] | -0.00898 | [-0.01490, -0.00331] |
| 2018–2022 | strong_core_exp4 | -0.00135 | [-0.00273, -0.00003] | -0.00363 | [-0.00674, -0.00061] |
| 2023–2025 | baseline_core4 | +0.00259 | [-0.00348, +0.00854] | +0.00515 | [-0.00781, +0.01768] |
| 2023–2025 | strong_core4 | -0.00033 | [-0.00306, +0.00240] | -0.00056 | [-0.00653, +0.00538] |
| 2023–2025 | baseline_core12 | -0.00112 | [-0.00577, +0.00368] | -0.00268 | [-0.01264, +0.00756] |
| 2023–2025 | strong_core12 | +0.00021 | [-0.00221, +0.00269] | +0.00057 | [-0.00479, +0.00607] |
| 2023–2025 | baseline_core_exp4 | -0.00344 | [-0.00660, -0.00023] | -0.00748 | [-0.01420, -0.00061] |
| 2023–2025 | strong_core_exp4 | -0.00099 | [-0.00264, +0.00070] | -0.00198 | [-0.00567, +0.00173] |
| 2026 | baseline_core4 | -0.01542 | [-0.02731, -0.00257] | -0.03334 | [-0.06025, -0.00539] |
| 2026 | strong_core4 | -0.00939 | [-0.01216, -0.00583] | -0.01967 | [-0.02697, -0.01217] |
| 2026 | baseline_core12 | +0.00590 | [-0.00427, +0.01792] | +0.01463 | [-0.00670, +0.03904] |
| 2026 | strong_core12 | +0.00360 | [-0.00062, +0.00758] | +0.00863 | [+0.00010, +0.01756] |
| 2026 | baseline_core_exp4 | -0.00488 | [-0.01209, +0.00830] | -0.00942 | [-0.02561, +0.01850] |
| 2026 | strong_core_exp4 | -0.00180 | [-0.00489, +0.00175] | -0.00362 | [-0.01110, +0.00398] |
| All periods | baseline_core4 | +0.00124 | [-0.00276, +0.00509] | +0.00250 | [-0.00616, +0.01087] |
| All periods | strong_core4 | -0.00156 | [-0.00353, +0.00040] | -0.00366 | [-0.00805, +0.00079] |
| All periods | baseline_core12 | -0.00133 | [-0.00404, +0.00131] | -0.00270 | [-0.00855, +0.00304] |
| All periods | strong_core12 | +0.00006 | [-0.00129, +0.00142] | +0.00001 | [-0.00299, +0.00306] |
| All periods | baseline_core_exp4 | -0.00386 | [-0.00592, -0.00192] | -0.00843 | [-0.01288, -0.00428] |
| All periods | strong_core_exp4 | -0.00123 | [-0.00227, -0.00022] | -0.00302 | [-0.00534, -0.00074] |

## Control checks

Maximum difference between rebuilt and original eight-game inputs: 0.
All variants use the same game-level EPA and success-rate definitions.
The exponentially weighted variant uses a four-game half-life and the latest 24 completed regular-season games.
Equal-weight variants use four, eight, or twelve games.

## Limits

- Earlier-period training begins in 2016. Recent-period training begins in 2021, matching the prior experiment setup.
- This experiment changes the averaging window, not the underlying game metrics or feature count.
- Elo selection uses the same prior-game grid for every variant. Market weights can differ because validation scores differ.
- The 2026 injury reports lack timestamps. The JSON report includes a sensitivity replay with current-season injury differences set to zero.
- The 2026 sample has only five week blocks. Its confidence intervals can be unstable.
- All periods have now been inspected. A selected candidate needs a frozen specification and evaluation on future games.
- Multiple exploratory comparisons increase the chance of a false finding. These intervals do not correct for that exploration.
- Market comparisons use closing odds. They do not prove an advantage at the time of a published pick.
- This offline replay does not change live picks.

## Reproduce

Reuse the original feature cache and the extended backtest inputs.
The JSON output records source hashes, candidate definitions, and per-game predictions.

```bash
uv run --project model python model/recency_experiments.py --download-inputs
```

- [nflverse play-by-play](https://github.com/nflverse/nflverse-data/releases/tag/pbp)
- [Earlier backtest](extended-backtest.md)

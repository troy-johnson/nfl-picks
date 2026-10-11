# Extended four-game backtest

The four-game specification stays fixed. No new feature or parameter is selected from these results.
Training expands by week. Blend selection uses only the latest prior season.

## Results

Lower Brier and log loss are better.

| Period | Games | Four-game blend accuracy | Blend Brier | Market Brier | Difference | 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2018–2022 validation | 1304 | 65.4% | 0.21130 | 0.21081 | +0.00049 | [-0.00003, +0.00096] |
| 2023–2025 development | 815 | 68.2% | 0.21014 | 0.21033 | -0.00019 | [-0.00062, +0.00024] |
| 2026 validation | 65 | 60.0% | 0.23330 | 0.23591 | -0.00261 | [-0.00709, +0.00041] |
| All periods, descriptive | 2184 | 66.3% | 0.21152 | 0.21138 | +0.00014 | [-0.00022, +0.00049] |
| 2026 without undated injuries | 65 | 60.0% | 0.23318 | 0.23591 | -0.00273 | [-0.00882, +0.00021] |

A negative difference favors the four-game blend. An interval that crosses zero does not establish an improvement.

## Paired comparisons

Intervals use 10,000 paired bootstrap samples of complete season-week blocks.
They preserve unequal game counts across weeks.

| Period | Comparison | Brier difference | 95% interval | Log-loss difference | 95% interval |
| --- | --- | ---: | --- | ---: | --- |
| 2018–2022 validation | team4_blend_vs_market | +0.00049 | [-0.00003, +0.00096] | +0.00118 | [+0.00002, +0.00225] |
| 2018–2022 validation | team4_raw_vs_team8_raw | -0.00097 | [-0.00428, +0.00233] | -0.00201 | [-0.00955, +0.00536] |
| 2018–2022 validation | team4_raw_vs_elo_tuned_injury_raw | +0.00236 | [-0.00138, +0.00622] | +0.00573 | [-0.00274, +0.01479] |
| 2018–2022 validation | team4_blend_vs_elo_tuned_injury_blend | +0.00011 | [-0.00021, +0.00042] | +0.00025 | [-0.00048, +0.00096] |
| 2023–2025 development | team4_blend_vs_market | -0.00019 | [-0.00062, +0.00024] | -0.00033 | [-0.00133, +0.00070] |
| 2023–2025 development | team4_raw_vs_team8_raw | -0.00085 | [-0.00426, +0.00265] | -0.00159 | [-0.00916, +0.00617] |
| 2023–2025 development | team4_raw_vs_elo_tuned_injury_raw | +0.00018 | [-0.00400, +0.00458] | +0.00117 | [-0.00841, +0.01128] |
| 2023–2025 development | team4_blend_vs_elo_tuned_injury_blend | -0.00023 | [-0.00055, +0.00009] | -0.00046 | [-0.00119, +0.00027] |
| 2026 validation | team4_blend_vs_market | -0.00261 | [-0.00709, +0.00041] | -0.00678 | [-0.01923, -0.00098] |
| 2026 validation | team4_raw_vs_team8_raw | -0.00125 | [-0.01972, +0.01445] | -0.00324 | [-0.04255, +0.02955] |
| 2026 validation | team4_raw_vs_elo_tuned_injury_raw | -0.00949 | [-0.02132, -0.00414] | -0.02021 | [-0.04595, -0.00874] |
| 2026 validation | team4_blend_vs_elo_tuned_injury_blend | -0.00261 | [-0.00709, +0.00041] | -0.00678 | [-0.01923, -0.00098] |

## Design and limits

- The development period is 2023–2025. Its predictions come from the original experiment report without refitting.
- The earlier-period replay scores 2018–2022 and starts training in 2016.
- The 2026 replay preserves the original 2021–2025 training features and adds completed 2026 games.
- The 2026 sample is small and covers only games completed when the source files were downloaded.
- The 2026 injury file lacks report timestamps. Its features cannot be verified as available before kickoff.
- The sensitivity replay sets both injury differences to zero for 2026. Historical training and blend selection stay unchanged.
- The earlier replay uses more training history than the original development replay. Its market and model controls share that history.
- Franchise aliases preserve team histories across relocations in the earlier replay.
- The challenger adds four-game team rates to the existing Elo, injury, and eight-game play-by-play features.
- The blend uses recorded closing odds. This test cannot establish a tradable advantage against earlier odds.
- Bootstrap intervals describe sampling uncertainty. They do not correct for earlier exploration across multiple candidates.
- The combined result includes the development sample. It is descriptive, not an independent validation result.
- These are reconstructed historical predictions. The challenger was not published before these games.

## Same-time archive

Matched 32 completed 2026 games to stored pregame odds.
The JSON report compares raw statistical probabilities with those odds.
The reconstructed blend uses closing odds, so its same-time comparison is not a prospective market test.

## Reproduce

First run the original matchup experiment to create its feature cache and development report.
The runner uses compressed play-by-play, injury reports, and weekly team stats cached in `model/artifacts/`.
The JSON report records hashes of the source files, original feature cache, and original experiment report.

```bash
uv run --project model python model/extended_backtest.py --download-inputs
```

Use `--refresh-features` to rebuild caches after a feature-builder change.

## Sources

- [nflverse play-by-play](https://github.com/nflverse/nflverse-data/releases/tag/pbp)
- [nflverse team stats](https://nflreadr.nflverse.com/reference/load_team_stats.html)
- [nflverse injuries](https://github.com/nflverse/nflverse-data/releases/tag/injuries)
- [nflverse schedule](https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv)

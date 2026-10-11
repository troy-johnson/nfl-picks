# Venue, travel, and QB age tests

Scored 815 regular-season games from 2023–2025 with expanding weekly training.

Each week uses earlier games for training and the prior season for blend selection.
These are retrospective tests. The 2025 results are a separate temporal check, not a new untouched holdout.

## Probability scores

Lower Brier and log loss are better.

| Candidate | Accuracy | Raw Brier | Raw log loss | Blended Brier |
| --- | ---: | ---: | ---: | ---: |
| Recorded closing market | 68.2% | 0.21033 | 0.60807 | 0.21033 |
| baseline | 62.3% | 0.22765 | 0.64650 | 0.21033 |
| elo_tuned_injury | 65.4% | 0.22026 | 0.63110 | 0.21037 |
| qb | 62.9% | 0.22741 | 0.64618 | 0.21033 |
| venue | 61.8% | 0.22919 | 0.64979 | 0.21033 |
| travel | 62.5% | 0.22793 | 0.64681 | 0.21054 |
| context | 62.7% | 0.22920 | 0.64948 | 0.21033 |
| strong_context | 64.8% | 0.22167 | 0.63454 | 0.21052 |
| qb_age | 62.9% | 0.22892 | 0.64980 | 0.21033 |
| market_stack | 67.6% | 0.21231 | 0.61290 | 0.21073 |
| market_context | 67.0% | 0.21358 | 0.61600 | 0.21048 |

## Findings

Additions that improve both raw scores: none.
Additions that improve both raw scores beyond two week-block standard errors: none.
These exploratory comparisons need confirmation with later games before a model change.

## Added-feature comparisons

Negative differences favor the added features. SE is the standard error across week blocks.

| Candidate vs control | Period | Brier difference | SE | Log-loss difference | SE |
| --- | --- | ---: | ---: | ---: | ---: |
| venue vs baseline | pooled | +0.00154 | 0.00067 | +0.00329 | 0.00149 |
| venue vs baseline | 2023 | +0.00360 | 0.00159 | +0.00783 | 0.00365 |
| venue vs baseline | 2024 | +0.00113 | 0.00073 | +0.00225 | 0.00155 |
| venue vs baseline | 2025 | -0.00013 | 0.00083 | -0.00021 | 0.00182 |
| travel vs baseline | pooled | +0.00027 | 0.00127 | +0.00030 | 0.00274 |
| travel vs baseline | 2023 | -0.00008 | 0.00213 | -0.00038 | 0.00443 |
| travel vs baseline | 2024 | -0.00063 | 0.00229 | -0.00191 | 0.00506 |
| travel vs baseline | 2025 | +0.00153 | 0.00231 | +0.00321 | 0.00491 |
| context vs baseline | pooled | +0.00154 | 0.00128 | +0.00297 | 0.00278 |
| context vs baseline | 2023 | +0.00253 | 0.00215 | +0.00526 | 0.00466 |
| context vs baseline | 2024 | +0.00043 | 0.00196 | +0.00031 | 0.00440 |
| context vs baseline | 2025 | +0.00168 | 0.00258 | +0.00336 | 0.00551 |
| strong_context vs elo_tuned_injury | pooled | +0.00141 | 0.00122 | +0.00344 | 0.00273 |
| strong_context vs elo_tuned_injury | 2023 | +0.00239 | 0.00203 | +0.00590 | 0.00453 |
| strong_context vs elo_tuned_injury | 2024 | -0.00019 | 0.00154 | -0.00054 | 0.00371 |
| strong_context vs elo_tuned_injury | 2025 | +0.00203 | 0.00267 | +0.00496 | 0.00582 |
| qb_age vs qb | pooled | +0.00151 | 0.00111 | +0.00362 | 0.00251 |
| qb_age vs qb | 2023 | +0.00205 | 0.00228 | +0.00577 | 0.00517 |
| qb_age vs qb | 2024 | +0.00153 | 0.00214 | +0.00315 | 0.00480 |
| qb_age vs qb | 2025 | +0.00095 | 0.00130 | +0.00193 | 0.00294 |
| market_context vs market_stack | pooled | +0.00127 | 0.00105 | +0.00310 | 0.00242 |
| market_context vs market_stack | 2023 | +0.00131 | 0.00176 | +0.00390 | 0.00403 |
| market_context vs market_stack | 2024 | +0.00094 | 0.00154 | +0.00123 | 0.00376 |
| market_context vs market_stack | 2025 | +0.00155 | 0.00221 | +0.00418 | 0.00489 |

## Blended comparisons

| Candidate vs control | Brier difference | SE | Log-loss difference | SE |
| --- | ---: | ---: | ---: | ---: |
| venue vs baseline | +0.00000 | 0.00000 | +0.00000 | 0.00000 |
| travel vs baseline | +0.00021 | 0.00023 | +0.00051 | 0.00053 |
| context vs baseline | +0.00000 | 0.00000 | +0.00000 | 0.00000 |
| strong_context vs elo_tuned_injury | +0.00015 | 0.00020 | +0.00048 | 0.00044 |
| qb_age vs qb | +0.00000 | 0.00000 | +0.00000 | 0.00000 |
| market_context vs market_stack | -0.00025 | 0.00038 | -0.00058 | 0.00083 |

## Error splits

These overlapping groups are descriptive. Small groups do not establish a predictive effect.
The baseline blend selects 100% market in every replay week, so its split scores equal the market scores.

| Split | Games | Raw Brier | Blended Brier | Market Brier | Blend accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| market_home_favorite | 487 | 0.21996 | 0.20676 | 0.20676 | 68.8% |
| market_away_favorite | 328 | 0.23908 | 0.21564 | 0.21564 | 67.4% |
| low_confidence | 358 | 0.25577 | 0.23644 | 0.23644 | 61.5% |
| medium_confidence | 233 | 0.22055 | 0.21199 | 0.21199 | 69.5% |
| high_confidence | 224 | 0.19011 | 0.16689 | 0.16689 | 77.7% |
| early_season | 191 | 0.23645 | 0.22300 | 0.22300 | 64.9% |
| late_season | 234 | 0.22557 | 0.20569 | 0.20569 | 67.9% |
| enclosed_venue | 277 | 0.22491 | 0.20546 | 0.20546 | 68.2% |
| open_venue | 538 | 0.22907 | 0.21284 | 0.21284 | 68.2% |
| artificial_surface | 358 | 0.22842 | 0.20562 | 0.20562 | 69.0% |
| grass_surface | 419 | 0.22536 | 0.21175 | 0.21175 | 68.7% |
| high_altitude | 26 | 0.23067 | 0.20438 | 0.20438 | 69.2% |
| long_away_trip | 179 | 0.22313 | 0.20384 | 0.20384 | 72.1% |
| early_away_body_clock | 105 | 0.23272 | 0.21068 | 0.21068 | 70.5% |
| short_rest | 58 | 0.22070 | 0.21651 | 0.21651 | 69.0% |
| neutral_site | 17 | 0.23472 | 0.22083 | 0.22083 | 70.6% |
| qb_age_35_plus | 179 | 0.23778 | 0.21706 | 0.21706 | 68.7% |
| observed_wind_15_plus | 46 | 0.20738 | 0.17600 | 0.17600 | 84.8% |
| observed_temp_40_or_less | 83 | 0.23036 | 0.21555 | 0.21555 | 65.1% |

## Limits

- Weather splits use recorded game weather. No weather feature enters a candidate model.
- A weather model needs historical forecasts saved before kickoff.
- Travel distance uses the team's latest prior home stadium. Actual travel itineraries are unknown.
- Time-zone features use the kickoff date, including daylight saving time.
- Roof enclosure uses stadium metadata and earlier roof observations, not the target game's roof status.
- Surface uses the historical schedule, rather than today's stadium surface.
- Age terms use the prior game's primary QB and birthdate. They do not use the target game's recorded starter.
- QB age terms test an age adjustment. They do not estimate a complete player aging curve.
- Recorded closing odds cannot establish an advantage against odds available when a live pick was published.
- Multiple exploratory comparisons increase the chance of a false finding.

## Data coverage

Missing feature values use training-set medians.

```json
{
  "venue_enclosure": 0,
  "venue_artificial": 44,
  "venue_altitude_km": 0,
  "division_game": 0,
  "travel_distance_diff": 0,
  "eastward_shift_diff": 0,
  "early_body_clock_diff": 0,
  "short_rest_diff": 0,
  "bye_rest_diff": 0,
  "road_streak_diff": 0,
  "qb_age_diff": 16,
  "qb_age_squared_diff": 16,
  "qb_over_35_diff": 16,
  "qb_under_24_diff": 16
}
```

## Sources

- [nflverse schedules](https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv)
- [Stadium coordinates and time zones](https://github.com/greerreNFL/Stadiums)
- [nflverse player birthdates](https://nflreadr.nflverse.com/reference/load_players.html)

## Reproduce

Build the existing feature cache if it is missing.
Then run the context experiments.

```bash
uv run --project model python model/experiments.py --candidates baseline --from-season 2023 --to-season 2025
uv run --project model python model/context_experiments.py --download-inputs
```

The JSON report records input hashes. Source files can change after this run.

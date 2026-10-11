# Recency forward test

The frozen candidate uses 24 completed games with a four-game half-life.
The control uses the original eight-game averages.
Both models exclude the entire target week from training and team histories.
Both models use the frozen estimator and prior-season blend selection.

The capture workflow now runs `model/recency_forward.py --capture` in the existing pregame window.
It saves raw probabilities, blended probabilities, model inputs, source hashes, and fresh multi-book odds.
The archive stores odds and predictions under `data/model-predictions/recency/<season>/<day>.json`.
Each archive preserves the first capture. The script refuses a save after kickoff.
If the first capture window is missed, a later window can capture games that have not started.
The archive excludes games that started before that capture.
The script verifies the frozen prediction sources before capture and evaluation.
The frozen specification remains unchanged, including its original status text.
This workflow replaces that status once the changes reach the scheduled branch.

Evaluation writes `data/model-evaluation/recency.json`.
It scores completed, non-tied regular-season games with valid pregame captures after the freeze.
It excludes duplicate games, changed specifications, changed prediction sources, and late captures.
The primary comparison uses raw Brier and log loss against the eight-game control.
Secondary comparisons use the market and market blends.
Paired intervals resample complete season-week blocks.
The report requires at least 250 completed games before reporting a confirmed primary improvement.
That minimum does not guarantee enough evidence. Both confidence intervals must also stay below zero.
Secondary comparisons remain exploratory. The scripts do not change published picks.

```bash
uv run --project model python model/recency_forward.py --capture
uv run --project model python model/recency_forward.py
```

Capture requires `THE_ODDS_API_KEY` and current nflverse inputs.
Current-season play-by-play refreshes on each due capture.
Historical play-by-play caches remain local and their hashes enter the archive.
New-season games with no available play-by-play cause capture to fail until the source becomes available.
Scheduled capture starts only after these files reach the workflow branch.

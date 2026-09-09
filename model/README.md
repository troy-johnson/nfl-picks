# Prediction model

V1 builds pregame rolling team-strength features from nflverse play-by-play, trains a logistic regression with chronological validation, then blends its probability with the de-vigged market probability using the prior season's Brier score to choose the blend weight.

The first weekly run is snapshotted. Later runs flag material market moves, favorite flips, quarterback changes, and severe outdoor weather. Injury headlines are intentionally not assigned arbitrary point values; current injury/starter news should be reviewed alongside market movement before the first kickoff.

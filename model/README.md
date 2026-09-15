# Prediction model

V1 builds pregame rolling team-strength features from nflverse play-by-play, trains a logistic regression with chronological validation, then blends its probability with the de-vigged market probability using the prior season's Brier score to choose the blend weight.

The Wednesday run creates the weekly snapshot. The final run occurs 60-120 minutes before the first kickoff. It refreshes market, quarterback, weather, and team news data. Injury headlines are intentionally not assigned arbitrary point values; they are shown for review beside market movement before the first kickoff.

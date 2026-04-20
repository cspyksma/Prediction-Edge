# Kalshi Edge Research Program: Exhaustive First-Wave Search With Robust Champion Selection

## Summary
Build a new research backtest flow whose sole optimization target is robust out-of-sample Kalshi trading edge, not pure game-winner accuracy. The system should search broadly across model families, feature families, market-aware variants, pure market strategies, simple ensembles, and bet-sizing policies, then rank contenders on time-sliced 2025-2026 Kalshi profitability with consistency guardrails.

Training and evaluation should be deliberately separated:
- Use 2015-2021 as the historical training and development era.
- Use 2025-2026 Kalshi as the true out-of-sample trading evaluation era.
- Use 2015-2021 SBRO data as a sharper-market reference during development and for market-aware feature construction where applicable.
- Do not treat SBRO and Kalshi as one continuous backtest.

The first wave should be exhaustive across practical non-neural approaches. Neural-network challengers come only after the first wave establishes a serious baseline.

## Core Research Structure
- Add a new fixed-split research command rather than overloading the existing Kalshi or walk-forward commands.
  It should support explicit `--train-start-date`, `--train-end-date`, `--eval-start-date`, and `--eval-end-date`.
- Implement a research pipeline that builds the full candidate set once, trains only on the train window, freezes every contender, and evaluates only on the Kalshi eval window.
- Treat the final unit of comparison as a full strategy, not just a classifier.
  A strategy is the combination of:
  - feature set
  - model family
  - market-input policy
  - bet-selection rule
  - bet-sizing rule
- Keep the current command family intact.
  Existing `historical-backtest-kalshi` and `walkforward-backtest` behavior should remain unchanged.

## Contender Families To Include In Wave One
- Baseball-only models.
  These use game/team/pitcher/bullpen/weather and similar baseball features without market inputs.
- Hybrid market-aware models.
  These use baseball features plus sharper-market inputs derived from SBRO-era pricing where available.
- Pure market-only strategies.
  These are allowed to ignore baseball features and compete directly if they better exploit Kalshi versus sharper-market dislocations.
- Simple ensembles.
  Include practical combinations such as probability averaging, weighted blends, and stackers across top non-neural contenders.
- Existing model families must be included:
  `logreg`, `histgb`, `knn`, `svm`, and the current Bayesian posterior benchmark.
- Add practical first-wave classical challengers where supported by the environment and dependency policy.
  The implementation should prioritize strong tabular contenders and avoid introducing neural nets in this wave.

## Feature Search in Wave One
- Run feature-family search alongside model-family search.
- Partition features into at least these families:
  - baseball fundamentals and form
  - pitcher and bullpen state
  - venue, travel, rest, streak, and schedule context
  - weather
  - market-derived reference features from SBRO era
  - pure market-only inputs
- Support contender variants with:
  - baseball-only features
  - market-only features
  - baseball plus market features
- Preserve feature provenance in results so it is always clear whether a winning contender is driven by baseball information, market information, or both.

## Strategy Layer and Sizing Search
- Include bet entry logic as part of the search space rather than assuming a single threshold.
- Candidate selection rules should include:
  - simple edge-threshold entry
  - probability-gap entry
  - filtered entry by confidence/calibration bucket
  - filtered entry by agreement or disagreement with sharper-market reference
- Include bet sizing in wave one because ROI is the target and fixed-size staking alone is too noisy.
- Candidate sizing rules should include:
  - flat stake baseline
  - capped proportional sizing
  - fractional Kelly with strict caps
  - confidence-bucket sizing
- Every sizing policy must include explicit hard caps so one or two bets cannot dominate the result.

## Data Flow and Evaluation Design
- Build a unified chronological research dataset across the combined train and eval date ranges using the existing feature pipeline where possible.
- Fit contenders on 2015-2021 only.
- Freeze contender parameters after training.
- Score 2025-2026 only against Kalshi replay quotes.
- Use SBRO only in the historical development context and in model variants or features that intentionally consume sharper-market information.
- Do not let 2025-2026 Kalshi data leak into training, feature tuning, or champion selection logic outside the defined time-sliced evaluation framework.

## Kalshi Out-of-Sample Evaluation
- Evaluate the 2025-2026 Kalshi period in sequential time slices rather than one single aggregate block.
- Each contender should produce:
  - aggregate Kalshi ROI
  - units won
  - number of bets
  - hit rate
  - average estimated edge
  - slice-by-slice ROI and units
  - max drawdown or equivalent loss-concentration metric
  - calibration summary on eval predictions
- Slice definitions should be fixed and chronological.
  The implementation should choose one consistent slicing rule and apply it to every contender.
- The main purpose of slicing is to reduce luck and expose instability, not to retrain during evaluation.

## Champion Selection
- The champion is the most robust money-making Kalshi strategy, not the most accurate winner classifier.
- Rank contenders using Kalshi trading results first, with consistency guardrails applied before crown selection.
- Champion selection should require:
  - minimum bet count
  - acceptable slice coverage
  - no extreme dependence on one or two slices
  - acceptable drawdown or loss concentration
- Among contenders that clear guardrails, rank primarily by out-of-sample Kalshi ROI.
- Use tiebreakers in this order:
  - higher units won
  - better slice consistency
  - lower drawdown
  - better calibration / lower log loss
- The research champion should be reported clearly but should not automatically replace the existing production champion path unless explicitly promoted later.

## Reporting and Interfaces
- Add a dedicated research CLI command for this program.
  It should be explicit that this is a fixed-split Kalshi-edge search, not a standard training command.
- The output should include:
  - overall research winner
  - per-contender summary table
  - family labels for each contender
  - feature-set labels for each contender
  - whether the contender is baseball-only, hybrid, or market-only
  - sizing rule and entry rule
  - time-slice stability table
- Results should make it obvious whether edge is coming from:
  - baseball signal
  - sharper-market anchoring
  - pure market dislocation
  - sizing logic
- Preserve enough metadata for later comparison and reranking without rerunning the entire experiment.

## Test Plan
- Add unit tests for fixed-split training/eval separation.
  Ensure train rows come only from 2015-2021 and eval scoring comes only from 2025-2026.
- Add tests that multiple contender families can run in one experiment and all are surfaced in results.
- Add tests for baseball-only, hybrid, and market-only contender labeling.
- Add tests that strategy metadata includes feature family, model family, entry rule, and sizing rule.
- Add tests for Kalshi slice aggregation and stability metrics.
- Add tests for champion selection guardrails so low-sample lucky contenders do not win.
- Add tests for no-data and partial-data conditions:
  - no Kalshi eval quotes
  - missing SBRO inputs for market-aware contenders
  - eval slices with sparse bets
- Add CLI tests that verify the research formatter prints the winner and per-contender tables.
- Keep the existing lazy import behavior so commands unrelated to xlsx reporting do not fail on optional reporting dependencies.

## Assumptions and Defaults
- “All machine learning methods possible” for wave one means an exhaustive practical non-neural search, not literally every algorithm in existence.
- Neural networks are explicitly deferred until after the first wave.
- The Bayesian posterior remains a benchmark contender in reporting unless later promoted into the full contender ranking logic.
- The final objective is robust Kalshi profitability, not winner-prediction leaderboard performance.
- SBRO is allowed as an input where useful because ROI is the objective, but baseball-only and market-only families must also compete so the source of edge remains interpretable.
- Consistency matters enough that a small-sample, high-ROI contender should lose to a slightly lower-ROI contender with stronger stability and coverage.

## Immediate First Implementation Pass
- Introduce the new fixed-split research backtest skeleton and CLI.
- Wire in current candidate models plus contender-family metadata.
- Add market-only and hybrid contender tracks.
- Add time-sliced Kalshi evaluation and robust champion selection.
- Add first-wave sizing and entry-rule search.
- Produce structured reporting rich enough to identify what kind of strategy is winning before expanding to neural nets.

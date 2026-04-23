#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# backfill_history.sh
#
# Backfills historical Kalshi market data for all available MLB seasons and
# then retrains the game-outcome model on the full multi-year dataset.
#
# Usage:
#   ./scripts/backfill_history.sh [--train-only] [--skip-2023] [--skip-2024] [--skip-2025]
#
# Options:
#   --train-only   Skip the backfill step; only retrain the model on whatever
#                  historical data is already in the database.
#   --skip-YYYY    Skip the backfill for the given season year.
#   --chunk-days N Use N-day chunks for each backfill run (default: 7).
#                  Smaller chunks are friendlier to rate limits; larger chunks
#                  reduce total overhead.
#
# What this script does
# ─────────────────────
# 1. Ingests SBRO (SportsBookReviewsOnline) closing-line workbooks from
#    ./sbro/*.xlsx covering 2015-2021. This populates
#    historical_market_priors with a single per-game moneyline close and the
#    devigged home/away fair probabilities. Idempotent.
#
# 2. For each MLB season with a Kalshi ticker (2025+), runs:
#      mlpm historical-backfill-kalshi --start-date YYYY-03-01 --end-date YYYY-10-31
#    The backfill is resumable (--no-resume is NOT passed), so re-running this
#    script safely skips already-completed chunks.
#
# 3. Trains the game-outcome model on the full range 2015-03-01 → today:
#      mlpm train-game-model --start-date 2015-03-01
#
# Notes
# ─────
# • Kalshi MLB game-market tickers only exist from 2025 onward. Pre-2025
#   market priors come from the SBRO archive (2015-2021). There is a data
#   gap for 2022-2024 where neither source covers the full season — the
#   market_home_implied_prob feature will be NaN for those games and the
#   classifier pipelines median-impute it.
# • SBRO files live in ./sbro/*.xlsx. The script calls `mlpm ingest-sbro` to
#   load them into historical_market_priors; train-game-model unions that
#   table with live quotes + Kalshi replay via _load_historical_market_priors.
# • Rate limiting: if you see rate_limited_count > 0 in the output, add a small
#   sleep between seasons or reduce --chunk-days.
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# ── Configuration ──────────────────────────────────────────────────────────
SEASONS=("2025")                 # Kalshi tickers available 2025+. Pre-2025 priors come from SBRO.
CHUNK_DAYS=7
TRAIN_ONLY=false
SKIP_SBRO=false
SKIP_SEASONS=()
SBRO_DIR="./sbro"                # Location of SportsBookReviewsOnline xlsx archives (2015-2021)
TRAIN_START_DATE="2015-03-01"    # Earliest date to train on — covers SBRO archive + Kalshi replay

# ── Argument parsing ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --train-only)   TRAIN_ONLY=true; shift ;;
        --skip-sbro)    SKIP_SBRO=true; shift ;;
        --skip-2023)    SKIP_SEASONS+=("2023"); shift ;;
        --skip-2024)    SKIP_SEASONS+=("2024"); shift ;;
        --skip-2025)    SKIP_SEASONS+=("2025"); shift ;;
        --chunk-days)   CHUNK_DAYS="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

should_skip() {
    local season="$1"
    for s in "${SKIP_SEASONS[@]:-}"; do
        [[ "$s" == "$season" ]] && return 0
    done
    return 1
}

# ── SBRO ingest (2015-2021 closing-line moneylines) ────────────────────────
if [[ "$TRAIN_ONLY" == "false" && "$SKIP_SBRO" == "false" ]]; then
    if [[ -d "$SBRO_DIR" ]]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  SBRO Historical Ingest ($SBRO_DIR)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        mlpm ingest-sbro --directory "$SBRO_DIR"
        echo "  SBRO ingest complete."
        echo ""
    else
        echo "[skip] SBRO directory '$SBRO_DIR' not found — skipping SBRO ingest."
    fi
fi

# ── Backfill ───────────────────────────────────────────────────────────────
if [[ "$TRAIN_ONLY" == "false" ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Kalshi Historical Backfill"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for YEAR in "${SEASONS[@]}"; do
        if should_skip "$YEAR"; then
            echo "[skip] Season $YEAR (--skip-$YEAR was passed)"
            continue
        fi

        START="${YEAR}-03-01"
        # Use Oct 31 for completed seasons; today for the current season
        CURRENT_YEAR="$(date +%Y)"
        if [[ "$YEAR" -lt "$CURRENT_YEAR" ]]; then
            END="${YEAR}-10-31"
        else
            END="$(date +%Y-%m-%d)"
        fi

        echo ""
        echo "▶ Season $YEAR: $START → $END  (chunk-days=$CHUNK_DAYS)"
        mlpm historical-backfill-kalshi \
            --start-date "$START" \
            --end-date   "$END"   \
            --chunk-days "$CHUNK_DAYS"
        echo "  Done."
    done

    echo ""
    echo "Backfill complete."
fi

# ── Retrain ────────────────────────────────────────────────────────────────
TODAY="$(date +%Y-%m-%d)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Training model on $TRAIN_START_DATE → $TODAY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
mlpm train-game-model \
    --start-date "$TRAIN_START_DATE" \
    --end-date   "$TODAY"

echo ""
echo "✓ All done. Model trained on full historical dataset."

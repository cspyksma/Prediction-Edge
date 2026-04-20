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
# 1. For each MLB season (March–October), runs:
#      mlpm historical-backfill-kalshi --start-date YYYY-03-01 --end-date YYYY-10-31
#    The backfill is resumable (--no-resume is NOT passed), so re-running this
#    script safely skips already-completed chunks.
#
# 2. Trains the game-outcome model on the full range 2023-03-01 → today:
#      mlpm train-game-model --start-date 2023-03-01
#
# Notes
# ─────
# • Kalshi started MLB game markets around early 2023, so 2022 data is sparse.
#   The script starts at 2023 by default; edit SEASONS below to extend.
# • The backfill stores data in the historical_kalshi_quotes DuckDB table, which
#   is now automatically used by train-game-model via _load_historical_market_priors.
# • Rate limiting: if you see rate_limited_count > 0 in the output, add a small
#   sleep between seasons or reduce --chunk-days.
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# ── Configuration ──────────────────────────────────────────────────────────
SEASONS=("2023" "2024" "2025")   # Edit to add/remove seasons
CHUNK_DAYS=7
TRAIN_ONLY=false
SKIP_SEASONS=()
TRAIN_START_DATE="2023-03-01"    # Earliest date to train on

# ── Argument parsing ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --train-only)   TRAIN_ONLY=true; shift ;;
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

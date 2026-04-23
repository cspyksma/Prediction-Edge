#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# backfill_all.sh
#
# Downloads MLB game fundamentals (schedule + scores + pitching + batting
# logs) and per-game stadium weather into the local DuckDB, year by year.
# Default window is 2015-01-01 through 2021-12-31, matching the SBRO
# sportsbook-odds xlsx files already sitting in `sbro/`. That gives you a
# fully database-first corpus for train/eval runs with market-rate
# comparisons (via ingest-sbro) through the 2021 season.
#
# Usage:
#   ./scripts/backfill_all.sh
#   ./scripts/backfill_all.sh --start-year 2018 --end-year 2021
#   ./scripts/backfill_all.sh --skip-mlb          # only run weather backfill
#   ./scripts/backfill_all.sh --skip-weather      # only run MLB backfill
#   ./scripts/backfill_all.sh --force-mlb         # re-fetch MLB months already populated
#   ./scripts/backfill_all.sh --force-weather     # re-fetch weather rows already populated
#
# What this script does
# ─────────────────────
# 1. For each year in the window, runs:
#      mlpm backfill-mlb     --start-date YYYY-01-01 --end-date YYYY-12-31
#    Populates: games, game_results, mlb_pitching_logs, mlb_batting_logs
#
# 2. For each year in the window (after step 1), runs:
#      mlpm backfill-weather --start-date YYYY-01-01 --end-date YYYY-12-31
#    Populates: game_weather  (one row per game_id, from Open-Meteo)
#
# Both commands are idempotent by default — re-running the script safely
# skips months/games that are already populated. Pass --force-* to override.
#
# Order matters: weather backfill looks up game rows (stadium + date), so
# MLB must be backfilled first. We process MLB all years, then weather all
# years, so the weather pass sees a complete game table.
#
# Notes
# ─────
# • Runtime: 7 seasons of MLB fundamentals is ~17,000 games × 1 boxscore
#   HTTP call each, plus the schedule calls. Expect a few hours on a warm
#   connection. Rerun-safe: kill with Ctrl-C and re-run, it picks up.
# • Kalshi market data is not included here (only exists from 2023). If you
#   want it later, use scripts/backfill_history.sh.
# • SBRO sportsbook closing lines (2015-2021) are a separate one-liner:
#   `mlpm ingest-sbro --directory sbro`.
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# ── Configuration ──────────────────────────────────────────────────────────
START_YEAR=2015
END_YEAR=2021
RUN_MLB=true
RUN_WEATHER=true
FORCE_MLB=false
FORCE_WEATHER=false

# ── Argument parsing ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-year)     START_YEAR="$2"; shift 2 ;;
        --end-year)       END_YEAR="$2";   shift 2 ;;
        --skip-mlb)       RUN_MLB=false;      shift ;;
        --skip-weather)   RUN_WEATHER=false;  shift ;;
        --force-mlb)      FORCE_MLB=true;     shift ;;
        --force-weather)  FORCE_WEATHER=true; shift ;;
        -h|--help)
            sed -n '3,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if (( START_YEAR > END_YEAR )); then
    echo "Error: --start-year ($START_YEAR) must be <= --end-year ($END_YEAR)." >&2
    exit 1
fi

# ── Helpers ────────────────────────────────────────────────────────────────
print_header() {
    local title="$1"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $title"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

run_mlb_year() {
    local year="$1"
    local extra=()
    if [[ "$FORCE_MLB" == "true" ]]; then
        extra+=(--force)
    fi

    local t_start=$SECONDS
    echo ""
    echo "▶ [MLB] Season ${year}: ${year}-01-01 → ${year}-12-31"
    mlpm backfill-mlb \
        --start-date "${year}-01-01" \
        --end-date   "${year}-12-31" \
        "${extra[@]}"
    local elapsed=$(( SECONDS - t_start ))
    printf "  Done in %dm %ds.\n" $(( elapsed / 60 )) $(( elapsed % 60 ))
}

run_weather_year() {
    local year="$1"
    local extra=()
    if [[ "$FORCE_WEATHER" == "true" ]]; then
        extra+=(--no-resume)
    fi

    local t_start=$SECONDS
    echo ""
    echo "▶ [Weather] Season ${year}: ${year}-01-01 → ${year}-12-31"
    mlpm backfill-weather \
        --start-date "${year}-01-01" \
        --end-date   "${year}-12-31" \
        "${extra[@]}"
    local elapsed=$(( SECONDS - t_start ))
    printf "  Done in %dm %ds.\n" $(( elapsed / 60 )) $(( elapsed % 60 ))
}

# ── Summary ────────────────────────────────────────────────────────────────
print_header "Backfill plan"
echo "  Window:        ${START_YEAR}-01-01 → ${END_YEAR}-12-31"
echo "  MLB step:      $( [[ "$RUN_MLB"     == "true" ]] && echo enabled || echo skipped )$( [[ "$FORCE_MLB"     == "true" ]] && echo " (force)" )"
echo "  Weather step:  $( [[ "$RUN_WEATHER" == "true" ]] && echo enabled || echo skipped )$( [[ "$FORCE_WEATHER" == "true" ]] && echo " (force)" )"
echo "  Working dir:   $(pwd)"

# ── MLB backfill (all years) ───────────────────────────────────────────────
if [[ "$RUN_MLB" == "true" ]]; then
    print_header "MLB fundamentals backfill  (statsapi.mlb.com)"
    for (( YEAR=START_YEAR; YEAR<=END_YEAR; YEAR++ )); do
        run_mlb_year "$YEAR"
    done
    echo ""
    echo "MLB backfill complete."
else
    echo ""
    echo "[skip] MLB backfill (--skip-mlb was passed)"
fi

# ── Weather backfill (all years, after MLB is populated) ───────────────────
if [[ "$RUN_WEATHER" == "true" ]]; then
    print_header "Stadium weather backfill  (api.open-meteo.com)"
    for (( YEAR=START_YEAR; YEAR<=END_YEAR; YEAR++ )); do
        run_weather_year "$YEAR"
    done
    echo ""
    echo "Weather backfill complete."
else
    echo ""
    echo "[skip] Weather backfill (--skip-weather was passed)"
fi

print_header "Done"
echo "  DuckDB now holds ${START_YEAR}-${END_YEAR} MLB fundamentals + weather."
echo ""
echo "  Suggested next steps:"
echo "    mlpm ingest-sbro --directory sbro   # load 2015-2021 closing odds"
echo "    mlpm research-kalshi-edge \\"
echo "      --train-start-date ${START_YEAR}-01-01 --train-end-date ${END_YEAR}-12-31 \\"
echo "      --eval-start-date  ${START_YEAR}-01-01 --eval-end-date  ${END_YEAR}-12-31"
echo ""

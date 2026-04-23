# ---------------------------------------------------------------------------
# backfill_all.ps1
#
# PowerShell sibling of backfill_all.sh. Downloads MLB game fundamentals
# (schedule + scores + pitching + batting logs) and per-game stadium weather
# into the local DuckDB, year by year. Default window is 2015-01-01 through
# 2021-12-31, matching the SBRO sportsbook-odds xlsx files already sitting in
# `sbro/`.
#
# Usage (from the MLPM project root):
#   .\scripts\backfill_all.ps1
#   .\scripts\backfill_all.ps1 -StartYear 2018 -EndYear 2021
#   .\scripts\backfill_all.ps1 -SkipMlb          # only run weather backfill
#   .\scripts\backfill_all.ps1 -SkipWeather      # only run MLB backfill
#   .\scripts\backfill_all.ps1 -ForceMlb         # re-fetch MLB months already populated
#   .\scripts\backfill_all.ps1 -ForceWeather     # re-fetch weather rows already populated
#
# What this script does
# ---------------------
# 1. For each year in the window, runs:
#      mlpm backfill-mlb     --start-date YYYY-01-01 --end-date YYYY-12-31
#    Populates: games, game_results, mlb_pitching_logs, mlb_batting_logs
#
# 2. For each year in the window (after step 1), runs:
#      mlpm backfill-weather --start-date YYYY-01-01 --end-date YYYY-12-31
#    Populates: game_weather  (one row per game_id, from Open-Meteo)
#
# Both commands are idempotent by default - re-running safely skips months
# and games that are already populated. Pass -ForceMlb / -ForceWeather to
# override.
#
# Order matters: weather backfill looks up game rows (stadium + date), so
# MLB must be backfilled first. This script processes all MLB years, then
# all weather years, so the weather pass sees a complete game table.
#
# Notes
# -----
# * Runtime: 7 seasons of MLB fundamentals is ~17,000 games x 1 boxscore
#   HTTP call each, plus the schedule calls. Expect a few hours on a warm
#   connection. Rerun-safe: Ctrl-C and re-run, it picks up.
# * Kalshi market data is not included here (only exists from 2023). Use
#   scripts\backfill_history.ps1 for that.
# * SBRO sportsbook closing lines (2015-2021) are a separate one-liner:
#   mlpm ingest-sbro --directory sbro
# ---------------------------------------------------------------------------

param(
    [int]$StartYear = 2015,
    [int]$EndYear = 2021,
    [switch]$SkipMlb,
    [switch]$SkipWeather,
    [switch]$ForceMlb,
    [switch]$ForceWeather
)

$ErrorActionPreference = "Stop"

# Add the project virtual environment's Scripts folder to PATH so mlpm is available
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectRoot

$VenvScripts = Join-Path $ProjectRoot ".venv\Scripts"
if (Test-Path $VenvScripts) {
    $env:PATH = "$VenvScripts;$env:PATH"
} else {
    Write-Warning "Could not find .venv\Scripts - make sure your virtual environment is activated."
}

if ($StartYear -gt $EndYear) {
    Write-Error "StartYear ($StartYear) must be <= EndYear ($EndYear)."
    exit 1
}

function Write-Header {
    param([string]$Title)
    Write-Host ""
    Write-Host "--------------------------------------------------------------"
    Write-Host "  $Title"
    Write-Host "--------------------------------------------------------------"
}

function Invoke-MlbYear {
    param([int]$Year)

    $start = Get-Date
    Write-Host ""
    Write-Host ">> [MLB] Season ${Year}: ${Year}-01-01 to ${Year}-12-31"

    $mlbArgs = @(
        "backfill-mlb",
        "--start-date", "${Year}-01-01",
        "--end-date",   "${Year}-12-31"
    )
    if ($ForceMlb) { $mlbArgs += "--force" }

    mlpm @mlbArgs
    if ($LASTEXITCODE -ne 0) {
        throw "mlpm backfill-mlb exited with code $LASTEXITCODE for year $Year"
    }

    $elapsed = (Get-Date) - $start
    Write-Host ("  Done in {0}m {1}s." -f [int]$elapsed.TotalMinutes, $elapsed.Seconds)
}

function Invoke-WeatherYear {
    param([int]$Year)

    $start = Get-Date
    Write-Host ""
    Write-Host ">> [Weather] Season ${Year}: ${Year}-01-01 to ${Year}-12-31"

    $wxArgs = @(
        "backfill-weather",
        "--start-date", "${Year}-01-01",
        "--end-date",   "${Year}-12-31"
    )
    if ($ForceWeather) { $wxArgs += "--no-resume" }

    mlpm @wxArgs
    if ($LASTEXITCODE -ne 0) {
        throw "mlpm backfill-weather exited with code $LASTEXITCODE for year $Year"
    }

    $elapsed = (Get-Date) - $start
    Write-Host ("  Done in {0}m {1}s." -f [int]$elapsed.TotalMinutes, $elapsed.Seconds)
}

# -- Summary --------------------------------------------------------------
Write-Header "Backfill plan"
Write-Host "  Window:        ${StartYear}-01-01 to ${EndYear}-12-31"
$mlbState     = if ($SkipMlb)     { "skipped" } else { if ($ForceMlb)     { "enabled (force)" } else { "enabled" } }
$weatherState = if ($SkipWeather) { "skipped" } else { if ($ForceWeather) { "enabled (force)" } else { "enabled" } }
Write-Host "  MLB step:      $mlbState"
Write-Host "  Weather step:  $weatherState"
Write-Host "  Working dir:   $(Get-Location)"

# -- MLB backfill (all years) ---------------------------------------------
if (-not $SkipMlb) {
    Write-Header "MLB fundamentals backfill  (statsapi.mlb.com)"
    for ($Year = $StartYear; $Year -le $EndYear; $Year++) {
        Invoke-MlbYear -Year $Year
    }
    Write-Host ""
    Write-Host "MLB backfill complete."
} else {
    Write-Host ""
    Write-Host "[skip] MLB backfill (-SkipMlb was passed)"
}

# -- Weather backfill (all years, after MLB is populated) -----------------
if (-not $SkipWeather) {
    Write-Header "Stadium weather backfill  (api.open-meteo.com)"
    for ($Year = $StartYear; $Year -le $EndYear; $Year++) {
        Invoke-WeatherYear -Year $Year
    }
    Write-Host ""
    Write-Host "Weather backfill complete."
} else {
    Write-Host ""
    Write-Host "[skip] Weather backfill (-SkipWeather was passed)"
}

Write-Header "Done"
Write-Host "  DuckDB now holds ${StartYear}-${EndYear} MLB fundamentals + weather."
Write-Host ""
Write-Host "  Suggested next steps:"
Write-Host "    mlpm ingest-sbro --directory sbro   # load 2015-2021 closing odds"
Write-Host "    mlpm research-kalshi-edge ``"
Write-Host "      --train-start-date ${StartYear}-01-01 --train-end-date ${EndYear}-12-31 ``"
Write-Host "      --eval-start-date  ${StartYear}-01-01 --eval-end-date  ${EndYear}-12-31"
Write-Host ""

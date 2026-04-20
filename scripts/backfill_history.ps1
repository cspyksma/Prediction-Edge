# ---------------------------------------------------------------------------
# backfill_history.ps1
#
# Backfills historical Kalshi market data for all available MLB seasons and
# then retrains the game-outcome model on the full multi-year dataset.
#
# Usage (from the MLPM project root):
#   .\scripts\backfill_history.ps1
#   .\scripts\backfill_history.ps1 -TrainOnly
#   .\scripts\backfill_history.ps1 -Skip2023 -ChunkDays 14
# ---------------------------------------------------------------------------

param(
    [switch]$TrainOnly,
    [switch]$Skip2023,
    [switch]$Skip2024,
    [switch]$Skip2025,
    [int]$ChunkDays = 7
)

$ErrorActionPreference = "Stop"

# Add the project virtual environment's Scripts folder to PATH so mlpm is available
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$VenvScripts = Join-Path $ProjectRoot ".venv\Scripts"
if (Test-Path $VenvScripts) {
    $env:PATH = "$VenvScripts;$env:PATH"
} else {
    Write-Warning "Could not find .venv\Scripts - make sure your virtual environment is activated."
}

$Seasons = @("2025")          # KXMLBGAME series data starts around 2025; 2023/2024 return empty
$TrainStartDate = "2023-03-01" # Train on MLB game data from 2023 (market feature defaults to 0.5 when missing)
$Today = (Get-Date).ToString("yyyy-MM-dd")
$CurrentYear = (Get-Date).Year

if (-not $TrainOnly) {
    Write-Host "--------------------------------------------------------------"
    Write-Host "  Kalshi Historical Backfill"
    Write-Host "--------------------------------------------------------------"

    foreach ($Year in $Seasons) {
        $Skip = ($Year -eq "2023" -and $Skip2023) -or
                ($Year -eq "2024" -and $Skip2024) -or
                ($Year -eq "2025" -and $Skip2025)

        if ($Skip) {
            Write-Host "[skip] Season ${Year}"
            continue
        }

        $Start = "${Year}-03-01"
        $End   = if ([int]$Year -lt $CurrentYear) { "${Year}-10-31" } else { $Today }

        Write-Host ""
        Write-Host ">> Season ${Year}: $Start to $End  (chunk-days=$ChunkDays)"
        mlpm historical-backfill-kalshi --start-date $Start --end-date $End --chunk-days $ChunkDays
        Write-Host "  Done."
    }

    Write-Host ""
    Write-Host "Backfill complete."
}

Write-Host ""
Write-Host "--------------------------------------------------------------"
Write-Host "  Training model on $TrainStartDate to $Today"
Write-Host "--------------------------------------------------------------"
mlpm train-game-model --start-date $TrainStartDate --end-date $Today

Write-Host ""
Write-Host "Done. Model trained on full historical dataset."

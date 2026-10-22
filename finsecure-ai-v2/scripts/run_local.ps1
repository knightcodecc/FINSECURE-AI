# FinSecure AI v2 - Local Launcher
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "  FinSecure AI v2 - Starting Local Environment   " -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan

# 1. Install dependencies
Write-Host "[1/4] Checking and installing dependencies..." -ForegroundColor Yellow
python -m pip install -r requirements.txt --quiet

# 2. Build aggregates if missing
Write-Host "[2/4] Verifying precomputed data aggregates..." -ForegroundColor Yellow
if (-not (Test-Path "data/aggregates_synthetic.json") -and -not (Test-Path "data/aggregates_paysim.json")) {
    Write-Host "Aggregates missing. Running scripts/build_aggregates.py..." -ForegroundColor Cyan
    python scripts/build_aggregates.py
} else {
    Write-Host "Precomputed aggregates found in data/." -ForegroundColor Green
}

# 3. Determine port from $env:PORT (default 8001)
$PORT = if ($env:PORT) { $env:PORT } else { 8001 }
$URL = "http://127.0.0.1:$PORT"
Write-Host "[3/4] Configuring server on port $PORT..." -ForegroundColor Yellow

# 4. Launch browser asynchronously
Write-Host "[4/4] Starting server and opening dashboard at $URL..." -ForegroundColor Green
Start-Process $URL

# Start uvicorn
python -m uvicorn api.main:app --host 127.0.0.1 --port $PORT --reload

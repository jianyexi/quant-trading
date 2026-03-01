#!/usr/bin/env pwsh
# start.ps1 - Build and start production server
# Usage: .\start.ps1 [-SkipBuild] [-Port 8080]
# Builds frontend, then runs the Rust backend serving both API and static UI.

param(
    [switch]$SkipBuild,
    [int]$Port = 8080
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not $SkipBuild) {
    Write-Host "[BUILD] Building frontend..." -ForegroundColor Cyan
    Push-Location web
    npm run build
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Frontend build failed" -ForegroundColor Red
        Pop-Location
        exit 1
    }
    Pop-Location
    Write-Host "[OK] Frontend built -> web/dist" -ForegroundColor Green
    Write-Host ""
}

Write-Host "[START] QuantTrader server on port $Port..." -ForegroundColor Cyan
Write-Host "   URL: http://localhost:$Port" -ForegroundColor Yellow
Write-Host ""

cargo run --package quant-cli -- --config config/default.toml serve

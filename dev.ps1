#!/usr/bin/env pwsh
# dev.ps1 - Start backend + frontend dev server for development
# Usage: .\dev.ps1
#   Backend: cargo run on :8080 (separate window)
#   Frontend: vite dev on :3000, proxies /api -> :8080 (this window)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Ensure cargo is in PATH (rustup default location)
$cargoPath = Join-Path $env:USERPROFILE ".cargo\bin"
if ((Test-Path $cargoPath) -and ($env:PATH -notlike "*$cargoPath*")) {
    $env:PATH = "$cargoPath;$env:PATH"
}

Write-Host "[DEV] Starting QuantTrader development servers..." -ForegroundColor Cyan
Write-Host "   Backend:  http://localhost:8080  (Rust API)" -ForegroundColor DarkGray
Write-Host "   Frontend: http://localhost:3000  (Vite HMR)" -ForegroundColor DarkGray
Write-Host ""

# Start backend in a new terminal window
$backend = Start-Process pwsh -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$PSScriptRoot'; Write-Host '[BACKEND] Building and starting...' -ForegroundColor Cyan; cargo run --package quant-cli -- --config config/default.toml serve"
) -PassThru

Write-Host "[OK] Backend starting in separate window (PID $($backend.Id))" -ForegroundColor Green
Write-Host "   Open http://localhost:3000 in your browser" -ForegroundColor Yellow
Write-Host "   Press Ctrl+C here to stop frontend. Close backend window separately." -ForegroundColor DarkGray
Write-Host ""

# Run frontend in this window (blocks until Ctrl+C)
try {
    Set-Location web
    npm run dev
} finally {
    Set-Location $PSScriptRoot
}

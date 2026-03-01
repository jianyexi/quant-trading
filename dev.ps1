#!/usr/bin/env pwsh
# dev.ps1 — Start backend + frontend dev server (HMR) in parallel
# Usage: .\dev.ps1
#   Backend: cargo run on :8080
#   Frontend: vite dev on :3000 (proxies /api → :8080)

$ErrorActionPreference = "Stop"

Write-Host "🚀 Starting QuantTrader development servers..." -ForegroundColor Cyan
Write-Host "   Backend:  http://localhost:8080  (Rust API)" -ForegroundColor DarkGray
Write-Host "   Frontend: http://localhost:3000  (Vite HMR)" -ForegroundColor DarkGray
Write-Host ""

$backend = Start-Process -PassThru -NoNewWindow pwsh -ArgumentList @(
    "-Command",
    "Set-Location '$PSScriptRoot'; cargo run --package quant-cli -- --config config/default.toml serve"
)

# Wait a moment for backend to start compiling
Start-Sleep -Seconds 2

$frontend = Start-Process -PassThru -NoNewWindow pwsh -ArgumentList @(
    "-Command",
    "Set-Location '$PSScriptRoot/web'; npm run dev"
)

Write-Host ""
Write-Host "✅ Both servers starting. Press Ctrl+C to stop." -ForegroundColor Green
Write-Host "   Open http://localhost:3000 in your browser." -ForegroundColor Yellow
Write-Host ""

try {
    # Wait for either process to exit
    while (-not $backend.HasExited -and -not $frontend.HasExited) {
        Start-Sleep -Seconds 1
    }
} finally {
    # Cleanup: stop both processes
    if (-not $backend.HasExited) {
        Write-Host "Stopping backend (PID $($backend.Id))..." -ForegroundColor DarkGray
        Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
    }
    if (-not $frontend.HasExited) {
        Write-Host "Stopping frontend (PID $($frontend.Id))..." -ForegroundColor DarkGray
        Stop-Process -Id $frontend.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "🛑 Servers stopped." -ForegroundColor Red
}

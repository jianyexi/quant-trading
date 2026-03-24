#!/usr/bin/env pwsh
<#
.SYNOPSIS
  Refresh-deploy quant-trading to an existing Azure VM via tarball + Docker rebuild.

.DESCRIPTION
  Reads VM connection info from data/azure-vm-info.json, packages the project,
  uploads via SCP, and runs docker compose up --build.

.PARAMETER SkipUpload
  Skip tarball creation and upload — only run docker compose rebuild on VM.

.PARAMETER VMHost
  Override VM host (IP or FQDN). Defaults to value from azure-vm-info.json.

.PARAMETER VMUser
  Override SSH user. Defaults to value from azure-vm-info.json.
#>

param(
    [switch]$SkipUpload,
    [string]$VMHost,
    [string]$VMUser
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

# ── Load VM info ─────────────────────────────────────────────────────
$infoPath = "data\azure-vm-info.json"
if (Test-Path $infoPath) {
    $vmInfo = Get-Content $infoPath | ConvertFrom-Json
    if (-not $VMHost) { $VMHost = $vmInfo.PublicIP }
    if (-not $VMUser) { $VMUser = $vmInfo.AdminUser }
}

if (-not $VMHost -or -not $VMUser) {
    Write-Host "[ERROR] VM connection info not found. Provide -VMHost and -VMUser, or ensure data/azure-vm-info.json exists." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[DEPLOY] Refreshing quant-trading on $VMUser@$VMHost" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Package & Upload ────────────────────────────────────────
if (-not $SkipUpload) {
    Write-Host "[1/3] Packaging project..." -ForegroundColor Green

    $gitBash = "C:\Program Files\Git\bin\bash.exe"
    $tarPath = "$env:TEMP\quant-deploy.tar.gz"

    if (Test-Path $gitBash) {
        & $gitBash -c "cd '$(Get-Location)' && tar czf '$($tarPath -replace '\\','/')' --exclude='target' --exclude='.git' --exclude='node_modules' --exclude='web/dist' --exclude='*.model' --exclude='__pycache__' --exclude='logs/*' --exclude='data/*.db' ."
    } else {
        tar -czf $tarPath --exclude='target' --exclude='.git' --exclude='node_modules' --exclude='web/dist' --exclude='*.model' --exclude='__pycache__' --exclude='logs/*' --exclude='data/*.db' .
    }

    $sizeMB = [math]::Round((Get-Item $tarPath).Length / 1MB, 1)
    Write-Host "  Tarball: $sizeMB MB" -ForegroundColor DarkGray

    Write-Host "[2/3] Uploading to VM..." -ForegroundColor Green
    scp -o StrictHostKeyChecking=no -o ConnectTimeout=15 $tarPath "${VMUser}@${VMHost}:~/quant-deploy.tar.gz"
    Remove-Item -Force $tarPath
    Write-Host "  Upload complete." -ForegroundColor DarkGray
} else {
    Write-Host "[1/3] Skipping upload (--SkipUpload)" -ForegroundColor Yellow
    Write-Host "[2/3] Skipping upload" -ForegroundColor Yellow
}

# ── Step 2: Rebuild on VM ───────────────────────────────────────────
Write-Host "[3/3] Rebuilding on VM..." -ForegroundColor Green

$remoteScript = @'
set -euo pipefail

cd ~/quant-trading

if [ -f ~/quant-deploy.tar.gz ]; then
    echo ">>> Extracting fresh code..."
    tar xzf ~/quant-deploy.tar.gz
    rm ~/quant-deploy.tar.gz
    echo ">>> Code updated."
fi

echo ">>> Stopping current containers..."
sudo docker compose down --timeout 30

echo ">>> Rebuilding and starting (this may take several minutes)..."
sudo docker compose up -d --build

echo ">>> Waiting for health check..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:8080/api/health >/dev/null 2>&1; then
        echo ">>> ✅ Application is healthy!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo ">>> ⚠️  Health check timed out after 10 minutes"
        sudo docker compose logs --tail=50
    fi
    sleep 10
done

echo ""
echo ">>> Container status:"
sudo docker compose ps
'@

ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 "${VMUser}@${VMHost}" "bash -s" @"
$remoteScript
"@

# ── Summary ──────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "[OK] Deployment refreshed!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Dashboard:  http://${VMHost}:8080"
Write-Host "  SSH:        ssh ${VMUser}@${VMHost}"
Write-Host "  Logs:       ssh ${VMUser}@${VMHost} 'cd quant-trading && sudo docker compose logs -f'"
Write-Host ""

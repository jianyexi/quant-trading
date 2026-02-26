<#
.SYNOPSIS
  Create an Azure VM in East Asia and deploy quant-trading via Docker.

.DESCRIPTION
  Prerequisites:
    1. Azure CLI installed: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli
    2. Logged in: az login
    3. Run from the repo root: .\scripts\deploy-azure.ps1

.NOTES
  VM: Standard_B2ms (2 vCPU, 8 GB RAM, ~$60/mo)
  Region: East Asia (Hong Kong)
#>

$ErrorActionPreference = "Stop"

# ── Configuration ────────────────────────────────────────────────────
$ResourceGroup = "rg-quant-trading"
$Location      = "eastasia"
$VmName        = "quant-trading-vm"
$VmSize        = "Standard_B2ms"
$VmImage       = "Canonical:ubuntu-24_04-lts:server:latest"
$AdminUser     = "quant"
$DiskSizeGB    = 64
$DnsLabel      = "quant-$(-join ((48..57)+(97..102) | Get-Random -Count 8 | ForEach-Object {[char]$_}))"

# ── Preflight ────────────────────────────────────────────────────────
Write-Host ""
if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Azure CLI not found. Install: https://aka.ms/install-azure-cli" -ForegroundColor Red
    exit 1
}
try { az account show 2>$null | Out-Null } catch {
    Write-Host "[ERROR] Not logged in. Run: az login" -ForegroundColor Red
    exit 1
}

# Generate SSH key if needed
$SshKeyPath = "$env:USERPROFILE\.ssh\id_rsa.pub"
if (-not (Test-Path $SshKeyPath)) {
    Write-Host "[WARN] SSH key not found, generating..." -ForegroundColor Yellow
    ssh-keygen -t rsa -b 4096 -f "$env:USERPROFILE\.ssh\id_rsa" -N '""' -q
}

Write-Host "[INFO] Deploying quant-trading to Azure East Asia" -ForegroundColor Green
Write-Host "  Resource Group: $ResourceGroup"
Write-Host "  VM: $VmName ($VmSize)"
Write-Host ""

# ── Step 1: Resource Group ───────────────────────────────────────────
Write-Host "[INFO] Creating resource group..." -ForegroundColor Green
az group create --name $ResourceGroup --location $Location --output none

# ── Step 2: Create VM ───────────────────────────────────────────────
Write-Host "[INFO] Creating VM (1-3 minutes)..." -ForegroundColor Green
$vmJson = az vm create `
    --resource-group $ResourceGroup `
    --name $VmName `
    --image $VmImage `
    --size $VmSize `
    --admin-username $AdminUser `
    --ssh-key-value $SshKeyPath `
    --os-disk-size-gb $DiskSizeGB `
    --public-ip-address-dns-name $DnsLabel `
    --output json

$vmObj = $vmJson | ConvertFrom-Json
$PublicIP = $vmObj.publicIpAddress
$FQDN = "$DnsLabel.$Location.cloudapp.azure.com"
Write-Host "[INFO] VM created! IP: $PublicIP  FQDN: $FQDN" -ForegroundColor Green

# ── Step 3: Open Port 8080 ──────────────────────────────────────────
Write-Host "[INFO] Opening port 8080..." -ForegroundColor Green
az vm open-port `
    --resource-group $ResourceGroup `
    --name $VmName `
    --port 8080 `
    --priority 1010 `
    --output none

# ── Step 4: Copy project to VM ──────────────────────────────────────
Write-Host "[INFO] Waiting for SSH..." -ForegroundColor Green
for ($i = 0; $i -lt 30; $i++) {
    $result = ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$AdminUser@$PublicIP" "echo ok" 2>$null
    if ($result -eq "ok") { break }
    Start-Sleep 5
}

Write-Host "[INFO] Copying project to VM..." -ForegroundColor Green

# Use tar via Git Bash (available on Windows with Git)
$gitBash = "C:\Program Files\Git\bin\bash.exe"
if (Test-Path $gitBash) {
    & $gitBash -c "cd '$(Get-Location)' && tar czf /tmp/quant-deploy.tar.gz --exclude='target' --exclude='.git' --exclude='node_modules' --exclude='web/dist' --exclude='*.model' --exclude='__pycache__' ."
    scp -o StrictHostKeyChecking=no /tmp/quant-deploy.tar.gz "${AdminUser}@${PublicIP}:~/"
} else {
    # Fallback: use PowerShell Compress-Archive (less efficient but works)
    Write-Host "[INFO] Git Bash not found, using rsync-like copy..." -ForegroundColor Yellow
    $excludes = @('target', '.git', 'node_modules', 'web\dist', '__pycache__')
    $tempDir = "$env:TEMP\quant-deploy"
    if (Test-Path $tempDir) { Remove-Item -Recurse -Force $tempDir }
    
    # Copy files, excluding large dirs
    robocopy . $tempDir /E /XD target .git node_modules __pycache__ /XF *.model /NFL /NDL /NJH /NJS /NC /NS | Out-Null
    
    $tarPath = "$env:TEMP\quant-deploy.tar.gz"
    tar -czf $tarPath -C $tempDir .
    scp -o StrictHostKeyChecking=no $tarPath "${AdminUser}@${PublicIP}:~/quant-trading-deploy.tar.gz"
    Remove-Item -Recurse -Force $tempDir
    Remove-Item -Force $tarPath
}

# ── Step 5: Install Docker & Deploy ─────────────────────────────────
Write-Host "[INFO] Installing Docker and deploying (5-15 min for first build)..." -ForegroundColor Green

$remoteScript = @'
set -euo pipefail

echo ">>> Installing Docker..."
sudo apt-get update -qq
sudo apt-get install -y -qq ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update -qq
sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker $USER

echo ">>> Extracting project..."
mkdir -p ~/quant-trading
cd ~/quant-trading
tar xzf ~/quant-trading-deploy.tar.gz 2>/dev/null || tar xzf ~/quant-deploy.tar.gz 2>/dev/null || true
rm -f ~/quant-trading-deploy.tar.gz ~/quant-deploy.tar.gz

mkdir -p data

echo ">>> Building and starting services..."
sudo docker compose up -d --build

echo ">>> Waiting for health check..."
for i in $(seq 1 60); do
  if curl -sf http://localhost:8080/api/health >/dev/null 2>&1; then
    echo ">>> Application is healthy!"
    break
  fi
  sleep 10
done

sudo docker compose ps
'@

ssh -o StrictHostKeyChecking=no "$AdminUser@$PublicIP" "bash -s" @"
$remoteScript
"@

# ── Summary ──────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "[INFO] Deployment complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Dashboard:  http://${FQDN}:8080"
Write-Host "  Public IP:  http://${PublicIP}:8080"
Write-Host "  SSH:        ssh $AdminUser@$PublicIP"
Write-Host ""
Write-Host "  Useful commands:"
Write-Host "    ssh $AdminUser@$PublicIP 'cd quant-trading && sudo docker compose logs -f'"
Write-Host "    ssh $AdminUser@$PublicIP 'cd quant-trading && sudo docker compose restart'"
Write-Host ""
Write-Host "  To tear down:"
Write-Host "    az group delete --name $ResourceGroup --yes --no-wait"
Write-Host ""

# Save connection info
$info = @{
    ResourceGroup = $ResourceGroup
    VmName = $VmName
    PublicIP = $PublicIP
    FQDN = $FQDN
    AdminUser = $AdminUser
    DashboardUrl = "http://${FQDN}:8080"
}
$info | ConvertTo-Json | Set-Content "data\azure-vm-info.json"
Write-Host "[INFO] Connection info saved to data\azure-vm-info.json" -ForegroundColor Green

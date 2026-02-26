#!/usr/bin/env bash
# ============================================================================
# deploy-azure.sh — Create Azure VM in East Asia & deploy quant-trading
#
# Prerequisites:
#   1. Azure CLI installed: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli
#   2. Logged in: az login
#   3. Run from the repo root: bash scripts/deploy-azure.sh
#
# This script will:
#   1. Create a resource group in East Asia
#   2. Create a VM (Ubuntu 24.04 LTS, Standard_B2ms)
#   3. Open ports 8080 (app) and 22 (SSH)
#   4. Copy the project to the VM
#   5. Install Docker and deploy with docker-compose
# ============================================================================
set -euo pipefail

# ── Configuration (edit as needed) ──────────────────────────────────
RESOURCE_GROUP="rg-quant-trading"
LOCATION="eastasia"
VM_NAME="quant-trading-vm"
VM_SIZE="Standard_B2ms"         # 2 vCPU, 8 GB RAM — ~$60/mo
VM_IMAGE="Canonical:ubuntu-24_04-lts:server:latest"
ADMIN_USER="quant"
SSH_KEY_PATH="$HOME/.ssh/id_rsa.pub"
DISK_SIZE_GB=64
DNS_LABEL="quant-$(openssl rand -hex 4)"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ── Preflight checks ───────────────────────────────────────────────
command -v az >/dev/null 2>&1 || error "Azure CLI not found. Install: https://aka.ms/install-azure-cli"
az account show >/dev/null 2>&1 || error "Not logged in. Run: az login"

if [ ! -f "$SSH_KEY_PATH" ]; then
  warn "SSH key not found at $SSH_KEY_PATH, generating one..."
  ssh-keygen -t rsa -b 4096 -f "${SSH_KEY_PATH%.pub}" -N "" -q
fi

info "Deploying to Azure East Asia (Hong Kong)"
info "  Resource Group: $RESOURCE_GROUP"
info "  VM: $VM_NAME ($VM_SIZE)"
echo ""

# ── Step 1: Resource Group ──────────────────────────────────────────
info "Creating resource group..."
az group create \
  --name "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --output none

# ── Step 2: Create VM ──────────────────────────────────────────────
info "Creating VM (this takes 1-3 minutes)..."
VM_OUTPUT=$(az vm create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME" \
  --image "$VM_IMAGE" \
  --size "$VM_SIZE" \
  --admin-username "$ADMIN_USER" \
  --ssh-key-value "$SSH_KEY_PATH" \
  --os-disk-size-gb "$DISK_SIZE_GB" \
  --public-ip-address-dns-name "$DNS_LABEL" \
  --output json)

PUBLIC_IP=$(echo "$VM_OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['publicIpAddress'])")
FQDN="$DNS_LABEL.$LOCATION.cloudapp.azure.com"
info "VM created! IP: $PUBLIC_IP  FQDN: $FQDN"

# ── Step 3: Open Ports ─────────────────────────────────────────────
info "Opening port 8080 for web dashboard..."
az vm open-port \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME" \
  --port 8080 \
  --priority 1010 \
  --output none

# ── Step 4: Copy project to VM ─────────────────────────────────────
info "Waiting for VM SSH to be ready..."
for i in $(seq 1 30); do
  if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$ADMIN_USER@$PUBLIC_IP" "echo ok" 2>/dev/null; then
    break
  fi
  sleep 5
done

info "Copying project files to VM..."
# Create a tarball excluding large/unnecessary files
tar czf /tmp/quant-trading-deploy.tar.gz \
  --exclude='target' \
  --exclude='.git' \
  --exclude='node_modules' \
  --exclude='web/dist' \
  --exclude='*.model' \
  --exclude='__pycache__' \
  .

scp -o StrictHostKeyChecking=no /tmp/quant-trading-deploy.tar.gz "$ADMIN_USER@$PUBLIC_IP:~/"
rm /tmp/quant-trading-deploy.tar.gz

# ── Step 5: Install Docker & Deploy ────────────────────────────────
info "Installing Docker and deploying on VM..."
ssh -o StrictHostKeyChecking=no "$ADMIN_USER@$PUBLIC_IP" 'bash -s' << 'REMOTE_SCRIPT'
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
tar xzf ~/quant-trading-deploy.tar.gz
rm ~/quant-trading-deploy.tar.gz

echo ">>> Creating data directory..."
mkdir -p data

echo ">>> Building and starting services (this takes 5-15 minutes for first build)..."
sudo docker compose up -d --build

echo ">>> Waiting for health check..."
for i in $(seq 1 60); do
  if curl -sf http://localhost:8080/api/health >/dev/null 2>&1; then
    echo ">>> Application is healthy!"
    break
  fi
  sleep 10
done

echo ">>> Done! Services:"
sudo docker compose ps
REMOTE_SCRIPT

# ── Summary ─────────────────────────────────────────────────────────
echo ""
echo "============================================================"
info "🎉 Deployment complete!"
echo "============================================================"
echo ""
echo "  Dashboard:  http://$FQDN:8080"
echo "  Public IP:  http://$PUBLIC_IP:8080"
echo "  SSH:        ssh $ADMIN_USER@$PUBLIC_IP"
echo ""
echo "  Useful commands:"
echo "    ssh $ADMIN_USER@$PUBLIC_IP 'cd quant-trading && sudo docker compose logs -f'"
echo "    ssh $ADMIN_USER@$PUBLIC_IP 'cd quant-trading && sudo docker compose restart'"
echo ""
echo "  To tear down:"
echo "    az group delete --name $RESOURCE_GROUP --yes --no-wait"
echo ""

#!/bin/bash

# Make sure the repository is cloned
# sudo apt-get -y update
# sudo apt install -y git
# git clone https://github.com/VukGlisovic/computer_vision.git

# Install libraries required for OpenCV
sudo apt-get install -y htop libxcb1 libgl1

# Install Pixi
curl -LsSf https://raw.githubusercontent.com/Pixi-Lang/pixi/main/install.sh | bash
echo "Pixi installed with version: $(pixi --version)"

# Install python environment
cd computer_vision/hdr_image_creation
pixi install

# Setup NordVPN meshnet (other options also possible like Tailscale)
sh <(curl -sSf https://downloads.nordcdn.com/apps/linux/install.sh)
sudo usermod -aG nordvpn ${USER}
sudo nordvpn login --token ${NORDVPN_TOKEN}
sudo nordvpn set meshnet on
sudo nordvpn meshnet peer list

# Register systemd service to start the web application on boot
sudo tee /etc/systemd/system/hdr-app.service > /dev/null <<EOF
[Unit]
Description=HDR Image Creation Web Application
After=network-online.target nordvpnd.service
Wants=network-online.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${HOME}/computer_vision/hdr_image_creation
ExecStart=${HOME}/.pixi/bin/pixi run app
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now hdr-app.service


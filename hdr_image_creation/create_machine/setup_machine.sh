#!/bin/bash

# Install git
sudo apt-get update
sudo apt install -y git htop
sudo apt-get install -y libxcb1 libgl1

# Install Pixi
curl -LsSf https://raw.githubusercontent.com/Pixi-Lang/pixi/main/install.sh | bash
echo "Pixi installed with version: $(pixi --version)"

# Clone repository
git clone https://github.com/VukGlisovic/computer_vision.git
echo "Repository 'computer_vision' cloned"

# Install python environment
cd computer_vision/hdr_image_creation
pixi install

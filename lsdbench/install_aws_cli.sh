#!/bin/bash

# Create necessary directories
mkdir -p ~/bin
mkdir -p ~/aws-cli

# Download AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"

# Extract files
python -m zipfile -e awscliv2.zip .

# Install AWS CLI
./aws/install --bin-dir ~/bin --install-dir ~/aws-cli

# Configure environment variables
if ! grep -q "export PATH=~/bin:\$PATH" ~/.bashrc; then
    echo 'export PATH=~/bin:$PATH' >> ~/.bashrc
fi

if [ -f ~/.zshrc ]; then
    if ! grep -q "export PATH=~/bin:\$PATH" ~/.zshrc; then
        echo 'export PATH=~/bin:$PATH' >> ~/.zshrc
    fi
fi

# Clean up temporary files
rm -rf awscliv2.zip aws 

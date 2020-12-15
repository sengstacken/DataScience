#!/bin/bash

set -e

# OVERVIEW
# This script installs a custom, persistent installation of conda on the Notebook Instance's EBS volume, and ensures
# that these custom environments are available as kernels in Jupyter.
# 
# The on-create script downloads and installs a custom conda installation to the EBS volume via Miniconda. Any relevant
# packages can be installed here.
#   1. ipykernel is installed to ensure that the custom environment can be used as a Jupyter kernel   
#   2. Ensure the Notebook Instance has internet connectivity to download the Miniconda installer


sudo -u ec2-user -i <<'EOF'
unset SUDO_UID

# Install a separate conda installation via Miniconda
echo "Installing a separate conda install"
WORKING_DIR=/home/ec2-user/SageMaker/custom-miniconda
mkdir -p "$WORKING_DIR"
wget https://repo.anaconda.com/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh -O "$WORKING_DIR/miniconda.sh"
bash "$WORKING_DIR/miniconda.sh" -b -u -p "$WORKING_DIR/miniconda" 
rm -rf "$WORKING_DIR/miniconda.sh"


# Create a custom conda environment
echo "Creating a custom conda env"
wget https://raw.githubusercontent.com/sengstacken/DataScience/master/test.yml -O "$WORKING_DIR/env.yml"
source "$WORKING_DIR/miniconda/bin/activate"

conda env create -f "$WORKING_DIR/env.yml"
conda activate new_earthmlnewtest

echo "Installing Ipykernel"
pip install --quiet ipykernel

# Customize these lines as necessary to install the required packages
# echo "Adding custom packages via conda and pip"
# conda install --yes numpy=1.18.1
# pip install --quiet boto3==1.15.18

EOF

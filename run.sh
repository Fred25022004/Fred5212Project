#!/bin/bash
# Activate the conda environment
source activate fred-pytorch-env

# Run the EDA script
python src/eda.py

# run train script
python src/train.py
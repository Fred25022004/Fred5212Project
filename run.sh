#!/bin/bash
# Activate the conda environment
source activate fred-pytorch-env

# Run the EDA script
python src/eda.py

# Run the model training script
python src/train_model.py

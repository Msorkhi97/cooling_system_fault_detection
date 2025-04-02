"""
Main script for training the Cooling System Fault Diagnosis model.

This script:
- Loads configuration settings from `configs/config.yaml`
- Prepares the training dataset (or loads it if already processed)
- Initializes and trains the model

Author: [Msorkhi97@gmail.com]
"""

# Standard Library Imports
from pathlib import Path

# Third-Party Library Imports
import yaml
import numpy as np

# Local Application Imports
from src.preprocessing import BuildDataSet
from src.training import TrainModel

# Load Configuration
with open("configs/config.yaml","r") as file:
        config = yaml.safe_load(file)

# Prepare Training Data
if not Path("outputs/train_data.npz").exists():
    print("### Generating training dataset...")
    x_train, y_train = BuildDataSet(config).split_data()
else:
    print("### Loading preprocessed training dataset...")
    train_data = np.load("outputs/train_data.npz")
    x_train, y_train = train_data["x_train"],train_data["y_train"]

print(f"### Training data shape: x_train = {x_train.shape}, y_train = {y_train.shape}")

# Train the Model
trainer = TrainModel(config, x_train, y_train)
trainer.train()
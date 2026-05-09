#!/bin/bash

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Clone the repo and switch to SM branch
git clone https://github.com/BirdWithDreams/atml-project-laplacianformer
cd atml-project-laplacianformer
git checkout SM

# Install dependencies
uv venv --python 3.11 .venv
source .venv/bin/activate
uv sync

# Set your wandb key
export WANDB_API_KEY=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

# Task 2 - Text Classification
python3 train.py task=nlp_classification model=laplacian datamodule=sst2
python3 train.py task=nlp_classification model=vanilla datamodule=sst2
python3 train.py task=nlp_classification model=laplacian datamodule=agnews
python3 train.py task=nlp_classification model=vanilla datamodule=agnews

# Task 1 - Object Detection
python3 train.py task=detection model=laplacian datamodule=voc
python3 train.py task=detection model=vanilla datamodule=voc

echo "All experiments done!"

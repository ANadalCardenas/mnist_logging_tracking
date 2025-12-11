# MNIST Logging & Tracking Project

This project provides a clean and modular training pipeline for MNIST classification and reconstruction tasks, with full support for experiment tracking via **Weights & Biases (WandB)** and **TensorBoard**.  
It is designed as a template for machine learning workflows where **reproducibility, logging, and experiment comparison** are important.

---

## Features

###  MNIST Classification & Reconstruction
- `run_classification.py`: trains a digit classifier  
- `run_reconstruction.py`: trains an autoencoder to reconstruct images  

###  Integrated Logging
Custom logger (`logger.py`) that:
- Logs to console + file
- Uses timestamps and consistent formatting
- Helps debugging and experiment traceability

###  WandB Tracking
The module `wandb.py` enables:
- Metric logging (accuracy, loss…)
- Uploading images (e.g., confusion matrices)
- Configuration tracking
- Storing run summaries

## Modular Codebase

model.py: contains neural networks for MNIST

utils.py: dataset handling, plotting, seed control…

main.py: orchestrator example for starting a training run

## Installation
1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

2. Install requirements
pip install -r requirements.txt

3. (Optional) Login to WandB

If using Weights & Biases:

wandb login

## Usage
### Run a classification experiment
python run_classification.py

### Run a reconstruction experiment
python run_reconstruction.py

### Launch TensorBoard
tensorboard --logdir runs/

## Project Structure
mnist_logging_tracking/
├── main.py
├── run_classification.py
├── run_reconstruction.py
├── model.py
├── logger.py
├── utils.py
├── wandb.py
├── tensorboard.py
├── requirements.txt
├── data/ (you must create this folder, dowload and save the MNIST dataset. https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

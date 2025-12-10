import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt

from logger import Logger
from datetime import datetime
from utils import TaskType
from typing import Optional


class WandbLogger(Logger):

    def __init__(
        self, 
        task: TaskType, 
        model: nn.Module,
    ):
        wandb.login()
        wandb.init(project="hands-on-monitoring")
        wandb.run.name = f'{task}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

        
        for name, param in model.named_parameters():
         if "weight" in name or "bias" in name:
            wandb.log({f"Weights/{name}": wandb.Histogram(param.detach().cpu())}, step=100)

    def log_reconstruction_training(
        self, 
        model: nn.Module, 
        epoch: int, 
        train_loss_avg: np.ndarray,
        val_loss_avg: np.ndarray,
        reconstruction_grid: Optional[torch.Tensor] = None,
    ):

       
        wandb.log({"Reconstruction/train_loss": train_loss_avg}, step=epoch)
        wandb.log({"Reconstruction/val_loss": val_loss_avg}, step=epoch)
        wandb.log({"Reconstruction/images": wandb.Image(reconstruction_grid)}, step=epoch)

        pass


    def log_classification_training(
        self, 
        epoch: int,
        train_loss_avg: float, 
        val_loss_avg: float, 
        train_acc_avg: float, 
        val_acc_avg: float, 
        fig: plt.Figure,
    ):
        
        wandb.log({"Classification/confusion_matrix": wandb.Image(fig)}, step=epoch)
        wandb.log({"Classification/val_loss": val_loss_avg}, step=epoch)
        wandb.log({"Classification/val_acc": val_acc_avg}, step=epoch)
        wandb.log({"Classification/train_loss": train_loss_avg}, step=epoch)
        wandb.log({"Classification/train_acc": train_acc_avg}, step=epoch)

        pass

    def log_embeddings(
        self, 
        model: nn.Module, 
        train_loader: torch.utils.data.DataLoader,
    ):
        out = model.encoder.linear.out_features
        columns = np.arange(out).astype(str).tolist()
        columns.insert(0, "target")
        columns.insert(0, "image")

        list_dfs = []

        for i in range(3): # take only 3 batches of data for plotting
            images, labels = next(iter(train_loader))

            for img, label in zip(images, labels):
                # forward img through the encoder
                image = wandb.Image(img)
                label = label.item()
                latent = model.encoder(img.unsqueeze(dim=0)).squeeze().detach().cpu().numpy().tolist()
                data = [image, label, *latent]

                df = pd.DataFrame([data], columns=columns)
                list_dfs.append(df)
        embeddings = pd.concat(list_dfs, ignore_index=True)

        table = wandb.Table(dataframe=embeddings)
        wandb.log({"Embeddings/latent": table})



    def log_model_graph(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
    ):
        # Wandb does not support logging the model graph
        pass
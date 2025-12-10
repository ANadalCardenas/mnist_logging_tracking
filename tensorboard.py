import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from typing import Optional
from logger import Logger
from utils import TaskType
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger(Logger):

    def __init__(
        self, 
        task: TaskType, 
    ):
        # Define the folder where we will store all the tensorboard logs
        logdir = os.path.join("logs", f"{task}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")

        # TODO: Initialize Tensorboard Writer with the previous folder 'logdir'
        self.writer = SummaryWriter(log_dir=logdir)


    def log_reconstruction_training(
        self, 
        model: nn.Module, 
        epoch: int, 
        train_loss_avg: np.ndarray,
        val_loss_avg: np.ndarray,
        reconstruction_grid: Optional[torch.Tensor] = None,
    ):

        
        self.writer.add_scalar("Reconstruction/train_loss", float(train_loss_avg), epoch)
        self.writer.add_scalar("Reconstruction/val_loss", float(val_loss_avg), epoch)
        self.writer.add_image("Reconstruction/images", reconstruction_grid, epoch)
        for name, weight in model.encoder.named_parameters():
            # Weight values histogram
            self.writer.add_histogram(f"{name}/value", weight.detach().cpu(), epoch)

            # Weight gradients histogram (if exists)
            if weight.grad is not None:
                self.writer.add_histogram(f"{name}/grad", weight.grad.detach().cpu(), epoch)


        pass



    def log_classification_training(
        self, 
        epoch: int,
        train_loss_avg: np.ndarray,
        val_loss_avg: np.ndarray,
        train_acc_avg: np.ndarray,
        val_acc_avg: np.ndarray,
        fig: plt.Figure,
    ):
        
        self.writer.add_figure("Classification/confusion_matrix", fig, epoch)
        self.writer.add_scalar("Classification/val_loss", float(val_loss_avg), epoch)
        self.writer.add_scalar("Classification/val_acc", float(val_acc_avg), epoch)
        self.writer.add_scalar("Classification/train_loss", float(train_loss_avg), epoch)
        self.writer.add_scalar("Classification/train_acc", float(train_acc_avg), epoch)


        pass


    def log_model_graph(
        self, 
        model: nn.Module, 
        train_loader: torch.utils.data.DataLoader,
    ):
        batch, _ = next(iter(train_loader))
             
        try:
            self.writer.add_graph(model, batch)
        except Exception as e:
            print(f" Could not log model graph: {e}")



    def log_embeddings(
        self, 
        model: nn.Module, 
        train_loader: torch.utils.data.DataLoader,
    ):
        list_latent = []
        list_images = []
        for i in range(10):
            batch, _ = next(iter(train_loader))

            # forward batch through the encoder
            list_latent.append(model.encoder(batch))
            list_images.append(batch)

        latent = torch.cat(list_latent)
        images = torch.cat(list_images)

        # images_2d = images.view(images.size(0), -1)
        self.writer.add_embedding(
            latent,
            metadata=None,
            label_img=images,
            tag="LatentSpace"
        )
        print("ℹEmbeddings logged")
        # Be patient! Projector logs can take a while


import os

from pathlib import Path
from dataclasses import asdict
import torch
import torch.nn.functional as F
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import logging

from vascular_superenhancement.training.model_zoo import (
    build_generator,
    build_discriminator,
)
from vascular_superenhancement.training.losses import (
    discriminator_loss,
    generator_gan_loss,
    generator_l1_loss,
)
from vascular_superenhancement.training.datasets import build_subjects_dataset
from vascular_superenhancement.training.transforms import build_transforms
from vascular_superenhancement.training.dataloading import build_train_loader
from vascular_superenhancement.utils.path_config import load_path_config

logger = logging.getLogger(__name__)

@hydra.main(
    version_base="1.1",
    config_path=str((Path(__file__).resolve().parents[3] / "hydra_configs").as_posix()),
    config_name="config"
)
def train_model(cfg: DictConfig):
    logger.info(cfg.keys())
    logger.info(cfg.model.generator)
    logger.info(cfg.model.discriminator)
    logger.info(cfg.train)
    logger.info(cfg.data)
    path_config = load_path_config(cfg.path_config.path_config_name)
    
    logger.info(cfg.path_config.keys())
    logger.info(path_config)
    logger.info(path_config.working_dir)
    
    logger.info(os.getcwd())
    
    
    # # 1. get gpu
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # # 2. build datasets and dataloaders
    # preprocessing_transforms = build_transforms(cfg.data)
    # train_dataset = build_subjects_dataset(
    #     "train",
    #     path_config.splits_path,
    #     path_config,
    #     transforms=preprocessing_transforms,
    # )
    
if __name__ == "__main__":
    train_model()



# @hydra.main(config_path="hydra_configs", config_name="config")
# def train(cfg: DictConfig):
#     # Load path config using existing function
#     path_config = load_path_config(cfg.path_config_name)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Build models
#     G = build_generator(cfg).to(device)
#     D = build_discriminator(cfg).to(device)

#     # Build datasets and dataloader
#     preprocessing_transforms = build_transforms(cfg.data)
#     train_dataset = build_subjects_dataset(
#         "train",
#         path_config.splits_path,
#         path_config,
#         transforms=preprocessing_transforms,
#     )
#     train_loader = build_train_loader(train_dataset, cfg.train)

#     # Optimizers
#     optimizer_G = torch.optim.Adam(G.parameters(), lr=cfg.train.lr, betas=(0.5, 0.999))
#     optimizer_D = torch.optim.Adam(D.parameters(), lr=cfg.train.lr, betas=(0.5, 0.999))

#     for epoch in range(cfg.train.num_epochs):
#         G.train()
#         D.train()
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.train.num_epochs}")

#         for batch in pbar:
#             # Calculate speed from velocity components
#             speed = torch.sqrt(
#                 batch["fvx"].data ** 2 + batch["fvy"].data ** 2 + batch["fvz"].data ** 2
#             )
#             input = speed.to(device)
#             target = batch["cine"].data.to(device)

#             # Train Discriminator
#             optimizer_D.zero_grad()
#             real_pred = D(target)
#             fake_img = G(input).detach()
#             fake_pred = D(fake_img)
#             loss_D = discriminator_loss(real_pred, fake_pred)
#             loss_D.backward()
#             optimizer_D.step()

#             # Train Generator
#             optimizer_G.zero_grad()
#             fake_img = G(input)
#             fake_pred = D(fake_img)
#             loss_G_GAN = generator_gan_loss(fake_pred)
#             loss_G_L1 = generator_l1_loss(fake_img, target, weight=cfg.train.lambda_l1)
#             loss_G = loss_G_GAN + loss_G_L1
#             loss_G.backward()
#             optimizer_G.step()

#             pbar.set_postfix({"loss_D": loss_D.item(), "loss_G": loss_G.item()})

#         # TODO: Add validation, checkpointing, and logging here

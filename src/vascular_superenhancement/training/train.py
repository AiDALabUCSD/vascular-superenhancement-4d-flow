import torch
import torch.nn.functional as F
from tqdm import tqdm

from vascular_superenhancement.training.models import build_generator, build_discriminator
from vascular_superenhancement.training.losses import (
    discriminator_loss,
    generator_gan_loss,
    generator_l1_loss,
)
from vascular_superenhancement.datasets import build_subjects_dataset
from vascular_superenhancement.training.transforms import build_transforms
from vascular_superenhancement.training.dataloader import build_train_loader

def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build models
    G = build_generator(cfg).to(device)
    D = build_discriminator(cfg).to(device)

    # Build datasets and dataloader
    transforms = build_transforms(cfg)
    train_dataset = build_subjects_dataset("train", cfg.splits_path, cfg.path_config, transforms=transforms)
    train_loader = build_train_loader(train_dataset, cfg)

    # Optimizers
    optimizer_G = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=(0.5, 0.999))

    for epoch in range(cfg.num_epochs):
        G.train()
        D.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")

        for batch in pbar:
            input = torch.cat([batch["fvx"].data, batch["fvy"].data, batch["fvz"].data], dim=1).to(device)
            target = batch["cine"].data.to(device)

            # Train Discriminator
            optimizer_D.zero_grad()
            real_pred = D(target)
            fake_img = G(input).detach()
            fake_pred = D(fake_img)
            loss_D = discriminator_loss(real_pred, fake_pred)
            loss_D.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            fake_img = G(input)
            fake_pred = D(fake_img)
            loss_G_GAN = generator_gan_loss(fake_pred)
            loss_G_L1 = generator_l1_loss(fake_img, target, weight=cfg.lambda_l1)
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizer_G.step()

            pbar.set_postfix({"loss_D": loss_D.item(), "loss_G": loss_G.item()})

        # TODO: Add validation, checkpointing, and logging here

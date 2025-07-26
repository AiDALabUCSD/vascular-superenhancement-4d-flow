import os
from pathlib import Path
# from dataclasses import asdict
import torch
import torchio as tio
# import torch.nn.functional as F
# from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import logging

from vascular_superenhancement.training.model_factory import (
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
    # Set up logging level based on debug flag
    if cfg.train.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    else:
        logging.getLogger().setLevel(logging.INFO)
        logger.setLevel(logging.INFO)
    
    logger.info("Setting up training...")
    logger.info(f"Current working directory: {Path(os.getcwd()).as_posix()}")
    
    # 0. print config
    # logger.info(f"Config: {cfg}")
    
    # 1. get gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        #print specs of device
        logger.info(f"{device.type} specs: {torch.cuda.get_device_properties(device)}")
    else:
        logger.info(f"Using {device.type}")
    
    # 2. load path config
    path_config = load_path_config(cfg.path_config.path_config_name)
    logger.info(f"Loaded path config: {path_config}")
    
    # # 3. build transforms
    training_transforms = build_transforms(cfg, train=True)
    validation_transforms = build_transforms(cfg, train=False)
    logger.info(f"Training transforms: {training_transforms}")
    logger.info(f"Validation transforms: {validation_transforms}")
        
    # 4. build datasets
    training_dataset = build_subjects_dataset(
        "train",
        Path(cfg.data.splits_path),
        cfg.path_config.path_config_name,
        transforms=training_transforms,
        debug=cfg.train.debug,  # Pass debug flag
    )
    validation_dataset = build_subjects_dataset(
        "validation",
        Path(cfg.data.splits_path),
        cfg.path_config.path_config_name,
        transforms=validation_transforms,
        debug=cfg.train.debug,  # Pass debug flag
    )

    logger.info(f"Training dataset length: {len(training_dataset)}")
    logger.info(f"Validation dataset length: {len(validation_dataset)}")
    
    # 5. build dataloaders
    training_loader = build_train_loader(training_dataset, cfg)
    validation_loader = build_train_loader(validation_dataset, cfg)
    
    # print number of batches in training and validation loader
    logger.info(f"Number of batches in training loader: {len(training_loader)}")
    logger.info(f"Number of batches in validation loader: {len(validation_loader)}")
    
    # 6. build models
    generator = build_generator(cfg).to(device)
    logger.info(f"Generator summary: {generator}")
    discriminator = build_discriminator(cfg).to(device)
    logger.info(f"Discriminator summary: {discriminator}")
    
    # 7. build optimizers
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=cfg.train.generator_lr, betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=cfg.train.discriminator_lr, betas=(0.5, 0.999))

    # 8. train
    for epoch in range(cfg.train.num_epochs):
        
        generator.train()
        discriminator.train()
        
        loss_discriminator_train = []
        loss_generator_gan_train = []
        loss_generator_l1_train = []
        loss_generator_train = []
        
        for i, batch in enumerate(training_loader):
            
            mag = batch["mag"][tio.DATA].to(device)
            fvx = batch["flow_vx"][tio.DATA].to(device)
            fvy = batch["flow_vy"][tio.DATA].to(device)
            fvz = batch["flow_vz"][tio.DATA].to(device)
            cine = batch["cine"][tio.DATA].to(device)
            # Calculate speed from velocity components
            speed = torch.sqrt(fvx ** 2 + fvy ** 2 + fvz ** 2)
            
            # base input
            input_base = torch.cat([mag, speed], dim=1)
            
            # construct real and fake inputs to discriminator
            input_to_discriminator_real = torch.cat([input_base, cine], dim=1)
            input_to_discriminator_fake = torch.cat([input_base, generator(input_base).detach()], dim=1)
            # check shapes of inputs to discriminator
            assert input_to_discriminator_real.shape == input_to_discriminator_fake.shape, "Real and fake discriminator inputs must have the same shape"
            assert input_to_discriminator_real.shape[1] == cfg.model.discriminator.in_channels, f"Config says {cfg.model.discriminator.in_channels} ch, real_D_input has {input_to_discriminator_real.shape[1]} ch"
            assert input_to_discriminator_fake.shape[1] == cfg.model.discriminator.in_channels, f"Config says {cfg.model.discriminator.in_channels} ch, fake_D_input has {input_to_discriminator_fake.shape[1]} ch"
            
            # train discriminator
            optimizer_discriminator.zero_grad()
            pred_from_discriminator_real = discriminator(input_to_discriminator_real)
            pred_from_discriminator_fake = discriminator(input_to_discriminator_fake)
            loss_discriminator = discriminator_loss(pred_from_discriminator_real, pred_from_discriminator_fake)
            loss_discriminator.backward()
            optimizer_discriminator.step()
            
            # freeze discriminator
            for param in discriminator.parameters():
                param.requires_grad_(False)
            
            # train generator
            optimizer_generator.zero_grad()
            pred_from_generator = generator(input_base)
            input_to_discriminator_for_generator = torch.cat([input_base, pred_from_generator], dim=1)
            pred_from_discriminator_for_generator = discriminator(input_to_discriminator_for_generator)
            loss_generator_gan = generator_gan_loss(pred_from_discriminator_for_generator)
            loss_generator_l1 = generator_l1_loss(pred_from_generator, cine, weight=cfg.train.lambda_l1)
            loss_generator = loss_generator_gan + loss_generator_l1
            loss_generator.backward()
            optimizer_generator.step()
            
            # log all losses formatted in a line
            logger.info(f"epoch {epoch}, batch {i}: loss_discriminator.item(): {loss_discriminator.item()}, loss_generator.item(): {loss_generator.item()}, loss_generator_gan.item(): {loss_generator_gan.item()}, loss_generator_l1.item(): {loss_generator_l1.item()}")
            
            # unfreeze discriminator
            for param in discriminator.parameters():
                param.requires_grad_(True)
        
        # 9. validation
        with torch.no_grad():
            generator.eval()
            discriminator.eval()
            
            loss_discriminator_val = []
            loss_generator_gan_val = []
            loss_generator_l1_val = []
            loss_generator_val = []
            
            for i, batch in enumerate(validation_loader):
                mag = batch["mag"][tio.DATA].to(device)
                fvx = batch["flow_vx"][tio.DATA].to(device)
                fvy = batch["flow_vy"][tio.DATA].to(device)
                fvz = batch["flow_vz"][tio.DATA].to(device)
                cine = batch["cine"][tio.DATA].to(device)
                logger.info(f"mag.shape: {mag.shape}, fvx.shape: {fvx.shape}, fvy.shape: {fvy.shape}, fvz.shape: {fvz.shape}, cine.shape: {cine.shape}")
                # calculate speed from velocity components
                speed = torch.sqrt(fvx ** 2 + fvy ** 2 + fvz ** 2)
                # base input
                input_base = torch.cat([mag, speed], dim=1)
                # construct real and fake inputs to discriminator
                input_to_discriminator_real = torch.cat([input_base, cine], dim=1)
                input_to_discriminator_fake = torch.cat([input_base, generator(input_base).detach()], dim=1)
                
                # get predictions from discriminator
                pred_from_discriminator_real = discriminator(input_to_discriminator_real)
                pred_from_discriminator_fake = discriminator(input_to_discriminator_fake)
                # get discriminator validation loss
                loss_discriminator = discriminator_loss(pred_from_discriminator_real, pred_from_discriminator_fake)
                loss_discriminator_val.append(loss_discriminator.item())
                
                # get predictions from generator
                pred_from_generator = generator(input_base)
                input_to_discriminator_for_generator = torch.cat([input_base, pred_from_generator], dim=1)
                pred_from_discriminator_for_generator = discriminator(input_to_discriminator_for_generator)
                # get gan validation losses
                loss_generator_gan_val.append(generator_gan_loss(pred_from_discriminator_for_generator).item())
                loss_generator_l1_val.append(generator_l1_loss(pred_from_generator, cine, weight=cfg.train.lambda_l1).item())
                loss_generator_val.append(loss_generator_gan_val[-1] + loss_generator_l1_val[-1])
                
            scalar_loss_discriminator_val = torch.tensor(loss_discriminator_val).mean()
            scalar_loss_generator_val = torch.tensor(loss_generator_val).mean()
            scalar_loss_generator_gan_val = torch.tensor(loss_generator_gan_val).mean()
            scalar_loss_generator_l1_val = torch.tensor(loss_generator_l1_val).mean()
            
            logger.info(f"epoch {epoch}: loss_discriminator_val: {scalar_loss_discriminator_val}, loss_generator_gan_val: {scalar_loss_generator_gan_val}, loss_generator_l1_val: {scalar_loss_generator_l1_val}, loss_generator_val: {scalar_loss_generator_val}")
            
            # 10. checkpoint
            if epoch % cfg.train.checkpoint_interval == 0:
                logger.info(f"Saving checkpoint for epoch {epoch}")
                checkpoint_dir = Path(os.getcwd()) / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
                
                checkpoint = {
                    "epoch": epoch,
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "optimizer_generator_state_dict": optimizer_generator.state_dict(),
                    "optimizer_discriminator_state_dict": optimizer_discriminator.state_dict(),
                    "loss_discriminator_val": scalar_loss_discriminator_val,
                    "loss_generator_val": scalar_loss_generator_val,
                    "loss_generator_gan_val": scalar_loss_generator_gan_val,
                    "loss_generator_l1_val": scalar_loss_generator_l1_val,
                }
                
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Checkpoint saved to {checkpoint_path}")  

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

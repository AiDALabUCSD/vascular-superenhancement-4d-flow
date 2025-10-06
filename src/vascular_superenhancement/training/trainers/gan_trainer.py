"""GAN trainer implementation for Pix2Pix-style training."""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torchio as tio
from omegaconf import DictConfig
import logging
from torchinfo import summary

from .base_trainer import BaseTrainer
from ..callbacks.base_callback import Callback
from ..model_factory import build_generator, build_discriminator
from ..losses import (
    discriminator_loss,
    generator_gan_loss,
    generator_l1_loss,
    generator_ssim_loss
)

logger = logging.getLogger(__name__)


class GanTrainer(BaseTrainer):
    """Trainer for GAN-based models (Pix2Pix, CycleGAN, etc.)"""
    
    def __init__(
        self,
        cfg: DictConfig,
        train_loader: tio.SubjectsLoader,
        train_dataset: Optional[tio.SubjectsDataset] = None,
        val_loader: Optional[tio.SubjectsLoader] = None,
        val_dataset: Optional[tio.SubjectsDataset] = None,
        test_loader: Optional[tio.SubjectsLoader] = None,
        test_dataset: Optional[tio.SubjectsDataset] = None,
        callbacks: Optional[List[Callback]] = None
    ):
        super().__init__(cfg=cfg,
                        train_loader=train_loader,
                        train_dataset=train_dataset,
                        val_loader=val_loader,
                        val_dataset=val_dataset,
                        test_loader=test_loader,
                        test_dataset=test_dataset,
                        callbacks=callbacks)
        
        # GAN-specific configuration
        self.lambda_disc = cfg.model.discriminator.weights.disc # Weight for discriminator loss
        self.lambda_l1 = cfg.model.generator.weights.l1 # Weight for L1 loss
        self.lambda_gan = cfg.model.generator.weights.gan # Weight for GAN loss
        self.lambda_ssim = cfg.model.generator.weights.ssim # Weight for SSIM loss
        self.disc_update_freq = cfg.train.get('disc_update_freq', 1)
        self.gen_update_freq = cfg.train.get('gen_update_freq', 1)
        
    def build_models(self) -> Dict[str, nn.Module]:
        """Build generator and discriminator models."""
        models = {
            'generator': build_generator(self.cfg),
            'discriminator': build_discriminator(self.cfg)
        }
        logger.info(f"Built generator with {sum(p.numel() for p in models['generator'].parameters())} parameters")
        logger.info(f"Generator summary: {models['generator']}")
        logger.info(f"Generator summary: {summary(models['generator'], input_size=(self.cfg.train.batch_size, self.cfg.model.generator.in_channels, self.cfg.train.patch_size[0], self.cfg.train.patch_size[1], self.cfg.train.patch_size[2]), depth=10)}")
        logger.info(f"Built discriminator with {sum(p.numel() for p in models['discriminator'].parameters())} parameters")
        logger.info(f"Discriminator summary: {summary(models['discriminator'], input_size=(self.cfg.train.batch_size, self.cfg.model.discriminator.in_channels, self.cfg.train.patch_size[0], self.cfg.train.patch_size[1], self.cfg.train.patch_size[2]), depth=10)}")
        logger.info(f"Discriminator summary: {models['discriminator']}")
        return models
    
    def build_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        """Build optimizers for generator and discriminator."""
        optimizers = {
            'generator': torch.optim.Adam(
                self.models['generator'].parameters(),
                lr=self.cfg.model.generator.optimizer.lr,
                betas=tuple(self.cfg.model.generator.optimizer.betas)
            ),
            'discriminator': torch.optim.Adam(
                self.models['discriminator'].parameters(),
                lr=self.cfg.model.discriminator.optimizer.lr,
                betas=tuple(self.cfg.model.discriminator.optimizer.betas)
            )
        }
        return optimizers
    
    # finished looking at this function
    def prepare_batch(self, batch: Any) -> Dict[str, torch.Tensor]:
        """Prepare batch data for training.
        
        Args:
            batch: Raw batch from dataloader
            
        Returns:
            Dictionary with prepared tensors
        """
        # Extract data and move to device
        mag = batch["mag"][tio.DATA].to(self.device)
        fvx = batch["flow_vx"][tio.DATA].to(self.device)
        fvy = batch["flow_vy"][tio.DATA].to(self.device)
        fvz = batch["flow_vz"][tio.DATA].to(self.device)
        cine = batch["cine"][tio.DATA].to(self.device)
        
        # Calculate speed from velocity components
        speed = torch.sqrt(fvx ** 2 + fvy ** 2 + fvz ** 2)
        
        # Prepare input
        input = torch.cat([mag, speed], dim=1)
        
        return {
            'input': input,
            'target': cine,
            'mag': mag,
            'fvx': fvx,
            'fvy': fvy,
            'fvz': fvz,
            'speed': speed,
            'batch_info': batch  # Keep original batch info for callbacks
        }
    
    # finished looking at this function
    def set_training_mode(self, mode: bool) -> None:
        """Set training/evaluation mode for models.
        
        For GANs, both generator and discriminator stay in train mode
        during training. Freezing is handled via requires_grad, not eval mode.
        
        Args:
            mode: True for training mode, False for evaluation mode
        """
        # During training, both models stay in train mode
        # During validation, both go to eval mode
        for model in self.models.values():
            model.train(mode)
    
    # finished looking at this function
    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        """Execute one training step for GAN.
        
        Note: Discriminator "freezing" is done via requires_grad=False,
        not by setting eval mode. Both models stay in train mode.
        
        Args:
            batch: Input batch from dataloader
            batch_idx: Index of current batch
            
        Returns:
            Dictionary containing losses and outputs
        """
        # Prepare batch
        data = self.prepare_batch(batch)
        input = data['input']
        target = data['target']
        
        # Generate fake images
        pred_g = self.models['generator'](input)
        
        outputs = {}
        
        # Train discriminator
        if self.global_step % self.disc_update_freq == 0:
            self.optimizers['discriminator'].zero_grad()
            
            # Prepare discriminator inputs
            input_d_real = torch.cat([input, target], dim=1)
            input_d_fake = torch.cat([input, pred_g.detach()], dim=1)
            
            assert input_d_real.shape == input_d_fake.shape, "Real and fake discriminator inputs must have the same shape"
            assert input_d_real.shape[1] == self.cfg.model.discriminator.in_channels, f"Config says {self.cfg.model.discriminator.in_channels} ch, real_d_input has {input_d_real.shape[1]} ch"
            assert input_d_fake.shape[1] == self.cfg.model.discriminator.in_channels, f"Config says {self.cfg.model.discriminator.in_channels} ch, fake_d_input has {input_d_fake.shape[1]} ch"
            
            # Get discriminator predictions
            pred_d_real = self.models['discriminator'](input_d_real)
            pred_d_fake = self.models['discriminator'](input_d_fake)
            
            # Calculate discriminator loss
            loss_d_unweighted = discriminator_loss(pred_d_real, pred_d_fake)
            loss_d = self.lambda_disc * loss_d_unweighted
            loss_d.backward()
            self.optimizers['discriminator'].step()
            
            outputs.update({
                'loss_discriminator': loss_d,
                'loss_discriminator_unweighted': loss_d_unweighted,
            })
        
        # Train generator
        if self.global_step % self.gen_update_freq == 0:
            # Freeze discriminator
            for param in self.models['discriminator'].parameters():
                param.requires_grad_(False)
            
            self.optimizers['generator'].zero_grad()
            
            # Get discriminator prediction for fake images

            input_d4g = torch.cat([input, pred_g], dim=1)
            pred_d4g = self.models['discriminator'](input_d4g)
            
            # Calculate generator losses
            loss_g_gan_unweighted = generator_gan_loss(pred_d4g)
            loss_g_l1_unweighted = generator_l1_loss(pred_g, target)
            loss_g_ssim_unweighted = generator_ssim_loss(pred_g, target)
            loss_g_total_unweighted = loss_g_gan_unweighted + loss_g_l1_unweighted + loss_g_ssim_unweighted
            
            loss_g_gan = self.lambda_gan * loss_g_gan_unweighted
            loss_g_l1 = self.lambda_l1 * loss_g_l1_unweighted
            loss_g_ssim = self.lambda_ssim * loss_g_ssim_unweighted
            loss_g_total = loss_g_gan + loss_g_l1 + loss_g_ssim
            
            loss_g_total.backward()
            self.optimizers['generator'].step()
            
            # Unfreeze discriminator
            for param in self.models['discriminator'].parameters():
                param.requires_grad_(True)
                
            # logger.info(f"e {self.current_epoch:04d}, b {batch_idx:04d}, g {self.global_step:04d}: d {loss_d.item():.4f}, g_gan {loss_g_gan.item():.4f}, g_l1 {loss_g_l1.item():.4f}, g_ssim {loss_g_ssim.item():.4f}, g {loss_g_total.item():.4f}")
            
            # Build log message with available losses
            log_msg = f"e {self.current_epoch:04d}, b {batch_idx:04d}, g {self.global_step:04d}:"
            if 'loss_discriminator' in outputs:
                log_msg += f" d {outputs['loss_discriminator'].item():.4f},"
            log_msg += f" g_gan {loss_g_gan.item():.4f}, g_l1 {loss_g_l1.item():.4f}, g_ssim {loss_g_ssim.item():.4f}, g {loss_g_total.item():.4f}"
            logger.info(log_msg)
            
            outputs.update({
                'loss_generator': loss_g_total,
                'loss_generator_gan': loss_g_gan,
                'loss_generator_l1': loss_g_l1,
                'loss_generator_ssim': loss_g_ssim,
                'loss_generator_unweighted': loss_g_total_unweighted,
                'loss_generator_gan_unweighted': loss_g_gan_unweighted,
                'loss_generator_l1_unweighted': loss_g_l1_unweighted,
                'loss_generator_ssim_unweighted': loss_g_ssim_unweighted,
            })
        
        # Add generated image for visualization callbacks
        outputs['generated'] = pred_g.detach()
        outputs['input'] = input
        outputs['target'] = target
        outputs['batch_info'] = data['batch_info']
        
        return outputs
    
    # finished looking at this function
    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        """Execute one validation step.
        
        Args:
            batch: Input batch from dataloader
            batch_idx: Index of current batch
            
        Returns:
            Dictionary containing losses and outputs
        """
        # Prepare batch
        data = self.prepare_batch(batch)
        input = data['input']
        target = data['target']
        
        # Generate fake images
        pred_g = self.models['generator'](input)
        
        # Prepare discriminator inputs
        input_d_real = torch.cat([input, target], dim=1)
        input_d_fake = torch.cat([input, pred_g], dim=1)
        
        # Get discriminator predictions
        pred_d_real = self.models['discriminator'](input_d_real)
        pred_d_fake = self.models['discriminator'](input_d_fake)
        
        # Calculate losses
        loss_d_unweighted = discriminator_loss(pred_d_real, pred_d_fake)
        loss_g_gan_unweighted = generator_gan_loss(pred_d_fake)
        loss_g_l1_unweighted = generator_l1_loss(pred_g, target)
        loss_g_ssim_unweighted = generator_ssim_loss(pred_g, target)
        loss_g_total_unweighted = loss_g_gan_unweighted + loss_g_l1_unweighted + loss_g_ssim_unweighted
        
        loss_d = self.lambda_disc * loss_d_unweighted
        loss_g_gan = self.lambda_gan * loss_g_gan_unweighted
        loss_g_l1 = self.lambda_l1 * loss_g_l1_unweighted
        loss_g_ssim = self.lambda_ssim * loss_g_ssim_unweighted
        loss_g_total = loss_g_gan + loss_g_l1 + loss_g_ssim
        
        outputs = {
            'loss_discriminator': loss_d,
            'loss_generator': loss_g_total,
            'loss_generator_gan': loss_g_gan,
            'loss_generator_l1': loss_g_l1,
            'loss_generator_ssim': loss_g_ssim,
            'loss_discriminator_unweighted': loss_d_unweighted,
            'loss_generator_gan_unweighted': loss_g_gan_unweighted,
            'loss_generator_l1_unweighted': loss_g_l1_unweighted,
            'loss_generator_ssim_unweighted': loss_g_ssim_unweighted,
            'loss_generator_unweighted': loss_g_total_unweighted,
            'generated': pred_g.detach(),
            'input': input,
            'target': target,
            'batch_info': data['batch_info']
        }
        
        return outputs
    
    # finished looking at this function. i dont even use a scheduler right now...
    def build_schedulers(self) -> Dict[str, Any]:
        """Build learning rate schedulers if configured."""
        schedulers = {}
        
        if self.cfg.train.get('use_lr_scheduler', False):
            # Example: Linear decay after certain epoch
            def lambda_rule(epoch):
                start_decay = self.cfg.train.get('lr_decay_start', 100)
                if epoch < start_decay:
                    return 1.0
                else:
                    return 1.0 - max(0, epoch - start_decay) / float(self.cfg.train.num_epochs - start_decay)
            
            schedulers['generator'] = torch.optim.lr_scheduler.LambdaLR(
                self.optimizers['generator'], lr_lambda=lambda_rule
            )
            schedulers['discriminator'] = torch.optim.lr_scheduler.LambdaLR(
                self.optimizers['discriminator'], lr_lambda=lambda_rule
            )
        
        return schedulers
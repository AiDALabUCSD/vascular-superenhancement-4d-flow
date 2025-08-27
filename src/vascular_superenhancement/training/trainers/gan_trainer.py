"""GAN trainer implementation for Pix2Pix-style training."""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torchio as tio
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import logging

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
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        callbacks: Optional[List[Callback]] = None
    ):
        super().__init__(cfg, train_loader, val_loader, test_loader, callbacks)
        
        # GAN-specific configuration
        self.lambda_l1 = cfg.train.lambda_l1
        self.lambda_ssim = cfg.train.get('lambda_ssim', 0.0)
        self.disc_update_freq = cfg.train.get('disc_update_freq', 1)
        self.gen_update_freq = cfg.train.get('gen_update_freq', 1)
        
    def build_models(self) -> Dict[str, nn.Module]:
        """Build generator and discriminator models."""
        models = {
            'generator': build_generator(self.cfg),
            'discriminator': build_discriminator(self.cfg)
        }
        logger.info(f"Built generator with {sum(p.numel() for p in models['generator'].parameters())} parameters")
        logger.info(f"Built discriminator with {sum(p.numel() for p in models['discriminator'].parameters())} parameters")
        return models
    
    def build_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        """Build optimizers for generator and discriminator."""
        optimizers = {
            'generator': torch.optim.Adam(
                self.models['generator'].parameters(),
                lr=self.cfg.train.generator_lr,
                betas=(0.5, 0.999)
            ),
            'discriminator': torch.optim.Adam(
                self.models['discriminator'].parameters(),
                lr=self.cfg.train.discriminator_lr,
                betas=(0.5, 0.999)
            )
        }
        return optimizers
    
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
        input_tensor = torch.cat([mag, speed], dim=1)
        
        return {
            'input': input_tensor,
            'target': cine,
            'mag': mag,
            'speed': speed,
            'batch_info': batch  # Keep original batch info for callbacks
        }
    
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
        input_tensor = data['input']
        target = data['target']
        
        # Generate fake images
        fake = self.models['generator'](input_tensor)
        
        outputs = {}
        
        # Train discriminator
        if self.global_step % self.disc_update_freq == 0:
            self.optimizers['discriminator'].zero_grad()
            
            # Prepare discriminator inputs
            real_input = torch.cat([input_tensor, target], dim=1)
            fake_input = torch.cat([input_tensor, fake.detach()], dim=1)
            
            # Get discriminator predictions
            pred_real = self.models['discriminator'](real_input)
            pred_fake = self.models['discriminator'](fake_input)
            
            # Calculate discriminator loss
            loss_d = discriminator_loss(pred_real, pred_fake)
            loss_d.backward()
            self.optimizers['discriminator'].step()
            
            outputs['loss_discriminator'] = loss_d
        
        # Train generator
        if self.global_step % self.gen_update_freq == 0:
            # Freeze discriminator
            for param in self.models['discriminator'].parameters():
                param.requires_grad_(False)
            
            self.optimizers['generator'].zero_grad()
            
            # Get discriminator prediction for fake images
            fake_input = torch.cat([input_tensor, fake], dim=1)
            pred_fake = self.models['discriminator'](fake_input)
            
            # Calculate generator losses
            loss_g_gan = generator_gan_loss(pred_fake)
            loss_g_l1 = generator_l1_loss(fake, target, weight=self.lambda_l1)
            loss_g_total = loss_g_gan + loss_g_l1
            
            # Add SSIM loss if configured
            if self.lambda_ssim > 0:
                loss_g_ssim = generator_ssim_loss(fake, target, weight=self.lambda_ssim)
                loss_g_total += loss_g_ssim
                outputs['loss_generator_ssim'] = loss_g_ssim
            
            loss_g_total.backward()
            self.optimizers['generator'].step()
            
            # Unfreeze discriminator
            for param in self.models['discriminator'].parameters():
                param.requires_grad_(True)
            
            outputs.update({
                'loss_generator': loss_g_total,
                'loss_generator_gan': loss_g_gan,
                'loss_generator_l1': loss_g_l1,
            })
        
        # Add generated image for visualization callbacks
        outputs['generated'] = fake.detach()
        outputs['input'] = input_tensor
        outputs['target'] = target
        outputs['batch_info'] = data['batch_info']
        
        return outputs
    
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
        input_tensor = data['input']
        target = data['target']
        
        # Generate fake images
        fake = self.models['generator'](input_tensor)
        
        # Prepare discriminator inputs
        real_input = torch.cat([input_tensor, target], dim=1)
        fake_input = torch.cat([input_tensor, fake], dim=1)
        
        # Get discriminator predictions
        pred_real = self.models['discriminator'](real_input)
        pred_fake = self.models['discriminator'](fake_input)
        
        # Calculate losses
        loss_d = discriminator_loss(pred_real, pred_fake)
        loss_g_gan = generator_gan_loss(pred_fake)
        loss_g_l1 = generator_l1_loss(fake, target, weight=self.lambda_l1)
        loss_g_total = loss_g_gan + loss_g_l1
        
        outputs = {
            'loss_discriminator': loss_d,
            'loss_generator': loss_g_total,
            'loss_generator_gan': loss_g_gan,
            'loss_generator_l1': loss_g_l1,
            'loss_total': loss_g_total,  # For early stopping
            'generated': fake.detach(),
            'input': input_tensor,
            'target': target,
            'batch_info': data['batch_info']
        }
        
        # Add SSIM loss if configured
        if self.lambda_ssim > 0:
            loss_g_ssim = generator_ssim_loss(fake, target, weight=self.lambda_ssim)
            outputs['loss_generator_ssim'] = loss_g_ssim
            outputs['loss_generator'] += loss_g_ssim
            outputs['loss_total'] += loss_g_ssim
        
        return outputs
    
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
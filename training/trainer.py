import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import os

from models.generator import Generator
from models.discriminator import Discriminator
from training.metrics import Metrics

class Trainer:
    """Class for training the face-conditional GAN."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        # Initialize optimizers
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=config['learning_rate'],
            betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=config['learning_rate'],
            betas=(0.5, 0.999)
        )
        
        # Initialize metrics
        self.metrics = Metrics(device=self.device)
        
        # Initialize wandb
        wandb.init(
            project=config['project_name'],
            config=config
        )
        
    def train(self, train_loader, num_epochs):
        """Train the GAN for specified number of epochs."""
        for epoch in range(num_epochs):
            self._train_epoch(train_loader, epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.config['save_interval'] == 0:
                self._save_checkpoint(epoch)
                
    def _train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
        
        for batch in progress_bar:
            # Prepare data
            real_images = batch['image'].to(self.device)
            embeddings = batch['embedding'].to(self.device)
            noise = torch.randn(real_images.size(0), 100).to(self.device)
            
            # Train discriminator
            self.optimizer_D.zero_grad()
            
            # Real images
            d_real = self.discriminator(real_images, embeddings)
            real_loss, _ = self.metrics.calculate_gan_loss(d_real, None)
            
            # Fake images
            fake_images = self.generator(embeddings, noise)
            d_fake = self.discriminator(fake_images.detach(), embeddings)
            _, fake_loss = self.metrics.calculate_gan_loss(None, d_fake)
            
            d_loss = real_loss + fake_loss
            d_loss.backward()
            self.optimizer_D.step()
            
            # Train generator
            self.optimizer_G.zero_grad()
            
            fake_images = self.generator(embeddings, noise)
            d_fake = self.discriminator(fake_images, embeddings)
            g_loss = F.binary_cross_entropy(d_fake, torch.ones_like(d_fake))
            l1_loss = self.metrics.calculate_l1_loss(real_images, fake_images)
            
            total_g_loss = g_loss + self.config['lambda_l1'] * l1_loss
            total_g_loss.backward()
            self.optimizer_G.step()
            
            # Log metrics
            metrics = {
                'epoch': epoch,
                'd_loss': d_loss.item(),
                'g_loss': g_loss.item(),
                'l1_loss': l1_loss.item(),
                'total_g_loss': total_g_loss.item()
            }
            
            if (epoch + 1) % self.config['fid_interval'] == 0:
                fid = self.metrics.calculate_fid(real_images, fake_images)
                metrics['fid'] = fid
                
            wandb.log(metrics)
            progress_bar.set_postfix(metrics)
            
    def _save_checkpoint(self, epoch):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict()
        }
        
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, f'checkpoints/checkpoint_epoch_{epoch}.pt')
        
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        return checkpoint['epoch'] 
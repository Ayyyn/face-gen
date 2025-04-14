import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.generator import Generator
from models.discriminator import Discriminator
from data.dataset import FaceDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Path to images directory')
    parser.add_argument('--embedding_dir', type=str, required=True,
                       help='Path to embeddings directory')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Path to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0002,
                       help='Learning rate')
    parser.add_argument('--lambda_l1', type=float, default=100.0,
                       help='Weight for L1 loss')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb
    wandb.init(project="face-gen", config=vars(args))
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Initialize optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # Initialize dataset and dataloader
    dataset = FaceDataset(args.image_dir, args.embedding_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Loss functions
    criterion_gan = torch.nn.BCEWithLogitsLoss()
    criterion_l1 = torch.nn.L1Loss()
    
    # Training loop
    for epoch in range(args.num_epochs):
        generator.train()
        discriminator.train()
        
        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs}'):
            # Get data
            real_images = batch['image'].to(device)
            embeddings = batch['embedding'].to(device)
            
            # Generate fake images
            noise = torch.randn(real_images.size(0), 100, device=device)
            fake_images = generator(embeddings, noise)
            
            # Train discriminator
            d_optimizer.zero_grad()
            
            # Real images
            real_pred = discriminator(real_images, embeddings)
            d_loss_real = criterion_gan(real_pred, torch.ones_like(real_pred))
            
            # Fake images
            fake_pred = discriminator(fake_images.detach(), embeddings)
            d_loss_fake = criterion_gan(fake_pred, torch.zeros_like(fake_pred))
            
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            d_optimizer.step()
            
            # Train generator
            g_optimizer.zero_grad()
            
            # Adversarial loss
            fake_pred = discriminator(fake_images, embeddings)
            g_loss_gan = criterion_gan(fake_pred, torch.ones_like(fake_pred))
            
            # L1 loss
            g_loss_l1 = criterion_l1(fake_images, real_images) * args.lambda_l1
            
            g_loss = g_loss_gan + g_loss_l1
            g_loss.backward()
            g_optimizer.step()
            
            # Log losses
            wandb.log({
                'd_loss': d_loss.item(),
                'g_loss_gan': g_loss_gan.item(),
                'g_loss_l1': g_loss_l1.item(),
                'g_loss': g_loss.item()
            })
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt'))
            
            # Log sample images
            with torch.no_grad():
                sample_images = fake_images[:8]  # Take first 8 images
                wandb.log({
                    'generated_images': [wandb.Image(img) for img in sample_images],
                    'real_images': [wandb.Image(img) for img in real_images[:8]]
                })

if __name__ == '__main__':
    main() 
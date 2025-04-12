import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
from torchvision.transforms import Normalize
import numpy as np
from scipy import linalg

class Metrics:
    """Class for calculating GAN evaluation metrics."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.inception = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception.eval()
        
    def calculate_fid(self, real_images, fake_images):
        """
        Calculate Frechet Inception Distance (FID).
        
        Args:
            real_images: tensor of real images
            fake_images: tensor of generated images
            
        Returns:
            fid_score: float
        """
        # Get features
        real_features = self._get_inception_features(real_images)
        fake_features = self._get_inception_features(fake_images)
        
        # Calculate statistics
        mu_real, sigma_real = self._calculate_statistics(real_features)
        mu_fake, sigma_fake = self._calculate_statistics(fake_features)
        
        # Calculate FID
        ssdiff = np.sum((mu_real - mu_fake) ** 2.0)
        covmean = linalg.sqrtm(sigma_real.dot(sigma_fake))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = ssdiff + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
        return float(fid)
    
    def _get_inception_features(self, images):
        """Extract features using Inception v3."""
        features = []
        with torch.no_grad():
            for img in images:
                img = img.unsqueeze(0).to(self.device)
                feature = self.inception(img)
                features.append(feature.cpu().numpy())
        return np.concatenate(features, axis=0)
    
    def _calculate_statistics(self, features):
        """Calculate mean and covariance of features."""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    @staticmethod
    def calculate_l1_loss(real_images, fake_images):
        """Calculate L1 loss between real and fake images."""
        return F.l1_loss(fake_images, real_images)
    
    @staticmethod
    def calculate_gan_loss(d_real, d_fake):
        """Calculate GAN loss components."""
        real_loss = F.binary_cross_entropy(d_real, torch.ones_like(d_real))
        fake_loss = F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))
        return real_loss, fake_loss 
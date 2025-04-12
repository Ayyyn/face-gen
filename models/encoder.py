import torch
import numpy as np
from insightface.app import FaceAnalysis
import cv2

class FaceEncoder:
    """Wrapper for ArcFace face embedding extractor."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(128, 128))
        
    def get_embedding(self, image):
        """
        Extract face embedding from an image.
        
        Args:
            image: numpy array of shape (H, W, 3) in RGB format
            
        Returns:
            embedding: torch tensor of shape (512,)
        """
        # Convert to BGR for insightface
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Get face embedding
        faces = self.app.get(image_bgr)
        if len(faces) == 0:
            raise ValueError("No face detected in the image")
        
        # Get the embedding of the first detected face
        embedding = torch.from_numpy(faces[0].embedding).float()
        return embedding.to(self.device)
    
    def preprocess_batch(self, images):
        """
        Process a batch of images to extract embeddings.
        
        Args:
            images: list of numpy arrays of shape (H, W, 3)
            
        Returns:
            embeddings: torch tensor of shape (batch_size, 512)
        """
        embeddings = []
        for img in images:
            try:
                emb = self.get_embedding(img)
                embeddings.append(emb)
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                continue
        
        if not embeddings:
            raise ValueError("No valid embeddings extracted from the batch")
            
        return torch.stack(embeddings) 
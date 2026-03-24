import os
import torch
import random
import torch.nn as nn

import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

# Deepfake detection dataset
class DeepfakeDetectionDataset(Dataset):
    def __init__(self, base_dir, processor, split="train", category="car", max_per_class=None, use_augmentation=True, seed=None):
        
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.processor = processor
        
        # Only use data augmentation for training set
        self.split = split
        self.use_augmentation = use_augmentation and split == "train"  
        # Set paths
        self.data_dir = os.path.join(base_dir, split, category)
        
        # Get paths for real and fake images
        self.real_dir = os.path.join(self.data_dir, "0_real")
        self.fake_dir = os.path.join(self.data_dir, "1_fake")
        
        print(f"Loading dataset - {split} - Real images directory: {self.real_dir}")
        print(f"Loading dataset - {split} - Fake images directory: {self.fake_dir}")
        
        # Get real and fake image paths
        self.real_images = [os.path.join(self.real_dir, f) for f in os.listdir(self.real_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.fake_images = [os.path.join(self.fake_dir, f) for f in os.listdir(self.fake_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

        # Shuffle data
        np.random.shuffle(self.real_images)
        np.random.shuffle(self.fake_images)
        
        # Optional data limitation
        if max_per_class is not None:
            self.real_images = self.real_images[:max_per_class]
            self.fake_images = self.fake_images[:max_per_class]
            print(f"Limiting maximum samples per class to: {max_per_class}")
        
        print(f"Loaded {len(self.real_images)} real images and {len(self.fake_images)} fake images")
        
        # Combine all images and labels
        self.images = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)  # 0: real, 1: fake
        
        # Shuffle data again
        combined = list(zip(self.images, self.labels))
        random.Random(self.seed).shuffle(combined)  # Use fixed seed
        self.images, self.labels = zip(*combined)
        # Set up data augmentation transforms
        self.augmentations = self._get_augmentations()
    
    def _get_augmentations(self):
        """Define data augmentation strategies, optimized for diffusion model characteristics"""
        torch.manual_seed(seed=6915)  # Set random seed for reproducibility
        return transforms.Compose([
            # Random horizontal flip
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),

            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            AddGaussianNoise(mean=0, std=0.02, p=0.1),
            JpegCompression(quality_lower=60, quality_upper=95, p=0.3),
            FFTBasedAugmentation(strength=0.1, p=0.1)
        ])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        
        # Load original image
        image = Image.open(image_path).convert("RGB")
        
        # Apply data augmentation (if enabled)
        if self.use_augmentation:
            image = self.augmentations(image)
        
        # Process image using LanguageBind processor
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        return inputs, torch.tensor(label, dtype=torch.long)
    

    # Custom transform: Add Gaussian noise
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., p=0.5):
        self.mean = mean
        self.std = std
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            img = F.to_tensor(img)
            noise = torch.randn_like(img) * self.std + self.mean
            noisy_img = img + noise
            noisy_img = torch.clamp(noisy_img, 0., 1.)
            return F.to_pil_image(noisy_img)
        return img
    
# JPEG compression simulation
class JpegCompression(object):
    def __init__(self, quality_lower=60, quality_upper=95, p=0.5):
        self.quality_lower = quality_lower
        self.quality_upper = quality_upper
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            quality = random.randint(self.quality_lower, self.quality_upper)
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img = Image.open(buffer)
        return img

# FFT-based frequency domain augmentation
class FFTBasedAugmentation(object):
    def __init__(self, strength=0.1, p=1.0):
        self.strength = strength
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            # Convert to numpy array
            img_np = np.array(img).astype(np.float32) / 255.0
            
            # Process each channel with FFT
            for i in range(3):  # Three RGB channels
                channel = img_np[..., i]
                
                # Apply FFT
                fft = np.fft.fft2(channel)
                fft_shift = np.fft.fftshift(fft)
                
                # Create high frequency mask (randomly mask some high frequency parts)
                rows, cols = channel.shape
                crow, ccol = rows // 2, cols // 2
                mask = np.ones((rows, cols), np.uint8)
                r = crow // 4  # High frequency radius
                center = (crow, ccol)
                x, y = np.ogrid[:rows, :cols]
                mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 >= r*r
                mask[mask_area] = 0
                
                # Randomly adjust magnitude
                magnitude_scale = 1.0 + (random.random() * 2 - 1) * self.strength
                fft_shift = fft_shift * (1 - mask) + fft_shift * mask * magnitude_scale
                
                # Inverse FFT
                ifft_shift = np.fft.ifftshift(fft_shift)
                ifft = np.fft.ifft2(ifft_shift)
                channel_processed = np.abs(ifft)
                
                # Normalize
                channel_processed = np.clip(channel_processed, 0, 1)
                img_np[..., i] = channel_processed
            
            # Convert back to PIL image
            img_np = (img_np * 255).astype(np.uint8)
            img = Image.fromarray(img_np)
            
        return img

# Pixel quantization, simulate discretization process
class PixelQuantization(object):
    def __init__(self, bits=[5, 6, 7, 8], p=0.2):
        self.bits = bits  # List of quantization bit depths
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            img = F.to_tensor(img)
            bit = random.choice(self.bits)
            levels = 2 ** bit
            img_quantized = torch.floor(img * levels) / levels
            img = F.to_pil_image(img_quantized)
        return img
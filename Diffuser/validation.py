import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
import numpy as np
import config
import diffusion_utils

class DigitalJudge(nn.Module):
    """ Pre-trained ResNet to extract feature embeddings """
    def __init__(self, device):
        super().__init__()
        # Load standard ResNet18 trained on ImageNet
        resnet = models.resnet18(pretrained=True)
        # Remove the final classification layer (we want features, not class labels)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.eval()
        self.to(device)

    def get_embeddings(self, images):
        with torch.no_grad():
            # Upsample 64x64 -> 224x224 (ResNet expects larger images)
            img_resized = F.interpolate(images, size=(224, 224), mode='bilinear')
            # Normalize for ImageNet stats
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img_norm = normalize(img_resized)
            
            emb = self.features(img_norm)
            return emb.flatten(start_dim=1).cpu().numpy()

def check_quality(model, dataloader, device, num_samples=50):
    print(f"\n👨‍⚕️ Digital Judge is Auditing (Sample size: {num_samples})...")
    judge = DigitalJudge(device)
    
    # 1. Get Real Features
    real_feats = []
    count = 0
    for batch in dataloader:
        batch = batch.to(device)
        # Unnormalize [-1, 1] -> [0, 1] for ResNet
        batch = (batch + 1) / 2
        
        feats = judge.get_embeddings(batch)
        real_feats.append(feats)
        count += len(batch)
        if count >= num_samples: break
    real_mean = np.mean(np.concatenate(real_feats), axis=0)

    # 2. Generate Synthetic Features
    syn_feats = []
    model.eval()
    
    # Generate in batches
    generated = 0
    while generated < num_samples:
        curr_batch = min(16, num_samples - generated)
        # Start from noise
        img = torch.randn((curr_batch, 3, config.IMAGE_SIZE, config.IMAGE_SIZE), device=device)
        
        # Fast Denoise Loop
        for i in range(0, config.TIMESTEPS)[::-1]:
            t = torch.full((curr_batch,), i, device=device, dtype=torch.long)
            predicted_noise = model(img, t)
            
            # Simple Diffusion Step
            alpha = diffusion_utils.alphas[i]
            alpha_hat = diffusion_utils.alphas_cumprod[i]
            beta = diffusion_utils.betas[i]
            if i > 0: noise = torch.randn_like(img)
            else: noise = torch.zeros_like(img)
            
            img = (1 / torch.sqrt(alpha)) * (img - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        
        # Post-process [0, 1]
        img = (img + 1) / 2
        img = torch.clamp(img, 0, 1)
        
        feats = judge.get_embeddings(img)
        syn_feats.append(feats)
        generated += curr_batch
        
    syn_mean = np.mean(np.concatenate(syn_feats), axis=0)

    # 3. Score (Euclidean Distance)
    diff = real_mean - syn_mean
    score = np.dot(diff, diff)
    
    print(f"📊 Feature Distance Score: {score:.4f}")
    if score < 50: 
        print("✅ PASS: High Feature Alignment.")
    else:
        print("⚠️ WARNING: Synthetic features diverge from Real.")
    
    return score
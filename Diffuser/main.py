import sys
sys.path.append('/content/drive/MyDrive/FYP/releif_code')

import config
import dataset
import models
import diffusion_utils
import validation  # <--- NEW IMPORT
import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Setup
real_img_dir = dataset.prepare_data()
ds = dataset.TargetedDataset(config.SPLIT_PATH, real_img_dir, config.TARGET_CLASS, dataset.get_transforms())
loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)

model = models.SimpleUnet().to(config.DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

losses = []
validation_scores = []

print(f"🔥 Training RELEIF Diffusion on {config.TARGET_CLASS}...")

for epoch in range(config.EPOCHS):
    for batch in loader:
        optimizer.zero_grad()
        t = torch.randint(0, config.TIMESTEPS, (config.BATCH_SIZE,), device=config.DEVICE).long()
        batch = batch.to(config.DEVICE)
        
        loss = diffusion_utils.get_loss(model, batch, t, config.DEVICE)
        loss.backward()
        optimizer.step()
        
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.5f}")
        losses.append(loss.item())
        
    # --- VALIDATION STEP ---
    # Run every 50 epochs to check quality
    if (epoch + 1) % 50 == 0:
        score = validation.check_quality(model, loader, config.DEVICE)
        validation_scores.append(score)
        
        # Save Model
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        torch.save(model.state_dict(), f"{config.MODEL_SAVE_DIR}/model_{epoch+1}.pth")

# Plot Results
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(losses)
plt.title("Training Loss")
plt.subplot(1,2,2)
plt.plot(range(49, config.EPOCHS, 50), validation_scores)
plt.title("Validation Score (Lower is Better)")
plt.show()
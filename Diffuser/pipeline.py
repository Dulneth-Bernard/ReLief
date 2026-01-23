import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# Internal Imports (No sys.path needed here since they are neighbors)
import config
import dataset
import models
import diffusion_utils
import validation

def run_releif_pipeline(target_class, num_samples_needed):
    """
    The Master Function.
    1. Sets up the environment for the specific Class.
    2. Trains a Targeted Diffusion Model.
    3. Generates the exact number of synthetic images needed.
    """
    print(f"\n" + "="*60)
    print(f"🚀 STARTING RELEIF PIPELINE FOR: {target_class}")
    print(f"🎯 Goal: Generate {num_samples_needed} synthetic images")
    print("="*60)

    # --- 1. DYNAMIC CONFIGURATION OVERRIDE ---
    config.TARGET_CLASS = target_class
    config.MODEL_SAVE_DIR = f'/content/drive/MyDrive/FYP/models/diffusion_{target_class}'
    config.SYNTHETIC_SAVE_DIR = f'/content/drive/MyDrive/FYP/synthetic_dataset/{target_class}'
    
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.SYNTHETIC_SAVE_DIR, exist_ok=True)
    
    print(f"📂 Model Dir: {config.MODEL_SAVE_DIR}")
    print(f"📂 Output Dir: {config.SYNTHETIC_SAVE_DIR}")

    # --- 2. DATA SETUP ---
    real_img_dir = dataset.prepare_data()
    
    ds = dataset.TargetedDataset(
        config.SPLIT_PATH, 
        real_img_dir, 
        target_class, 
        dataset.get_transforms()
    )
    
    if len(ds) == 0:
        print(f"❌ ERROR: No images found for class {target_class}. Check spelling!")
        return

    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"✅ Loaded {len(ds)} real images for training.")

    # --- 3. MODEL INIT ---
    model = models.SimpleUnet().to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Check for existing checkpoint
    saved_models = sorted([f for f in os.listdir(config.MODEL_SAVE_DIR) if f.endswith('.pth')])
    start_epoch = 0
    if len(saved_models) > 0:
        last_model = os.path.join(config.MODEL_SAVE_DIR, saved_models[-1])
        model.load_state_dict(torch.load(last_model, map_location=config.DEVICE))
        print(f"🔄 Resumed from checkpoint: {saved_models[-1]}")
        try: start_epoch = int(saved_models[-1].split('_')[-1].replace('.pth', ''))
        except: start_epoch = 0

    # --- 4. TRAINING LOOP ---
    if start_epoch < config.EPOCHS:
        print(f"🔥 Training for {config.EPOCHS - start_epoch} epochs...")
        model.train()
        
        for epoch in range(start_epoch, config.EPOCHS):
            epoch_loss = 0
            for batch in loader:
                optimizer.zero_grad()
                t = torch.randint(0, config.TIMESTEPS, (config.BATCH_SIZE,), device=config.DEVICE).long()
                batch = batch.to(config.DEVICE)
                
                loss = diffusion_utils.get_loss(model, batch, t, config.DEVICE)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 50 == 0:
                avg_loss = epoch_loss / len(loader)
                print(f"   Epoch {epoch+1} | Loss: {avg_loss:.5f}")
                torch.save(model.state_dict(), f"{config.MODEL_SAVE_DIR}/model_epoch_{epoch+1}.pth")

        print("✅ Training Complete.")
    else:
        print("✅ Model already trained to target epochs. Skipping training.")

    # --- 5. MASS GENERATION LOOP ---
    print(f"🏭 Generating {num_samples_needed} images...")
    model.eval()
    
    existing_files = len(os.listdir(config.SYNTHETIC_SAVE_DIR))
    if existing_files >= num_samples_needed:
        print(f"✅ Target reached! Folder already has {existing_files} images.")
        return

    generated_count = existing_files
    
    with torch.no_grad():
        pbar = tqdm(total=num_samples_needed - generated_count)
        while generated_count < num_samples_needed:
            current_batch = min(config.BATCH_SIZE, num_samples_needed - generated_count)
            img = torch.randn((current_batch, 3, config.IMAGE_SIZE, config.IMAGE_SIZE), device=config.DEVICE)
            
            for i in range(0, config.TIMESTEPS)[::-1]:
                t = torch.full((current_batch,), i, device=config.DEVICE, dtype=torch.long)
                predicted_noise = model(img, t)
                
                alpha = diffusion_utils.alphas[i]
                alpha_hat = diffusion_utils.alphas_cumprod[i]
                beta = diffusion_utils.betas[i]
                
                if i > 0: noise = torch.randn_like(img)
                else: noise = torch.zeros_like(img)
                
                img = (1 / torch.sqrt(alpha)) * (img - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

            img = (img + 1) / 2
            img = torch.clamp(img, 0, 1)
            
            for j in range(current_batch):
                fname = f"syn_{target_class}_{generated_count+j:05d}.jpg"
                save_path = os.path.join(config.SYNTHETIC_SAVE_DIR, fname)
                save_image(img[j], save_path)
            
            generated_count += current_batch
            pbar.update(current_batch)
        pbar.close()

    print(f"🎉 SUCCESS: {target_class} Pipeline Finished.")
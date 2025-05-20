import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.unet2d import UNet
import numpy as np
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

class BraTSValDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        image = data['image']
        return torch.tensor(image, dtype=torch.float32), self.files[idx]

def plot_prediction(image, pred_mask, filename):
    image = image[0].cpu().numpy()
    pred_mask = (pred_mask > 0.5).float().cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Input Slice")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask[0], cmap='Reds', alpha=0.6)
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def inference():
    data_dir = "processed_data/preprocessed_val_slices"
    output_dir = "predicted_val_masks"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load("best_unet_model.pt"))
    model.eval()

    dataset = BraTSValDataset(data_dir)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False, pin_memory=True)

    print(f"🔍 Starting inference on {len(dataset)} slices...")
    total_saved = 0

    with torch.no_grad():
        for batch_idx, (images, paths) in enumerate(tqdm(val_loader, desc="Inferring")):
            images = images.to(device)
            outputs = model(images)
            outputs = outputs[:, :, :images.shape[2], :images.shape[3]]

            for i in range(images.shape[0]):
                fname = os.path.basename(paths[i]).replace(".npz", "_pred.png")
                out_path = os.path.join(output_dir, fname)
                plot_prediction(images[i], outputs[i], out_path)
                total_saved += 1

            # print(f"Batch {batch_idx + 1}/{len(val_loader)}: saved {total_saved} masks...")

    print(f"\n✅ Inference done! {total_saved} predictions saved in '{output_dir}'")

if __name__ == "__main__":
    inference()

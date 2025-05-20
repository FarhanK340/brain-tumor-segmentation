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
import pandas as pd
import matplotlib.pyplot as plt

class BrainMRIDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        image = data['image']
        mask = data['mask']
        mask[mask > 0] = 1  # binarize
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32).unsqueeze(0), self.files[idx]

def dice_score(pred, target, smooth=1.):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
    return dice

def iou_score(pred, target, smooth=1.):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def plot_prediction(image, pred_mask, true_mask, filename):
    image = image[0].cpu().numpy()  # pick one modality (e.g., t1n)
    pred_mask = (pred_mask > 0.5).float().cpu().numpy()
    true_mask = true_mask.cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Input Slice")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask[0], cmap='Reds', alpha=0.6)
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(true_mask[0], cmap='Blues', alpha=0.6)
    plt.title("Ground Truth Mask")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def evaluate():
    data_dir = "processed_data/preprocessed_slices"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load("best_unet_model.pt"))
    model.eval()

    dataset = BrainMRIDataset(data_dir)
    # val_size = int(0.2 * len(dataset))
    split = torch.load("split_indices.pt")
    val_subset = torch.utils.data.Subset(dataset, split["val_indices"])
    val_loader = DataLoader(val_subset, batch_size=4, shuffle=False, pin_memory=True)

    bce = nn.BCELoss()
    results = []

    print("🔍 Evaluating on validation set...")
    with torch.no_grad():
        for batch_idx, (images, masks, paths) in enumerate(tqdm(val_loader, desc="Evaluating")):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            _, _, h, w = outputs.shape
            masks = masks[:, :, :h, :w]
            outputs = outputs[:, :, :h, :w]

            dice = dice_score(outputs, masks)
            iou = iou_score(outputs, masks)
            loss = bce(outputs, masks)

            for i in range(images.shape[0]):
                fname = os.path.basename(paths[i])
                results.append({
                    "filename": fname,
                    "dice": dice[i].item(),
                    "iou": iou[i].item(),
                    "bce_loss": loss.item()
                })

                if batch_idx == 0 and i < 2:  # Plot first 2 examples
                    plot_prediction(images[i], outputs[i], masks[i], f"prediction_{fname.replace('.npz', '.png')}")

    df = pd.DataFrame(results)
    df.to_csv("evaluation_results.csv", index=False)

    print("\n✅ Evaluation Summary:")
    print(df.describe()[['dice', 'iou', 'bce_loss']])
    print("\n📊 Saved metrics to 'evaluation_results.csv'")
    print("🖼️ Saved prediction overlays as 'prediction_*.png'")

if __name__ == "__main__":
    evaluate()

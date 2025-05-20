import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import time
import glob
import numpy as np
from tqdm import tqdm
from src.models.unet2d import UNet
from torch.utils.data import DataLoader
import torch.nn as nn
import torch


class BrainMRIDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        image = data['image']  # shape [4, H, W]
        mask = data['mask']    # shape [H, W]
        mask[mask > 0] = 1
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32).unsqueeze(0)


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(2, 3))
    loss = 1 - ((2. * intersection + smooth) /
                (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth))
    return loss.mean()


def train(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for images, masks in tqdm(loader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        _, _, h, w = outputs.shape
        masks = masks[:, :, :h, :w]
        loss = criterion(outputs, masks) + dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating", leave=False):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            _, _, h, w = outputs.shape
            masks = masks[:, :, :h, :w]
            loss = criterion(outputs, masks) + dice_loss(outputs, masks)
            val_loss += loss.item()
    return val_loss / len(loader)


def main():
    data_dir = "./processed_data/preprocessed_slices"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)

    dataset = BrainMRIDataset(data_dir)
    print(f"[DEBUG] Found {len(dataset)} slice files in: {data_dir}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    torch.save({
        "train_indices": train_set.indices,
        "val_indices": val_set.indices
    }, "split_indices.pt")

    train_loader = DataLoader(train_set, batch_size=4,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=4,
                            shuffle=False, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    for epoch in range(1, 11):
        t0 = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        t1 = time.time()
        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {t1 - t0:.2f}s")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_unet_model.pt")


if __name__ == "__main__":
    main()

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

# === Path to one sample folder ===
sample_dir = "./../brats2025_data/brats2025-gli-pre-challenge-trainingdata/BraTS2025-GLI-PRE-Challenge-TrainingData/BraTS-GLI-00000-000"

# === Load each modality ===
modalities = {
    "t1n": "BraTS-GLI-00000-000-t1n.nii.gz",
    "t1c": "BraTS-GLI-00000-000-t1c.nii.gz",
    "t2f": "BraTS-GLI-00000-000-t2f.nii.gz",
    "t2w": "BraTS-GLI-00000-000-t2w.nii.gz",
    "seg": "BraTS-GLI-00000-000-seg.nii.gz"
}


# === Load and display middle slice ===
def show_middle_slice(modality, label=False):
    img = nib.load(os.path.join(sample_dir, modality)).get_fdata()
    mid_slice = img.shape[2] // 2
    plt.imshow(img[:, :, mid_slice], cmap='gray' if not label else 'nipy_spectral')
    plt.title(f"{modality} - Slice {mid_slice}")
    plt.axis('off')
    plt.show()

for mod, file in modalities.items():
    show_middle_slice(file, label=(mod == "seg"))

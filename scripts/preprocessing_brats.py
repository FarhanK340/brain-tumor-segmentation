import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
import imageio

input_root = "brats2025_data/brats2025-gli-pre-challenge-validationdata/BraTS2025-GLI-PRE-Challenge-ValidationData/"
output_root = "processed_data/preprocessed_val_slices"

os.makedirs(output_root, exist_ok=True)

patients = sorted(os.listdir(input_root))
modalities = ["t1n", "t1c", "t2f", "t2w"]

def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

for pid in tqdm(patients):
    p_dir = os.path.join(input_root, pid)
    
    try:
        vols = []
        for m in modalities:
            vol = nib.load(os.path.join(p_dir, f"{pid}-{m}.nii.gz")).get_fdata()
            vols.append(normalize(vol))

        # seg = nib.load(os.path.join(p_dir, f"{pid}-seg.nii.gz")).get_fdata()
        vols = np.stack(vols, axis=0)  # shape: [4, H, W, D]

        for i in range(vols.shape[3]):
            img_slice = vols[:, :, :, i]  # shape: [4, H, W]
            # mask_slice = seg[:, :, i]
            # if np.sum(mask_slice) == 0:
            #     continue  # Skip slices with no tumor

            # Save as .npz
            np.savez_compressed(
                os.path.join(output_root, f"{pid}_slice_{i:03d}.npz"),
                image=img_slice.astype(np.float32),
                # mask=mask_slice.astype(np.uint8)
            )
    except Exception as e:
        print(f"⚠️ Failed {pid}: {e}")

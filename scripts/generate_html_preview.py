import os

# Path to your predicted mask overlays (adjust if needed)
mask_dir = "predicted_val_masks"
output_file = "val_predictions_preview_3.html"

# List all PNGs
image_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

# Create HTML file
with open(output_file, "w") as f:
    f.write("<html><head><title>Validation Predictions Preview</title></head><body>\n")
    f.write("<h1>Validation Tumor Predictions</h1>\n")
    for img in image_files[1890:1930]:  # shows the range of images
        f.write(f"<div style='display:inline-block;margin:10px;text-align:center;'>\n")
        f.write(f"<img src='{mask_dir}/{img}' width='512'><br>\n")
        f.write(f"{img}</div>\n")
    f.write("</body></html>\n")

print(f"✅ Preview generated: {output_file}")

import os
import glob
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, SiglipForImageClassification

MODEL_NAME = "prithivMLmods/Minc-Materials-23"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SiglipForImageClassification.from_pretrained(MODEL_NAME).to(DEVICE)
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model.eval()

RGB_ROOT = ""

SCENES    = ["scene1", "scene2","scene3"]         
MATERIALS = ["metal", "glass", "plastic", "cardboard"]

def load_rgb_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img

def main():
    emb_dict = {}   # key: "scene1/metal"

    with torch.no_grad():
        for scene in SCENES:
            scene_dir = os.path.join(RGB_ROOT, scene)
            for root, dirs, files in os.walk(scene_dir):
                for fname in files:
                    if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        continue

                    img_path = os.path.join(root, fname)
                    img = load_rgb_image(img_path)

                    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
                    vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
                    img_emb = F.normalize(vision_outputs.pooler_output, dim=-1)
                    stem = Path(fname).stem  
                    img_id = stem.replace("_rgb", "")  
                    key = f"{scene}/{img_id}"            
                    img_emb = F.normalize(img_emb, dim=-1)
                    emb_dict[key] = img_emb.squeeze(0).cpu().numpy()  

                    print(f"Extracted embedding for {key} from {img_path}")

    np.savez("minc_rgb_embeddings.npz", **emb_dict)
    print("Saved embeddings to minc_rgb_embeddings.npz")

if __name__ == "__main__":
    main()

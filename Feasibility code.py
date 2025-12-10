import cv2
import numpy as np
import os
import pandas as pd

IMG_DIR = "sample_images"

def extract_features(img_path, label):
    img = cv2.imread(img_path)
    if img is None:
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    #Mean HSV values
    mean_h = float(np.mean(h))
    mean_s = float(np.mean(s))
    mean_v = float(np.mean(v))

    #Edge density
    edges = cv2.Canny(img, 100, 200)
    edge_density = float(np.sum(edges > 0) / edges.size)

    return {
        "image": os.path.basename(img_path),
        "label": label,
        "mean_hue": mean_h,
        "mean_saturation": mean_s,
        "mean_value": mean_v,
        "edge_density": edge_density
    }

rows = []

for filename in os.listdir(IMG_DIR):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(IMG_DIR, filename)

        # auto-label based on filename
        if "cozy" in filename.lower():
            label = "cozy"
        elif "non" in filename.lower():
            label = "noncozy"
        else:
            label = "unknown"

        feats = extract_features(path, label)
        if feats:
            rows.append(feats)

#output
df = pd.DataFrame(rows)
df.to_csv("features.csv", index=False)

print("Feature extraction complete. Saved to features.csv")
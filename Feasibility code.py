import cv2
import numpy as np
import os
import pandas as pd

#folder containing example images
IMG_DIR = "sample_images"

def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    #convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mean_h = np.mean(h)
    mean_s = np.mean(s)
    mean_v = np.mean(v)

    #edge density
    edges = cv2.Canny(img, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    return {
        "image": os.path.basename(img_path),
        "mean_hue": mean_h,
        "mean_saturation": mean_s,
        "mean_value": mean_v,
        "edge_density": edge_density
    }

rows = []

for filename in os.listdir(IMG_DIR):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(IMG_DIR, filename)
        feats = extract_features(path)
        if feats:
            rows.append(feats)

df = pd.DataFrame(rows)
df.to_csv("features.csv", index=False)

print("Feature extraction complete. Saved to features.csv")
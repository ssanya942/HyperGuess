import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
from pathlib import Path

# === User-defined paths ===
NORMAL_FILES = [
    "/Users/sanya/Desktop/uncalib-phantoms/normal-1001249258-20250609-132330-091097.loraw",
    "/Users/sanya/Desktop/uncalib-phantoms/normal-1001249258-20250609-132333-677060.loraw",
    "/Users/sanya/Desktop/uncalib-phantoms/normal-1001249258-20250609-132340-810996.loraw",
    "/Users/sanya/Desktop/uncalib-phantoms/normal-1001249258-20250609-132346-379105.loraw",
    "/Users/sanya/Desktop/uncalib-phantoms/normal-1001249258-20250609-132347-026963.loraw",
    "/Users/sanya/Desktop/uncalib-phantoms/normal-1001249258-20250609-132347-676870.loraw",
    "/Users/sanya/Desktop/uncalib-phantoms/normal-1001249258-20250609-132356-677469.loraw",
    "/Users/sanya/Desktop/uncalib-phantoms/normal-1001249258-20250609-132357-490909.loraw"
]
PATHO_FILES = [
    "/Users/sanya/Desktop/uncalib-phantoms/path-1001249258-20250609-131851-966003.loraw",
    "/Users/sanya/Desktop/uncalib-phantoms/path-1001249258-20250609-131852-563379.loraw",
    "/Users/sanya/Desktop/uncalib-phantoms/path-1001249258-20250609-131853-002999.loraw",
    "/Users/sanya/Desktop/uncalib-phantoms/path-1001249258-20250609-131856-366533.loraw",
    "/Users/sanya/Desktop/uncalib-phantoms/path-1001249258-20250609-131856-768569.loraw",
    "/Users/sanya/Desktop/uncalib-phantoms/path-1001249258-20250609-131904-075038.loraw",
    "/Users/sanya/Desktop/uncalib-phantoms/path-1001249258-20250609-131904-491091.loraw",
    "/Users/sanya/Desktop/uncalib-phantoms/path-1001249258-20250609-131911-482970.loraw"
]
CALIBRATION_FOLDER = "/Users/sanya/Develop/lo_own_test/MBLL/data/zoom_calibration"
RGB_SAVE_FOLDER = "./phantom_images_rgb"
CACHE_FILE = "predictions_cache.json"

from lo.sdk.api.acquisition.io.open import open as LoOpen
from lo.sdk.api.acquisition.data.formats import LORAWtoRGB8
from lo.sdk.api.acquisition.data.decode import SpectralDecoder

decoder = SpectralDecoder.from_calibration(CALIBRATION_FOLDER, None)
os.makedirs(RGB_SAVE_FOLDER, exist_ok=True)

X_all, y_all = [], []
predictions_cache = {}

def process_and_save(file_path, label):
    with LoOpen(file_path) as f:
        f.seek(0)
        frame = f.read()
        metadata, scene_frame, spectra = decoder(frame, LORAWtoRGB8)
        coords = metadata.sampling_coordinates
        gray = scene_frame.mean(axis=2)
        intensities, valid_indices = [], []
        for i, (x, y) in enumerate(coords):
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < gray.shape[0] and 0 <= xi < gray.shape[1]:
                intensities.append(gray[yi, xi])
                valid_indices.append(i)
        spectra_filtered = spectra[valid_indices]
        spectra_norm = spectra_filtered / (np.linalg.norm(spectra_filtered, axis=1, keepdims=True) + 1e-8)
        X_all.append(spectra_norm)
        y_all.append(np.full(len(spectra_norm), label))

        file_name = Path(file_path).stem + ".png"
        save_path = os.path.join(RGB_SAVE_FOLDER, file_name)
        norm_rgb = scene_frame.astype(np.float32) / 255.0
        norm_rgb = (norm_rgb / (norm_rgb.max() + 1e-8) * 255).astype(np.uint8)
        Image.fromarray(norm_rgb).save(save_path)

        return save_path, label, spectra_norm, valid_indices, coords

all_files = [(f, 0) for f in NORMAL_FILES] + [(f, 1) for f in PATHO_FILES]

for file_path, label in all_files:
    rgb_path, lbl, spectra_norm, valid_indices, coords = process_and_save(file_path, label)
    print(f"Processed {file_path} → {rgb_path}")

X = np.vstack(X_all)
y = np.hstack(y_all)
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_pca, y)

# Save model predictions for each file
for file_path, label in all_files:
    with LoOpen(file_path) as f:
        f.seek(0)
        frame = f.read()
        metadata, scene_frame, spectra = decoder(frame, LORAWtoRGB8)
        coords = metadata.sampling_coordinates
        gray = scene_frame.mean(axis=2)
        intensities, valid_indices = [], []
        for i, (x, y) in enumerate(coords):
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < gray.shape[0] and 0 <= xi < gray.shape[1]:
                intensities.append(gray[yi, xi])
                valid_indices.append(i)
        spectra_filtered = spectra[valid_indices]
        spectra_norm = spectra_filtered / (np.linalg.norm(spectra_filtered, axis=1, keepdims=True) + 1e-8)
        spectra_pca = pca.transform(spectra_norm)
        pixel_preds = clf.predict(spectra_pca).tolist()
        model_label = int(np.round(np.mean(pixel_preds)).item())
        valid_coords = [(float(coords[i][0]), float(coords[i][1])) for i in valid_indices]
        predictions_cache[file_path] = {
            "model_label": model_label,
            "pixel_preds": pixel_preds,
            "valid_coords": valid_coords
        }

with open(CACHE_FILE, "w") as f:
    json.dump(predictions_cache, f, indent=2)

print(f"✅ All RGB images saved to: {RGB_SAVE_FOLDER}")
print(f"✅ Prediction cache written to: {CACHE_FILE}")

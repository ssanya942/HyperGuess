import streamlit as st
import numpy as np
import os
import json
import random
import tempfile
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from lo.sdk.api.acquisition.data.coordinates import NearestUpSample
from lo.sdk.api.acquisition.io.open import open as LoOpen
from lo.sdk.api.acquisition.data.formats import LORAWtoRGB8
from lo.sdk.api.acquisition.data.decode import SpectralDecoder

# === Configurable paths ===
CALIBRATION_FOLDER = "/Users/sanya/Develop/lo_own_test/MBLL/data/zoom_calibration"
FIELD_CALIBRATION_FILE = None
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
PREDICTIONS_CACHE = "predictions_cache.json"

decoder = SpectralDecoder.from_calibration(CALIBRATION_FOLDER, FIELD_CALIBRATION_FILE)


def process_file(file_path, label):
    with LoOpen(file_path) as f:
        f.seek(0)
        frame = f.read()
        metadata, scene_frame, spectra = decoder(frame, LORAWtoRGB8)
        coords = metadata.sampling_coordinates
        gray = scene_frame.mean(axis=2)
        intensities = np.array([
            gray[int(round(y)), int(round(x))] for x, y in coords
            if 0 <= int(round(y)) < gray.shape[0] and 0 <= int(round(x)) < gray.shape[1]
        ])
        top_indices = np.argsort(intensities)[-spectra.shape[0]:]
        spectra_filtered = spectra[top_indices]
        spectra_norm = spectra_filtered / (np.linalg.norm(spectra_filtered, axis=1, keepdims=True) + 1e-8)
        labels = np.full(spectra_norm.shape[0], label)
        return spectra_norm, labels


@st.cache_resource
def get_model():
    X_all, y_all = [], []
    for file in NORMAL_FILES:
        X, y = process_file(file, 0)
        X_all.append(X)
        y_all.append(y)
    for file in PATHO_FILES:
        X, y = process_file(file, 1)
        X_all.append(X)
        y_all.append(y)
    X = np.vstack(X_all)
    y = np.hstack(y_all)
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_pca, y)
    return clf, pca


clf, pca = get_model()

# === Safe load for predictions cache ===
if os.path.exists(PREDICTIONS_CACHE):
    try:
        with open(PREDICTIONS_CACHE, "r") as f:
            predictions_cache = json.load(f)
    except json.JSONDecodeError:
        st.warning("⚠️ Cache file was corrupted. Resetting.")
        os.remove(PREDICTIONS_CACHE)
        predictions_cache = {}
else:
    predictions_cache = {}


def safe_json_dump(data, filename):
    with tempfile.NamedTemporaryFile('w', delete=False, dir=os.path.dirname(filename)) as tf:
        json.dump(data, tf)
        tempname = tf.name
    os.replace(tempname, filename)


@st.cache_data(show_spinner=False)
def load_demo_images():
    selected = random.sample(NORMAL_FILES, 3) + random.sample(PATHO_FILES, 2)
    all_files = [(f, 0 if "normal" in f else 1) for f in selected]
    random.shuffle(all_files)
    images = []
    for file_path, label in all_files:
        with LoOpen(file_path) as f:
            f.seek(0)
            frame = f.read()
            metadata, scene_frame, spectra = decoder(frame, LORAWtoRGB8)
            rgb_image = scene_frame.astype(np.float32) / 255.0
            rgb_image = rgb_image / (rgb_image.max() + 1e-8)
            images.append({
                "file": file_path,
                "label": label,
                "metadata": metadata,
                "scene_frame": scene_frame,
                "spectra": spectra,
                "coords": metadata.sampling_coordinates,
                "rgb": rgb_image
            })
    return images


st.title("🧠 HyperGuess")

if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "round_complete" not in st.session_state:
    st.session_state.round_complete = False
if "audience_answers" not in st.session_state:
    st.session_state.audience_answers = {}

images = load_demo_images()

if st.session_state.current_idx >= len(images):
    st.session_state.round_complete = True

if not st.session_state.round_complete:
    img = images[st.session_state.current_idx]
    file_id = img["file"]

    st.image(np.clip(img["rgb"], 0.0, 1.0), caption=f"Image {st.session_state.current_idx+1}: Audience Inspection", use_container_width=True)

    audience_choice = st.radio("What do you think this is?", ("Normal", "Pathological"), key=f"choice_{st.session_state.current_idx}")

    col1, col2 = st.columns(2)
    if col1.button("⬅️ Previous"):
        st.session_state.current_idx = max(0, st.session_state.current_idx - 1)
    if col2.button("Next ➡️"):
        st.session_state.current_idx = min(len(images), st.session_state.current_idx + 1)

    if st.button("🔍 Reveal Model Prediction", key=f"predict_{st.session_state.current_idx}"):
        if file_id in predictions_cache:
            entry = predictions_cache[file_id]
            model_label = entry["model_label"]
        else:
            scene_frame = img["scene_frame"]
            coords = img["coords"]
            spectra = img["spectra"]
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
            valid_coords = [(float(img["coords"][i][0]), float(img["coords"][i][1])) for i in valid_indices]
            predictions_cache[file_id] = {
                "model_label": model_label,
                "pixel_preds": pixel_preds,
                "valid_coords": valid_coords
            }
            safe_json_dump(predictions_cache, PREDICTIONS_CACHE)

        st.session_state.audience_answers[file_id] = audience_choice
        st.markdown(f"**Audience Answer:** {audience_choice}")
        st.markdown(f"**Model Prediction:** {'Normal' if model_label == 0 else 'Pathological'}")
        st.markdown(f"**Ground Truth:** {'Normal' if img['label'] == 0 else 'Pathological'}")

        st.progress((st.session_state.current_idx + 1) / len(images), text=f"Question {st.session_state.current_idx + 1} of {len(images)}")

else:
    st.success("✅ You've completed all 5 questions!")
    if st.button("🔁 Restart"):
        st.session_state.current_idx = 0
        st.session_state.round_complete = False
        st.session_state.audience_answers = {}
        st.cache_data.clear()
        st.rerun()

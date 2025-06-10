import streamlit as st
import os
import json
import random
from PIL import Image

# === Configuration ===
PREDICTIONS_CACHE = "/Users/sanya/Develop/lo_own_test/MBLL/app/predictions_cache.json"
RGB_IMAGE_DIR = "/Users/sanya/Develop/lo_own_test/MBLL/app/phantom_images_rgb"

# === Load predictions ===
with open(PREDICTIONS_CACHE, "r") as f:
    predictions = json.load(f)

# === App Title ===
st.title("🧠 HyperGuess")

# === Initialize session state ===
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "round_complete" not in st.session_state:
    st.session_state.round_complete = False
if "audience_answers" not in st.session_state:
    st.session_state.audience_answers = {}
if "shuffled_images" not in st.session_state:
    image_files = list(predictions.keys())
    random.shuffle(image_files)
    st.session_state.shuffled_images = [
        {"file": f, "label": predictions[f]["model_label"]} for f in image_files[:5]
    ]

images = st.session_state.shuffled_images

# === End of round check ===
if st.session_state.current_idx >= len(images):
    st.session_state.round_complete = True

# === Main Quiz ===
if not st.session_state.round_complete:
    img = images[st.session_state.current_idx]
    img_path = os.path.join(
        RGB_IMAGE_DIR, os.path.basename(img["file"]).replace(".loraw", ".png")
    )

    st.image(
        img_path,
        caption=f"Image {st.session_state.current_idx + 1} of {len(images)}",
        use_container_width=True,
    )

    # === Use fixed key for stable radio state ===
    if f"choice_{st.session_state.current_idx}" not in st.session_state:
        st.session_state[f"choice_{st.session_state.current_idx}"] = "Normal"

    audience_choice = st.radio(
        "What do you think this is?",
        ("Normal", "Pathological"),
        index=0 if st.session_state[f"choice_{st.session_state.current_idx}"] == "Normal" else 1,
        key=f"radio_{st.session_state.current_idx}",
    )
    st.session_state[f"choice_{st.session_state.current_idx}"] = audience_choice

    # === Navigation buttons ===
    col1, col2 = st.columns(2)
    if col1.button("⬅️ Previous"):
        st.session_state.current_idx = max(0, st.session_state.current_idx - 1)
        st.rerun()
    if col2.button("Next ➡️"):
        st.session_state.current_idx = min(len(images), st.session_state.current_idx + 1)
        st.rerun()

    # === Check Answer button ===
    if st.button("✅ Check Answer"):
        model_label = img["label"]
        ground_truth = 'Normal' if model_label == 0 else 'Pathological'
        audience_ans = audience_choice
        st.session_state.audience_answers[img["file"]] = audience_ans

        # Display answers
        st.markdown(f"**Audience Answer:** {audience_ans}")
        st.markdown(f"**Model Prediction:** {ground_truth}")
        st.markdown(f"**Ground Truth:** {ground_truth}")

        # Feedback
        if audience_ans == ground_truth:
            st.success("🎉 Correct! You matched the ground truth.")
        else:
            st.error("😬 Oops — AI beat you this time.")

        st.progress(
            (st.session_state.current_idx + 1) / len(images),
            text=f"Question {st.session_state.current_idx + 1} of {len(images)}"
        )

# === Round Complete Screen ===
else:
    # Score calculation
    correct = 0
    for img in images:
        fname = img["file"]
        gt = 'Normal' if img["label"] == 0 else 'Pathological'
        if st.session_state.audience_answers.get(fname) == gt:
            correct += 1

    st.success("✅ You've completed all 5 questions!")
    st.markdown(f"### 🧮 Your Score: {correct} / {len(images)} ")

    if st.button("🔁 Restart"):
        st.session_state.current_idx = 0
        st.session_state.round_complete = False
        st.session_state.audience_answers = {}
        image_files = list(predictions.keys())
        random.shuffle(image_files)
        st.session_state.shuffled_images = [
            {"file": f, "label": predictions[f]["model_label"]} for f in image_files[:5]
        ]
        st.rerun()

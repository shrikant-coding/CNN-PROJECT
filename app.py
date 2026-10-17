
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "cifar100_cnn.keras"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"

st.set_page_config(
    page_title="CIFAR-100 CNN Classifier",
    page_icon="🧠",
    layout="wide"
)


@st.cache_resource
def load_trained_model():
    if not MODEL_PATH.exists():
        return None

    return tf.keras.models.load_model(str(MODEL_PATH))


@st.cache_data
def load_metadata():
    if not METADATA_PATH.exists():
        return None

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


st.title("🧠 CNN Image Classification")
st.caption("Artificial Neural Networks and Deep Learning Project")

st.markdown("""
This application uses a Convolutional Neural Network (CNN) trained
on the **CIFAR-100 dataset**, containing 60,000 images across 100 classes.

Upload an image to see the model's predicted class and its top five results.
""")

if not MODEL_PATH.exists():
    st.error("The trained model was not found.")
    st.warning(
        "Run `python -u train.py` in your project folder first. "
        "Wait for training to finish, then restart this app."
    )
    st.code(
        "cd /d \"D:\\CNN PROJECT\"\n"
        "python -u train.py\n"
        "python -m streamlit run app.py",
        language="powershell"
    )
    st.stop()

try:
    model = load_trained_model()
    metadata = load_metadata()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

if model is None:
    st.error("Unable to load the model file.")
    st.stop()

if metadata is None:
    st.error("metadata.json is missing. Run train.py again.")
    st.stop()

class_names = metadata["class_names"]

left, right = st.columns(2)

with left:
    st.metric("Dataset images", "60,000")

with right:
    st.metric("Classes", "100")

if "test_accuracy" in metadata:
    st.metric(
        "Test accuracy",
        f"{metadata['test_accuracy'] * 100:.2f}%"
    )

st.divider()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Uploaded image")
            st.image(image, use_container_width=True)

        # CIFAR-100 images are 32x32 RGB images.
        resized_image = image.resize(
            (32, 32),
            Image.Resampling.LANCZOS
        )

        image_array = np.asarray(
            resized_image, dtype=np.float32
        ) / 255.0

        image_array = np.expand_dims(image_array, axis=0)

        with st.spinner("Classifying image..."):
            probabilities = model.predict(
                image_array, verbose=0
            )[0]

        top_indices = np.argsort(probabilities)[::-1][:5]
        best_index = int(top_indices[0])
        best_class = class_names[best_index]
        confidence = float(probabilities[best_index]) * 100

        with col2:
            st.subheader("Prediction")
            st.success(
                f"Predicted class: {best_class.replace('_', ' ').title()}"
            )
            st.metric("Model confidence", f"{confidence:.2f}%")

            st.subheader("Top 5 predictions")

            for index in top_indices:
                label = class_names[int(index)].replace("_", " ").title()
                score = float(probabilities[index])

                st.write(f"**{label}** — {score * 100:.2f}%")
                st.progress(score)

        st.info(
            "Note: CIFAR-100 contains small, low-resolution images. "
            "Predictions on ordinary real-world photos may be inaccurate."
        )

    except Exception as error:
        st.error(f"Could not process the image: {error}")

st.divider()
st.caption("CNN architecture: Convolution, Batch Normalization, "
           "Max Pooling, Dropout, Dense and Softmax.")

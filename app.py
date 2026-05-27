import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input

# =========================================================
# Page Config
# =========================================================

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩺",
    layout="centered"
)

# =========================================================
# Load Trained Model
# =========================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "model.keras"
    )

    return model

model = load_model()

# =========================================================
# Constants
# =========================================================

IMG_SIZE = 224

# =========================================================
# OpenCV Preprocessing
# =========================================================

def preprocess_image(image):

    # PIL -> NumPy
    image = np.array(image)

    # Resize
    image = cv2.resize(image, (224, 224))

    # Check channels
    if len(image.shape) == 3:

        # RGB image
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

    else:

        # Already grayscale
        gray = image

    # Gaussian Blur
    blur = cv2.GaussianBlur(
        gray,
        (5,5),
        0
    )

    # CLAHE Enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    enhanced = clahe.apply(blur)

    # Convert grayscale -> RGB
    image = cv2.cvtColor(
        enhanced,
        cv2.COLOR_GRAY2RGB
    )

    # DenseNet preprocessing
    image = preprocess_input(image)

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    return image

# =========================================================
# Prediction Function
# =========================================================

def predict_pneumonia(image):

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)[0][0]

    return prediction

# =========================================================
# Streamlit UI
# =========================================================

st.title("🩺 Pneumonia Detection System")

st.write(
    """
    Upload a Chest X-ray image to detect
    whether the patient has Pneumonia.
    """
)

# =========================================================
# File Upload
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

# =========================================================
# Prediction Section
# =========================================================

if uploaded_file is not None:

    # Open Image
    image = Image.open(uploaded_file)

    # Display Image
    st.image(
        image,
        caption="Uploaded Chest X-ray",
        use_container_width=True
    )

    # Predict Button
    if st.button("Predict"):

        with st.spinner("Analyzing X-ray..."):

            prediction = predict_pneumonia(image)

            confidence = float(prediction)

            # Threshold
            if prediction > 0.5:

                st.error(
                    f"⚠️ Pneumonia Detected\n\n"
                    f"Confidence: {confidence:.2%}"
                )

            else:

                st.success(
                    f"✅ Normal Chest X-ray\n\n"
                    f"Confidence: {(1-confidence):.2%}"
                )

# =========================================================
# Footer
# =========================================================

st.markdown("---")

st.markdown(
    "Built using DenseNet121 + OpenCV + Streamlit"
)



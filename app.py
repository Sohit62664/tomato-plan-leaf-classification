import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops

# ------------------------------------------------------------
# APP CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Tomato Leaf Disease Classification",
    page_icon="🍅",
    layout="centered"
)

MODEL_PATH = "model.joblib"
IMG_SIZE = 128

CLASS_NAMES = {
    "Tomato___healthy": "Healthy",
    "Tomato___Bacterial_spot": "Bacterial Spot",
    "Tomato___Early_blight": "Early Blight",
    "Tomato___Late_blight": "Late Blight"
}


# ------------------------------------------------------------
# LOAD TRAINED MODEL
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ------------------------------------------------------------
# FEATURE EXTRACTION
# Same preprocessing/features used during training
# ------------------------------------------------------------

def extract_features_from_image(img):

    if img is None:
        return None, None, None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # OpenCV image is BGR -> RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Gaussian blur
    blurred = cv2.GaussianBlur(rgb, (5, 5), 0)

    # RGB -> HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)

    # --------------------------------------------------------
    # Leaf segmentation
    # --------------------------------------------------------

    lower_green = np.array([25, 25, 20])
    upper_green = np.array([100, 255, 255])

    mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Same fallback used during training
    if np.count_nonzero(mask) < 0.05 * mask.size:
        mask = np.ones_like(mask) * 255

    # --------------------------------------------------------
    # COLOR FEATURES
    # --------------------------------------------------------

    pixels_rgb = rgb[mask > 0]
    pixels_hsv = hsv[mask > 0]

    if len(pixels_rgb) == 0:
        return None, mask, None

    mean_r = np.mean(pixels_rgb[:, 0])
    mean_g = np.mean(pixels_rgb[:, 1])
    mean_b = np.mean(pixels_rgb[:, 2])

    mean_h = np.mean(pixels_hsv[:, 0])
    mean_s = np.mean(pixels_hsv[:, 1])
    mean_v = np.mean(pixels_hsv[:, 2])

    green_ratio = np.mean(
        (hsv[:, :, 0] >= 25) &
        (hsv[:, :, 0] <= 100)
    )

    yellow_ratio = np.mean(
        (hsv[:, :, 0] >= 15) &
        (hsv[:, :, 0] <= 35) &
        (hsv[:, :, 1] > 50)
    )

    # --------------------------------------------------------
    # TEXTURE FEATURES - GLCM
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    gray_quantized = (gray / 32).astype(np.uint8)

    glcm = graycomatrix(
        gray_quantized,
        distances=[1],
        angles=[0],
        levels=8,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(
        glcm, "contrast"
    )[0, 0]

    correlation = graycoprops(
        glcm, "correlation"
    )[0, 0]

    energy = graycoprops(
        glcm, "energy"
    )[0, 0]

    homogeneity = graycoprops(
        glcm, "homogeneity"
    )[0, 0]

    # --------------------------------------------------------
    # SHAPE FEATURES
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:

        largest = max(
            contours,
            key=cv2.contourArea
        )

        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(
            largest,
            True
        )

        if perimeter > 0:
            circularity = (
                4 * np.pi * area /
                (perimeter * perimeter)
            )
        else:
            circularity = 0

    else:
        area = 0
        perimeter = 0
        circularity = 0

    features = np.array([[
        mean_r,
        mean_g,
        mean_b,
        mean_h,
        mean_s,
        mean_v,
        green_ratio,
        yellow_ratio,
        contrast,
        correlation,
        energy,
        homogeneity,
        area,
        perimeter,
        circularity
    ]])

    segmented = cv2.bitwise_and(
        rgb,
        rgb,
        mask=mask
    )

    return features, mask, segmented


# ------------------------------------------------------------
# USER INTERFACE
# ------------------------------------------------------------

st.title("🍅 Tomato Leaf Disease Classification")

st.write(
    "Upload a tomato leaf image to classify it as "
    "Healthy, Bacterial Spot, Early Blight, or Late Blight."
)

st.info(
    "This is an AI-based preliminary classification system "
    "using Digital Image Processing, GLCM features, and "
    "an RBF-kernel SVM."
)

uploaded_file = st.file_uploader(
    "Upload a tomato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:
        st.error("Could not read the uploaded image.")
        st.stop()

    features, mask, segmented = extract_features_from_image(
        image
    )

    if features is None:
        st.error(
            "Unable to extract features from this image."
        )
        st.stop()

    st.subheader("Image Processing")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            caption="Original",
            use_container_width=True
        )

    with col2:
        st.image(
            mask,
            caption="Leaf Mask",
            use_container_width=True
        )

    with col3:
        st.image(
            segmented,
            caption="Segmented Image",
            use_container_width=True
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    try:
        model = load_model()

        prediction = model.predict(features)[0]

        display_name = CLASS_NAMES.get(
            prediction,
            prediction
        )

        st.subheader("Prediction")

        if display_name == "Healthy":
            st.success(
                f"Prediction: {display_name}"
            )
        else:
            st.warning(
                f"Prediction: {display_name}"
            )

        st.caption(
            "The model was trained on four tomato leaf classes "
            "from the PlantVillage dataset."
        )

    except FileNotFoundError:
        st.error(
            "model.joblib was not found. "
            "Place model.joblib in the same folder as app.py."
        )

    except Exception as e:
        st.error(
            f"Prediction failed: {str(e)}"
        )


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.markdown("---")
st.caption(
    "Tomato Leaf Disease Classification | "
    "Image Processing + GLCM + RBF-SVM"
)

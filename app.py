import streamlit as st
import numpy as np
import cv2
from skimage import filters, segmentation, color, exposure
from skimage.util import img_as_float
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Interactive Digital Image Processing Application",
    layout="wide",
    page_icon="🖼️"
)

# ================================
# HEADER
# ================================
st.markdown("""
# **Interactive Digital Image Processing Application**
Enhancement • FFT • Color Processing • Segmentation  
""")

# ================================
# IMAGE UPLOADER
# ================================
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"],
    help="Upload an image file for processing"
)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = img_as_float(img_rgb)

    st.subheader("Original Image")
    st.image(img_rgb, caption="Original Image", width="content")

    # =====================================================
    # SECTION 1 — IMAGE ENHANCEMENT
    # =====================================================
    st.header("Image Enhancement")

    col1, col2, col3 = st.columns(3)

    with col1:
        brightness = st.slider("Brightness", -100, 100, 0)
    with col2:
        contrast = st.slider("Contrast", 0.5, 3.0, 1.0)
    with col3:
        blur_val = st.slider("Gaussian Blur", 0, 25, 0)

    enhanced = img_rgb.astype(np.float32)
    enhanced = enhanced * contrast + brightness
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    if blur_val > 0:
        enhanced = cv2.GaussianBlur(enhanced, (blur_val * 2 + 1, blur_val * 2 + 1), 0)

    st.subheader("Enhanced Image")
    st.image(enhanced, width="content")

    # =====================================================
    # SECTION 2 — FFT
    # =====================================================
    st.header("Frequency Domain (FFT)")

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    st.subheader("FFT Magnitude Spectrum")
    st.image(magnitude_spectrum, clamp=True, width="content")

    # =====================================================
    # SECTION 3 — COLOR PROCESSING
    # =====================================================
    st.header("Color Processing")

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Grayscale")
        gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        st.image(gray_img, width="content")

    with colB:
        st.subheader("HSV Color Space")
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        hsv_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        st.image(hsv_vis, width="content")

    # =====================================================
    # SECTION 4 — SEGMENTATION
    # =====================================================
    st.header("Segmentation")

    seg_method = st.selectbox(
        "Choose segmentation method",
        ["Otsu Thresholding", "Canny Edge Detection", "SLIC Superpixel"]
    )

    if seg_method == "Otsu Thresholding":
        thresh = filters.threshold_otsu(gray)
        binary = gray > thresh
        st.image(binary, caption="Otsu Threshold Result", width="content")

    elif seg_method == "Canny Edge Detection":
        edge_low = st.slider("Low Threshold", 10, 200, 50)
        edge_high = st.slider("High Threshold", 100, 300, 150)
        edges = cv2.Canny(gray, edge_low, edge_high)
        st.image(edges, caption="Canny Edge Result", width="content")

    elif seg_method == "SLIC Superpixel":
        n_segments = st.slider("Number of Segments", 50, 400, 200)
        segments = segmentation.slic(img_float, n_segments=n_segments, start_label=1)
        colored = color.label2rgb(segments, img_rgb, kind="avg")
        st.image(colored, caption="SLIC Superpixel Segmentation", width="content")

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.title("🖼️ Image Outline Redrawer")

# Step 1: User uploads an image
uploaded_file = st.file_uploader("Upload a PNG or JPEG image", type=["png", "jpg", "jpeg"])
x = st.number_input("Enter Width (X)", min_value=10, max_value=2000, value=240)
y = st.number_input("Enter Height (Y)", min_value=10, max_value=2000, value=400)

if uploaded_file is not None:
    # Step 2: Convert to OpenCV format
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Step 3: Resize
    resized = cv2.resize(image_cv, (x, y), interpolation=cv2.INTER_AREA)

    # Step 4: Grayscale + Blur
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 5: Adaptive Threshold
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    # Step 6: Dilate (optional)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Step 7: Invert for white background
    final_outline = cv2.bitwise_not(dilated)

    # Step 8: Show and Download
    st.image(final_outline, caption="Outlined Image", channels="GRAY")

    # Download button
    result_pil = Image.fromarray(final_outline)
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button("Download Outline", byte_im, file_name="outlined.png", mime="image/png")

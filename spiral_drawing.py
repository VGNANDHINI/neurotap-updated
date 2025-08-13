import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.title("🖌️ Spiral Drawing Parkinson's Risk Assessment")

# ---------------- Dot Spiral Background ----------------
def create_dot_spiral(size=400, points=800):
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    center = size//2
    a, b = 5, 2
    for i in range(points):
        theta = i * 0.3
        r = a + b*theta
        x = int(center + r * np.cos(theta))
        y = int(center + r * np.sin(theta))
        if 0 <= x < size and 0 <= y < size:
            img[y, x] = [0,0,0]  # black dot
    return img

dot_spiral = create_dot_spiral()

# ---------------- Drawing Canvas ----------------
canvas_result = st_canvas(
    stroke_width=3,
    stroke_color="#FF0000",
    background_image=Image.fromarray(dot_spiral),
    height=400,
    width=400,
    drawing_mode="freedraw",
    key="canvas",
)

# ---------------- Analysis ----------------
def analyze_spiral(drawing_array, dot_spiral):
    # 1. Convert to binary arrays: drawn vs dots
    drawn = drawing_array[:,:,0] < 250  # True where user drew
    dots = dot_spiral[:,:,0] < 10       # True where spiral dots are
    # 2. Compute alignment: how many drawn pixels are on/near dots
    overlap = np.logical_and(drawn, dots)
    coverage = overlap.sum() / dots.sum()  # fraction of dots traced
    # 3. Compute extra stray pixels (outside spiral)
    stray_pixels = np.logical_and(drawn, np.logical_not(dots))
    stray_ratio = stray_pixels.sum() / drawn.sum() if drawn.sum()>0 else 0
    # 4. Simple heuristic risk scoring
    # Higher coverage + low stray -> low risk
    score = 100 * (1 - coverage + stray_ratio)
    score = np.clip(score,0,100)
    # Risk level
    if score < 30:
        level = "Low"
        warning = "✅ Your spiral tracing is smooth and accurate. Continue regular monitoring."
    elif score < 60:
        level = "Moderate"
        warning = "⚠️ Some irregularities detected. Consider consulting a neurologist."
    else:
        level = "High"
        warning = "❌ Significant tremor detected. Immediate medical consultation recommended."
    return score, level, warning

# ---------------- Show Result Button ----------------
if canvas_result.image_data is not None:
    if st.button("Show Result"):
        img_array = np.array(canvas_result.image_data)
        score, level, warning = analyze_spiral(img_array, dot_spiral)
        st.subheader("🧠 Parkinson's Risk Assessment")
        st.write(f"Risk Score: {score:.2f}/100")
        st.write(f"Risk Level: {level}")
        st.write(f"Guidance: {warning}")
        # Optional: show overlay of drawn pixels
        overlay = np.array(canvas_result.image_data)
        overlay[np.logical_not(overlay[:,:,0]<250)] = 255
        st.image(overlay, caption="Your Tracing Overlay")

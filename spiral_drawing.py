# spiral_drawing.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image

def create_dot_spiral(size=400, dots=40):
    """Create a dot spiral as background for user tracing."""
    img = np.ones((size, size), dtype=np.uint8) * 255  # white background
    cx, cy = size//2, size//2
    max_radius = size//2 - 10
    theta = np.linspace(0, 4*np.pi, dots)
    r = np.linspace(10, max_radius, dots)
    for i in range(dots):
        x = int(cx + r[i]*np.cos(theta[i]))
        y = int(cy + r[i]*np.sin(theta[i]))
        if 0 <= x < size and 0 <= y < size:
            img[y, x] = 0  # black dot
    return img

def analyze_spiral(drawing_array):
    """Simple heuristic risk analysis based on stroke smoothness and gaps."""
    # Normalize
    norm = drawing_array / 255.0
    stroke_pixels = np.sum(norm < 0.5)
    total_pixels = norm.size
    coverage = stroke_pixels / total_pixels

    # Heuristic for tremor: measure pixel roughness
    grad = np.gradient(norm.astype(float))
    roughness = np.mean(np.abs(grad[0]) + np.abs(grad[1]))

    # Risk calculation
    risk_score = min(100, (roughness*200 + (0.5 - coverage)*100))
    if risk_score < 30:
        risk_level = "Low"
        suggestion = "Your spiral drawing looks smooth and consistent. No immediate concern."
    elif risk_score < 60:
        risk_level = "Moderate"
        suggestion = "Some irregularities detected. Consider consulting a neurologist for precaution."
    else:
        risk_level = "High"
        suggestion = "High irregularities detected. Strongly consider clinical assessment for Parkinson’s risk."

    return risk_score, risk_level, suggestion

def show_spiral_canvas():
    """Display Streamlit canvas and analyze spiral drawing on button click."""
    st.header("🖌️ Spiral Drawing Test")
    st.write("Trace the spiral dots as accurately as possible.")

    # Create dot spiral PIL image
    dot_spiral = create_dot_spiral()
    dot_spiral_img = Image.fromarray(dot_spiral)

    # Canvas
    canvas_result = st_canvas(
        stroke_width=3,
        stroke_color="#FF0000",
        background_image=dot_spiral_img,
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="canvas",
    )

    # Show user drawing
    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data)

        # Button to analyze
        if st.button("Show Result"):
            drawing_array = np.array(canvas_result.image_data[:, :, 0])
            risk_score, risk_level, suggestion = analyze_spiral(drawing_array)

            st.subheader("🩺 Spiral Drawing Analysis Result")
            st.write(f"Risk Score: {risk_score:.2f}/100")
            st.write(f"Risk Level: {risk_level}")
            st.write(f"Clinical Suggestion: {suggestion}")

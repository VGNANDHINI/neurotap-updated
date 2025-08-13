# parkinson_spiral_app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np

st.title("Parkinson's Spiral Drawing Test 🌀")
st.write("Draw a spiral and click 'Analyze Result' to see a simulated diagnosis.")

# Step 1: Drawing canvas
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=3,
    stroke_color="#000000",
    background_color="#FFFFFF",
    update_streamlit=True,
    height=400,
    width=400,
    drawing_mode="freedraw",
    key="canvas",
)

# Step 2: Analyze Button
if st.button("Analyze Result"):
    if canvas_result.image_data is None:
        st.warning("Please draw a spiral first.")
    else:
        # Convert image to grayscale
        img_gray = np.mean(canvas_result.image_data[:, :, :3], axis=2)

        # Measure "wobbliness" - just a fake metric for demonstration
        stroke_pixels = np.sum(img_gray < 200)  # Count drawn pixels
        variation = np.std(img_gray[img_gray < 200])  # Variation in stroke darkness

        # Simple logic for diagnosis
        if stroke_pixels < 5000:
            diagnosis = "Drawing too small - test not valid."
        elif variation > 50:
            diagnosis = "Possible Parkinson's detected - Please consult a neurologist."
        else:
            diagnosis = "Normal drawing pattern - Parkinson's unlikely."

        st.subheader("Diagnosis Result:")
        st.write(diagnosis)

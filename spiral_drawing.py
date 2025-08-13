# spiral_drawing.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np

def show_spiral_canvas():
    st.subheader("Draw Your Spiral Below")

    # Canvas for drawing
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",      # Transparent fill
        stroke_width=3,
        stroke_color="#000000",          # Black pen
        background_color="#FFFFFF",      # White background
        update_streamlit=True,
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Analyse Result"):
        if canvas_result.image_data is not None:
            # Very simple fake "analysis"
            pixel_data = np.array(canvas_result.image_data)
            black_pixels = np.sum(pixel_data[:, :, 0] < 10)  # Count black pixels
            if black_pixels > 2000:
                st.error("Result: High drawing deviation — Possible Parkinson's sign.")
            else:
                st.success("Result: Drawing looks normal.")
        else:
            st.warning("Please draw a spiral first!")

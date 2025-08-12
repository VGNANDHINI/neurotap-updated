# spiral_drawing.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np

def show_spiral_canvas():
    st.header("🖌️ Spiral Drawing Test")
    canvas_result = st_canvas(
        stroke_width=2,
        stroke_color="#000000",
        background_color="#fff",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="spiral_canvas",
    )
    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data)
        drawing_array = np.array(canvas_result.image_data[:, :, 0])
        # add feature extraction here

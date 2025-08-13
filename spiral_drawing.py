from streamlit_drawable_canvas import st_canvas
import streamlit as st
import numpy as np
from PIL import Image

def create_dot_spiral(size=400, dots=30):
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

def show_spiral_canvas():
    st.header("🖌️ Spiral Drawing Test")

    # create PIL image for dot spiral
    dot_spiral = create_dot_spiral()
    dot_spiral_img = Image.fromarray(dot_spiral.astype('uint8'))

    # Canvas
    canvas_result = st_canvas(
        stroke_width=3,
        stroke_color="#FF0000",
        background_image=dot_spiral_img,  # must be PIL Image
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data)

        if st.button("Analyze Spiral"):
            # convert canvas data to grayscale array for analysis
            drawing_array = np.array(canvas_result.image_data[:, :, 0])
            # call your analysis function here, e.g., analyze_spiral(drawing_array)
            st.success("Analysis completed! (Placeholder for risk score and suggestions)")

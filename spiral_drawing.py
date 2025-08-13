import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image

def show_spiral_canvas():
    st.header("🖌️ Spiral Drawing Test")

    # Create a simple dot spiral
    size = 400
    arr = np.full((size, size), 255, dtype=np.uint8)
    center = size // 2
    spacing = 20
    for r in range(spacing, center, spacing):
        theta = np.linspace(0, 4*np.pi, 8*r)
        x = (r * np.cos(theta) + center).astype(int)
        y = (r * np.sin(theta) + center).astype(int)
        arr[y.clip(0,size-1), x.clip(0,size-1)] = 0

    background_img = Image.fromarray(np.stack([arr]*3, axis=2))

    # Canvas for drawing
    canvas_result = st_canvas(
        stroke_width=3,
        stroke_color="#FF0000",
        background_image=background_img,
        height=size,
        width=size,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data, caption="Your Drawing")
        if st.button("Analyze Drawing"):
            img_array = np.array(canvas_result.image_data[:, :, 0])
            stroke_pixels = np.sum(img_array < 200)
            coverage = stroke_pixels / img_array.size
            risk_score = int(coverage * 100)

            if risk_score < 20:
                level = "Low"
                suggestion = "Spiral looks normal."
            elif risk_score < 40:
                level = "Moderate"
                suggestion = "Slight irregularities. Consider checking with a doctor."
            else:
                level = "High"
                suggestion = "Significant irregularities detected. Recommend medical evaluation."

            st.subheader("Result")
            st.write(f"**Risk Score:** {risk_score}/100")
            st.write(f"**Risk Level:** {level}")
            st.write(f"**Suggestion:** {suggestion}")

import streamlit as st
from streamlit_drawable_canvas import st_canvas

def show_spiral_canvas():
    st.write("Draw a spiral below:")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=2,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="spiral",
    )
    if st.button("Analyse Drawing"):
        st.success("Analysis complete. (This is just a placeholder.)")

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
from skimage import img_as_bool
from scipy.signal import detrend
from scipy.fftpack import fft, fftfreq
from scipy.ndimage import binary_fill_holes

# ------------------ Helper Functions ------------------
def _to_grayscale(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY) if img.shape[2]==4 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def _binarize(img_gray):
    th = threshold_otsu(img_gray)
    bw = (img_gray < th).astype(np.uint8)
    bw = binary_fill_holes(bw).astype(np.uint8)
    return bw

def _largest_contour(binary):
    contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: return None
    return max(contours, key=cv2.contourArea).squeeze()

def _stroke_length(contour):
    if contour is None or len(contour) < 2: return 0.0
    diffs = np.diff(contour.astype(np.float32), axis=0)
    return float(np.hypot(diffs[:,0], diffs[:,1]).sum())

def _curvature_signal(contour):
    if contour is None or len(contour) < 5: return np.array([])
    x, y = contour[:,0], contour[:,1]
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    denom = (dx**2 + dy**2)**1.5
    denom[denom == 0] = np.finfo(float).eps
    return np.abs(dx*ddy - dy*ddx)/denom

def _tremor_metric_from_curvature(curv):
    if curv.size < 8: return 0.0
    c = detrend(curv - curv.mean())
    n = c.size
    C = fft(c)
    psd = np.abs(C)**2
    freqs = fftfreq(n)
    pos = freqs>0
    freqs_pos, psd_pos = freqs[pos], psd[pos]
    if freqs_pos.size==0: return 0.0
    cutoff = np.median(freqs_pos)
    hf_power = psd_pos[freqs_pos>cutoff].sum()
    total_power = psd_pos.sum()
    if total_power <=0: return 0.0
    return float(hf_power/total_power)

# ------------------ Main Analysis ------------------
def analyze_spiral(image_array):
    gray = _to_grayscale(image_array)
    bw = _binarize(gray)
    contour = _largest_contour(bw*255)
    stroke_len = _stroke_length(contour)
    curv = _curvature_signal(contour)
    tremor = _tremor_metric_from_curvature(curv)

    # Heuristic risk scoring
    score = min(100, tremor*100 + (stroke_len/400)*50)
    if score < 30: level = "Low"
    elif score < 60: level = "Moderate"
    else: level = "High"

    return {
        "stroke_length": stroke_len,
        "tremor_metric": tremor,
        "risk_score": score,
        "risk_level": level
    }

# ------------------ Streamlit Canvas ------------------
def show_spiral_canvas():
    st.header("🖌️ Spiral Drawing Test")
    canvas_result = st_canvas(
        stroke_width=3,
        stroke_color="#000000",
        background_color="#ffffff",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="spiral_canvas",
    )

    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data)
        if st.button("Show Result"):
            result = analyze_spiral(np.array(canvas_result.image_data))
            st.success(f"🧠 Parkinson's Risk Score: {result['risk_score']:.2f}")
            st.info(f"Risk Level: {result['risk_level']}")
            st.write("**Stroke Length:**", result["stroke_length"])
            st.write("**Tremor Metric:**", result["tremor_metric"])

import streamlit as st
import numpy as np
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

# Optional canvas import guard
try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except Exception:
    HAS_CANVAS = False

from scipy.signal import detrend
from scipy.fftpack import fft, fftfreq

# ------------------ Utilities ------------------
def pil_from_canvas_array(arr):
    """Convert canvas image_data (HxWx4 RGBA numpy) to PIL Image (RGBA)."""
    arr = (arr.astype('uint8'))
    return Image.fromarray(arr, mode='RGBA')

def to_grayscale_arr(img_pil):
    """Return grayscale numpy array (H, W) from PIL image."""
    gray = img_pil.convert('L')
    return np.array(gray)

def otsu_threshold(gray):
    """Compute Otsu threshold on 8-bit grayscale array."""
    hist, bin_edges = np.histogram(gray.flatten(), bins=256, range=(0,255))
    total = gray.size
    current_max, threshold = 0, 0
    sum_total = np.dot(np.arange(256), hist)
    sumB, wB = 0.0, 0.0
    for i in range(256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        between = wB * wF * (mB - mF) ** 2
        if between > current_max:
            current_max = between
            threshold = i
    return threshold

def largest_connected_component(binary):
    """Return mask of largest connected component (binary: 0/1) using stack-based flood fill."""
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    max_mask = np.zeros_like(binary, dtype=np.uint8)
    max_size = 0
    # 8-neighbors
    neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for r in range(h):
        for c in range(w):
            if binary[r,c] and not visited[r,c]:
                stack = [(r,c)]
                visited[r,c] = True
                comp = []
                while stack:
                    y,x = stack.pop()
                    comp.append((y,x))
                    for dy,dx in neighbors:
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < h and 0 <= nx < w and binary[ny,nx] and not visited[ny,nx]:
                            visited[ny,nx] = True
                            stack.append((ny,nx))
                if len(comp) > max_size:
                    max_size = len(comp)
                    max_mask.fill(0)
                    ys, xs = zip(*comp)
                    max_mask[np.array(ys), np.array(xs)] = 1
    return max_mask

def estimate_stroke_length_from_mask(mask):
    """Approximate stroke length by sorting stroke pixels by angle around centroid and summing Euclidean distances."""
    pts = np.column_stack(np.nonzero(mask))  # (row, col)
    if pts.shape[0] < 2:
        return 0.0, pts
    # centroid
    cy = pts[:,0].mean()
    cx = pts[:,1].mean()
    # compute angles and sort
    angles = np.arctan2(pts[:,0]-cy, pts[:,1]-cx)
    order = np.argsort(angles)
    pts_sorted = pts[order]
    diffs = np.diff(pts_sorted.astype(np.float32), axis=0)
    segs = np.hypot(diffs[:,0], diffs[:,1])
    length = float(segs.sum())
    # If the above underestimates because angles wrap, also try sorting by radius to get approximate polyline
    return length, pts_sorted

def curvature_from_points(pts_sorted):
    """Compute absolute curvature along ordered points (x/y). pts_sorted are (row, col)."""
    if pts_sorted is None or pts_sorted.shape[0] < 5:
        return np.array([])
    x = pts_sorted[:,1].astype(np.float64)  # col -> x
    y = pts_sorted[:,0].astype(np.float64)  # row -> y
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx**2 + dy**2)**1.5
    denom[denom == 0] = np.finfo(float).eps
    kappa = np.abs(dx * ddy - dy * ddx) / denom
    return kappa

def tremor_metric_from_curvature(curv):
    """Return relative HF power of curvature signal (0-1)."""
    if curv.size < 8:
        return 0.0
    c = detrend(curv - curv.mean())
    n = c.size
    C = fft(c)
    psd = np.abs(C)**2
    freqs = fftfreq(n)
    pos = freqs > 0
    freqs_pos = freqs[pos]
    psd_pos = psd[pos]
    if freqs_pos.size == 0:
        return 0.0
    cutoff = np.median(freqs_pos)
    hf = psd_pos[freqs_pos > cutoff].sum()
    total = psd_pos.sum()
    if total <= 0:
        return 0.0
    return float(hf/total)

def count_intersections(mask):
    pad = np.pad(mask, 1, mode='constant')
    coords = np.argwhere(pad==1)
    cnt = 0
    for (r,c) in coords:
        nb = pad[r-1:r+2, c-1:c+2]
        if nb.sum() - 1 > 2:
            cnt += 1
    return int(cnt)

def plot_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return img

# ------------------ Analysis Pipeline ------------------
def analyze_spiral_from_pil(img_pil, debug_plots=True):
    """
    Input: PIL image (RGBA/RGB/Gray)
    Output: dict with features and visuals (PIL images)
    """
    gray = to_grayscale_arr(img_pil)  # HxW uint8
    # Binarize with Otsu
    th = otsu_threshold(gray)
    bw = (gray < th).astype(np.uint8)
    # keep largest stroke
    largest_mask = largest_connected_component(bw)
    # stroke length estimate
    stroke_len, pts_sorted = estimate_stroke_length_from_mask(largest_mask)
    # curvature and tremor
    curv = curvature_from_points(pts_sorted) if pts_sorted is not None and pts_sorted.size>0 else np.array([])
    trem = tremor_metric_from_curvature(curv)
    # edge density
    edge_density = float(largest_mask.sum()) / float(largest_mask.size)
    # intersections
    intersections = count_intersections(largest_mask)
    # curvature stats
    curv_mean = float(curv.mean()) if curv.size else 0.0
    curv_std = float(curv.std()) if curv.size else 0.0

    # Heuristic risk score (0-100)
    # Weights are heuristic; you can retrain later
    w_trem, w_curv, w_inter = 0.6, 0.25, 0.15
    trem_n = min(max(trem, 0.0), 1.0)
    curv_n = min(curv_std / 0.05, 1.0)
    inter_n = min(intersections / 5.0, 1.0)
    score = 100.0 * (w_trem*trem_n + w_curv*curv_n + w_inter*inter_n)
    if score < 30:
        level = "Low"
    elif score < 60:
        level = "Moderate"
    else:
        level = "High"

    features = {
        "stroke_length": stroke_len,
        "edge_density": edge_density,
        "curvature_mean": curv_mean,
        "curvature_std": curv_std,
        "tremor_metric": trem,
        "intersections": intersections,
        "risk_score": score,
        "risk_level": level
    }

    visuals = {}
    if debug_plots:
        # Overview (original, binary, largest mask)
        fig, axs = plt.subplots(1,3, figsize=(9,3))
        axs[0].imshow(img_pil)
        axs[0].set_title("Original")
        axs[0].axis('off')
        axs[1].imshow(gray, cmap='gray')
        axs[1].set_title("Gray")
        axs[1].axis('off')
        axs[2].imshow(largest_mask, cmap='gray')
        axs[2].set_title("Largest Stroke")
        axs[2].axis('off')
        visuals['overview'] = plot_to_image(fig)
        plt.close(fig)

        # radial histogram if we have pts
        if pts_sorted is not None and pts_sorted.shape[0] > 5:
            cy, cx = pts_sorted[:,0].mean(), pts_sorted[:,1].mean()
            d = np.sqrt((pts_sorted[:,0]-cy)**2 + (pts_sorted[:,1]-cx)**2)
            fig2, ax2 = plt.subplots(figsize=(4,3))
            ax2.hist(d, bins=30)
            ax2.set_title("Radial distance histogram")
            visuals['radial_hist'] = plot_to_image(fig2)
            plt.close(fig2)

        # curvature plot
        if curv.size:
            fig3, ax3 = plt.subplots(figsize=(6,2))
            ax3.plot(curv)
            ax3.set_title("Curvature along points")
            ax3.set_xlabel("index")
            ax3.set_ylabel("abs curvature")
            visuals['curvature_plot'] = plot_to_image(fig3)
            plt.close(fig3)

    return {"features": features, "visuals": visuals}

# ------------------ Streamlit UI ------------------
def show_spiral_canvas():
    st.header("🖌️ Spiral Drawing Test (no OpenCV required)")
    if not HAS_CANVAS:
        st.error("Drawable canvas dependency not found. Please add `streamlit-drawable-canvas` to requirements.")
        return

    st.write("Draw a spiral starting from the center. When finished, press **Analyze Drawing**.")

    canvas_result = st_canvas(
        stroke_width=3,
        stroke_color="#000000",
        background_color="#ffffff",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="spiral_canvas_nocv"
    )

    if canvas_result.image_data is not None:
        img_pil = pil_from_canvas_array(canvas_result.image_data)
        st.image(img_pil, caption="Your drawing", width=300)

        if st.button("Analyze Drawing"):
            with st.spinner("Analyzing…"):
                out = analyze_spiral_from_pil(img_pil, debug_plots=True)
                f = out["features"]
                v = out["visuals"]

            st.success(f"Risk score: {f['risk_score']:.1f} / 100")
            st.write(f"**Risk level:** {f['risk_level']}")
            st.write("**Features:**")
            st.json(f)

            st.subheader("Diagnostics")
            if 'overview' in v:
                st.image(v['overview'], caption="Original / Gray / Largest stroke", use_column_width=False)
            if 'radial_hist' in v:
                st.image(v['radial_hist'], caption="Radial histogram")
            if 'curvature_plot' in v:
                st.image(v['curvature_plot'], caption="Curvature plot")

            st.warning("This is a screening heuristic — not a medical diagnosis. Consult a neurologist for confirmation.")

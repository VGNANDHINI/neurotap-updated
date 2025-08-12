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
# spiral_analysis.py
import numpy as np
import cv2
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
from skimage import img_as_bool
from scipy.signal import detrend
from scipy.fftpack import fft, fftfreq
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt
import io
from PIL import Image

def _to_grayscale(img):
    if img.ndim == 3:
        # RGBA or RGB -> convert to gray using luminosity
        return cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY) if img.shape[2] == 4 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def _binarize(img_gray):
    # adaptive thresholding via Otsu
    th = threshold_otsu(img_gray)
    bw = (img_gray < th).astype(np.uint8)  # strokes usually darker -> True where stroke present
    # fill small holes
    bw = binary_fill_holes(bw).astype(np.uint8)
    return bw

def _largest_contour(binary):
    contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    # choose largest by area
    c = max(contours, key=cv2.contourArea)
    return c.squeeze()  # Nx2 array of points

def _stroke_length(contour):
    if contour is None or len(contour) < 2:
        return 0.0
    # compute polyline length
    diffs = np.diff(contour.astype(np.float32), axis=0)
    segs = np.hypot(diffs[:,0], diffs[:,1])
    return float(segs.sum())

def _radius_stats(contour, center):
    if contour is None or len(contour) == 0:
        return (0.0, 0.0)
    pts = contour.astype(np.float32)
    d = np.sqrt((pts[:,0]-center[0])**2 + (pts[:,1]-center[1])**2)
    return float(d.mean()), float(d.std())

def _skeleton_and_branch_points(binary):
    # convert to boolean image for skeletonize
    b = img_as_bool(binary)
    skel = skeletonize(b).astype(np.uint8)
    # find branch points: pixel with >2 neighbors
    pad = np.pad(skel, 1, mode='constant')
    branch_pts = []
    coords = np.argwhere(pad==1)
    for (r,c) in coords:
        nb = pad[r-1:r+2, c-1:c+2]
        count = nb.sum() - 1  # exclude center
        if count > 2:
            branch_pts.append((r-1,c-1))  # unpad coords
    return skel, branch_pts

def _curvature_signal(contour):
    # returns curvature array along contour (absolute curvature)
    if contour is None or len(contour) < 5:
        return np.array([])
    pts = contour.astype(np.float64)
    # parameterize by t
    x = pts[:,0]
    y = pts[:,1]
    # derivatives using central differences
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx**2 + dy**2)**1.5
    # avoid division by zero
    denom[denom == 0] = np.finfo(float).eps
    kappa = np.abs(dx * ddy - dy * ddx) / denom
    return kappa

def _tremor_metric_from_curvature(curv):
    # tremor metric: relative power in high-frequency band of curvature
    if curv.size < 8:
        return 0.0
    # detrend and window
    c = detrend(curv - curv.mean())
    n = c.size
    # FFT
    C = fft(c)
    psd = np.abs(C)**2
    freqs = fftfreq(n)
    # consider positive freqs only
    pos = freqs > 0
    freqs_pos = freqs[pos]
    psd_pos = psd[pos]
    if freqs_pos.size == 0:
        return 0.0
    # pick "high frequency" as > median frequency
    cutoff = np.median(freqs_pos)
    hf_power = psd_pos[freqs_pos > cutoff].sum()
    total_power = psd_pos.sum()
    if total_power <= 0:
        return 0.0
    return float(hf_power / total_power)

def _edge_density(binary):
    # amount of stroke pixels relative to image area
    return float(binary.sum()) / float(binary.size)

def _count_intersections(skel):
    # count pixels with >2 neighbors
    pad = np.pad(skel, 1, mode='constant')
    coords = np.argwhere(pad==1)
    count = 0
    for (r,c) in coords:
        nb = pad[r-1:r+2, c-1:c+2]
        if nb.sum() - 1 > 2:
            count += 1
    return int(count)

def _plot_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return np.array(img)

def analyze_spiral(image_array, debug_plots=True):
    """
    image_array: numpy array from streamlit canvas (H x W x 4 RGBA) or RGB
    Returns: dict with features, risk_score (0-100), risk_level, and optional visualization images
    """
    out = {}
    # 1. grayscale
    gray = _to_grayscale(image_array)
    # 2. binarize
    bw = _binarize(gray)  # 0/1
    bw_uint8 = (bw*255).astype(np.uint8)
    # 3. find largest contour (stroke)
    contour = _largest_contour(bw_uint8)
    H,W = gray.shape
    center = (W/2.0, H/2.0)
    # 4. stroke length
    stroke_len = _stroke_length(contour)
    out['stroke_length'] = stroke_len
    # 5. center & radius stats
    mean_r, std_r = _radius_stats(contour, center) if contour is not None else (0.0,0.0)
    out['mean_radius'] = mean_r
    out['radius_std'] = std_r
    # 6. skeleton and branch points
    skel, branch_pts = _skeleton_and_branch_points(bw)
    out['branch_points'] = len(branch_pts)
    out['skeleton_pixels'] = int(skel.sum())
    # 7. curvature and tremor metric
    curv = _curvature_signal(contour)
    out['curvature_mean'] = float(curv.mean()) if curv.size else 0.0
    out['curvature_std'] = float(curv.std()) if curv.size else 0.0
    tremor = _tremor_metric_from_curvature(curv)
    out['tremor_metric'] = tremor
    # 8. edge density & intersections
    out['edge_density'] = _edge_density(bw)
    out['intersections'] = _count_intersections(skel)
    # 9. simple heuristic risk score (0-100)
    # Weights chosen heuristically; you'll calibrate with labeled data later.
    # higher tremor, higher radius_std, more intersections, higher curvature_std => higher risk
    w = {
        'tremor': 0.45,
        'radius_std': 0.20,
        'curv_std': 0.20,
        'intersections': 0.15
    }
    # normalize components to [0,1] using safe anchors
    tremor_n = min(max(tremor,0.0),1.0)
    radius_n = min(std_r / (max(H,W)/8.0 + 1e-6), 1.0)  # anchor: if std ~ size/8 -> 1.0
    curv_n = min(max(out['curvature_std'] / 0.05, 0.0), 1.0)  # anchor
    inter_n = min(out['intersections'] / 5.0, 1.0)  # >5 intersections -> 1.0
    score = 100.0 * (w['tremor']*tremor_n + w['radius_std']*radius_n + w['curv_std']*curv_n + w['intersections']*inter_n)
    out['risk_score'] = float(score)
    # risk level
    if score < 30:
        level = "Low"
    elif score < 60:
        level = "Moderate"
    else:
        level = "High"
    out['risk_level'] = level

    visuals = {}
    if debug_plots:
        # visual 1: original, binarized, edges, skeleton
        fig, axes = plt.subplots(1,4, figsize=(12,3))
        axes[0].imshow(image_array if image_array.ndim==3 else cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB))
        axes[0].set_title("Original")
        axes[0].axis('off')
        axes[1].imshow(bw, cmap='gray')
        axes[1].set_title("Binarized")
        axes[1].axis('off')
        edges = cv2.Canny(bw_uint8, 50,150)
        axes[2].imshow(edges, cmap='gray')
        axes[2].set_title("Edges")
        axes[2].axis('off')
        axes[3].imshow(skel, cmap='gray')
        axes[3].set_title("Skeleton")
        axes[3].axis('off')
        visuals['overview'] = _plot_to_image(fig)
        plt.close(fig)

        # visual 2: radial distance histogram
        if contour is not None:
            pts = contour.astype(np.float32)
            d = np.sqrt((pts[:,0]-center[0])**2 + (pts[:,1]-center[1])**2)
            fig2, ax2 = plt.subplots(figsize=(4,3))
            ax2.hist(d, bins=30)
            ax2.set_title("Radial distance histogram")
            visuals['radial_hist'] = _plot_to_image(fig2)
            plt.close(fig2)
        # visual 3: curvature plot
        if curv.size:
            fig3, ax3 = plt.subplots(figsize=(4,2))
            ax3.plot(curv, linewidth=1)
            ax3.set_title("Curvature along contour")
            ax3.set_xlabel("point index")
            ax3.set_ylabel("abs curvature")
            visuals['curvature_plot'] = _plot_to_image(fig3)
            plt.close(fig3)

    return {
        "features": out,
        "visuals": visuals
    }

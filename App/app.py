"""
🚗 Egyptian Car Plate Detection System
Streamlit app for detecting license plates in vehicle images.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import sys
import random
import glob

# ── path setup ───────────────────────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
sys.path.insert(0, APP_DIR)

from utils import PlateDetector, compute_image_stats

# ── paths ────────────────────────────────────────────────────────────────────
IMAGES_DIR = os.path.join(ROOT_DIR, "Data", "EALPR Vechicles dataset", "Vehicles")
LABELS_DIR = os.path.join(ROOT_DIR, "Data", "EALPR Vechicles dataset", "Vehicles Labeling")

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Plate Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-banner {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    text-align: center;
    color: white;
    box-shadow: 0 8px 32px rgba(48, 43, 99, 0.35);
}
.hero-banner h1 { font-size: 2.2rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
.hero-banner p  { font-size: 1rem; opacity: 0.8; margin-top: 0.4rem; }

.det-card {
    background: linear-gradient(135deg, #667eea20, #764ba220);
    border: 1px solid #764ba240;
    padding: 1.2rem 1.5rem;
    border-radius: 12px;
    text-align: center;
    margin: 0.5rem 0;
}
.det-card h2 { color: #764ba2; margin: 0 0 0.3rem 0; font-size: 1.6rem; }
.det-card p  { margin: 0; color: #555; }

.stat-pill {
    display: inline-block;
    background: #667eea15;
    border: 1px solid #667eea30;
    border-radius: 8px;
    padding: 0.4rem 1rem;
    margin: 0.2rem;
    font-size: 0.85rem;
}

.footer-text {
    text-align: center;
    color: #999;
    padding: 2rem 0 1rem;
    font-size: 0.82rem;
}

/* Button styling */
div.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.5rem;
    font-weight: 600;
    transition: transform 0.2s, box-shadow 0.2s;
}
div.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}
</style>
""", unsafe_allow_html=True)

# ── session state ────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── detector ─────────────────────────────────────────────────────────────────
labels_dir = LABELS_DIR if os.path.isdir(LABELS_DIR) else None
detector = PlateDetector(labels_dir=labels_dir)

# ── helper: collect sample images from dataset ──────────────────────────────
@st.cache_data
def get_sample_images(n=12):
    if not os.path.isdir(IMAGES_DIR):
        return []
    all_imgs = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
    all_imgs += sorted(glob.glob(os.path.join(IMAGES_DIR, "*.png")))
    if len(all_imgs) == 0:
        return []
    rng = random.Random(42)
    return rng.sample(all_imgs, min(n, len(all_imgs)))


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    mode = st.radio(
        "Input Mode",
        ["📤 Upload Image", "🎲 Random from Dataset"],
        help="Upload your own image or pick a random one from the dataset.",
    )

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info(
        "**Egyptian Car Plate Detection**\n\n"
        "• OpenCV-based plate detection\n"
        "• Supports YOLO ground-truth labels\n"
        "• Contour + edge dual-pipeline\n"
        "• Works with any vehicle image"
    )

    dataset_ok = os.path.isdir(IMAGES_DIR)
    if dataset_ok:
        n_imgs = len(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
        st.success(f"📂 Dataset loaded — **{n_imgs}** images")
    else:
        st.warning("Dataset folder not found. Upload images instead.")

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="hero-banner">'
    "<h1>🚗 Egyptian Car Plate Detection</h1>"
    "<p>Upload a vehicle image or pick one from the dataset — plates are detected instantly</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Image source ─────────────────────────────────────────────────────────────
image_np = None
image_name = None

if mode == "📤 Upload Image":
    uploaded = st.file_uploader(
        "Drop an image here",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a clear photo of a vehicle with a visible license plate.",
    )
    if uploaded is not None:
        pil = Image.open(uploaded).convert("RGB")
        image_np = np.array(pil)
        image_name = uploaded.name

else:  # Random from Dataset
    samples = get_sample_images(n=20)
    if samples:
        col_btn = st.columns([1, 1, 3])
        with col_btn[0]:
            if st.button("🎲 Pick Random", use_container_width=True):
                st.session_state["rand_idx"] = random.randint(0, len(samples) - 1)
        idx = st.session_state.get("rand_idx", 0)
        chosen = samples[idx % len(samples)]
        pil = Image.open(chosen).convert("RGB")
        image_np = np.array(pil)
        image_name = os.path.basename(chosen)
    else:
        st.warning("No dataset images found. Use **Upload Image** mode instead.")

# ── Detection pipeline ───────────────────────────────────────────────────────
if image_np is not None:
    st.markdown("---")

    col_orig, col_det = st.columns(2, gap="large")

    # Run detection
    detections = detector.detect(image_np, image_name=image_name)
    annotated = detector.annotate_image(image_np, detections)
    stats = compute_image_stats(image_np)

    with col_orig:
        st.markdown("#### 📷 Original Image")
        st.image(image_np, caption=image_name or "Uploaded Image", use_container_width=True)

    with col_det:
        st.markdown("#### 🔍 Detection Results")
        st.image(annotated, caption=f"{len(detections)} plate(s) detected", use_container_width=True)

    # ── Result summary ───────────────────────────────────────────────────────
    n_det = len(detections)
    source_tag = detections[0]["source"] if n_det > 0 else "—"
    best_conf = max((d["confidence"] for d in detections), default=0)

    st.markdown(
        f'<div class="det-card">'
        f'<h2>{"✅" if n_det > 0 else "⚠️"} {n_det} Plate{"s" if n_det != 1 else ""} Detected</h2>'
        f'<p>Best confidence: <strong>{best_conf:.0%}</strong> &nbsp;|&nbsp; '
        f'Source: <strong>{source_tag.replace("_", " ").title()}</strong></p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Tabs: Crops / Stats / Details ────────────────────────────────────────
    tab_crops, tab_stats, tab_details = st.tabs(
        ["🔎 Plate Crops", "📊 Image Statistics", "📋 Detection Details"]
    )

    with tab_crops:
        crops = detector.extract_plate_crops(image_np, detections)
        if crops:
            crop_cols = st.columns(min(len(crops), 4))
            for i, crop in enumerate(crops):
                with crop_cols[i % len(crop_cols)]:
                    st.image(crop, caption=f"Plate {i+1}", use_container_width=True)
        else:
            st.info("No plate regions to crop.")

    with tab_stats:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Resolution", f"{stats['width']}×{stats['height']}")
        m2.metric("Brightness", f"{stats['mean_brightness']:.0f}/255")
        m3.metric("Contrast (σ)", f"{stats['contrast']:.1f}")
        m4.metric("Sharpness", f"{stats['sharpness']:.0f}")

        # Edge density bar
        st.markdown(f"**Edge Density:** {stats['edge_density']}%")
        st.progress(min(stats["edge_density"] / 30.0, 1.0))

        # Brightness histogram
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        hist_vals = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
        fig_hist = go.Figure(
            go.Bar(
                x=list(range(64)),
                y=hist_vals,
                marker_color="#667eea",
            )
        )
        fig_hist.update_layout(
            title="Brightness Histogram",
            xaxis_title="Bin",
            yaxis_title="Pixel Count",
            height=280,
            margin=dict(l=40, r=20, t=40, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab_details:
        if detections:
            df = pd.DataFrame(detections)
            df["area"] = df["w"] * df["h"]
            df["aspect_ratio"] = (df["w"] / df["h"]).round(2)
            df = df.rename(columns={
                "x": "X", "y": "Y", "w": "Width", "h": "Height",
                "confidence": "Confidence", "source": "Method",
                "area": "Area (px²)", "aspect_ratio": "Aspect Ratio",
            })
            display_cols = ["X", "Y", "Width", "Height", "Confidence", "Method", "Area (px²)", "Aspect Ratio"]
            st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

            # Confidence chart
            fig_conf = px.bar(
                df, x=df.index.astype(str), y="Confidence",
                color="Confidence",
                color_continuous_scale=["#764ba2", "#667eea", "#43e97b"],
                labels={"x": "Detection #"},
                title="Detection Confidence Scores",
            )
            fig_conf.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_conf, use_container_width=True)
        else:
            st.info("No detections to display.")

    # ── Save to history ──────────────────────────────────────────────────────
    st.session_state.history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "image": image_name or "upload",
        "plates": n_det,
        "best_conf": round(best_conf, 2),
        "source": source_tag,
    })

# ── History section ──────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    with st.expander("📈 Detection History", expanded=False):
        hdf = pd.DataFrame(st.session_state.history)
        st.dataframe(hdf, use_container_width=True, hide_index=True)

        csv = hdf.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            data=csv,
            file_name="detection_history.csv",
            mime="text/csv",
        )

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer-text">'
    "🚗 Egyptian Car Plate Detection System &nbsp;·&nbsp; "
    "Powered by OpenCV &amp; Streamlit &nbsp;·&nbsp; "
    "© 2024"
    "</div>",
    unsafe_allow_html=True,
)
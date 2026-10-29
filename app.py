"""Digital Image Processing & Enhancement Studio.
Run with:  streamlit run app.py
"""
import io
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st

import enhance as E

# ---------------------------------------------------------------- page config & styling
st.set_page_config(
    page_title="Image Enhancement Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Base typography and layout */
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 3.5rem; max-width: 1220px; }

/* Main title */
.studio-title {
    font-size: 2.25rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #0F172A;
    text-align: center;
    margin-top: 0.5rem;
    margin-bottom: 2.2rem;
}

@media (max-width: 768px) {
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
    }
    .studio-title {
        font-size: 1.65rem !important;
        margin-bottom: 1.2rem !important;
    }
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    border-right: 1px solid #E2E8F0;
    background-color: #F8FAFC;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* Slide Header & Metadata */
.slide-header {
    border-bottom: 1px solid #F1F5F9;
    padding-bottom: 0.9rem;
    margin-bottom: 1.2rem;
}
.slide-title {
    font-size: 1.28rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 0.2rem;
}
.slide-description {
    font-size: 0.92rem;
    color: #475569;
}

/* Image Column Frames */
.frame-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.6rem;
}
.frame-badge {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
}
.badge-orig {
    background: #F1F5F9;
    color: #475569;
    border: 1px solid #E2E8F0;
}
.badge-enh {
    background: #EEF2FF;
    color: #2457F5;
    border: 1px solid #DBEAFE;
}
.meta-tag {
    font-size: 0.75rem;
    color: #64748B;
    font-weight: 500;
}
.metric-badge {
    background: #ECFDF5;
    color: #059669;
    border: 1px solid #D1FAE5;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.2rem 0.55rem;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers & asset loader
def _load_asset(filename, is_gray=False):
    asset_path = os.path.join(os.path.dirname(__file__), "assets", filename)
    if os.path.exists(asset_path):
        flag = cv2.IMREAD_GRAYSCALE if is_gray else cv2.IMREAD_COLOR
        im = cv2.imread(asset_path, flag)
        if im is not None:
            return im if is_gray else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return None


def show_image(container, image, caption=None):
    try:
        container.image(image, caption=caption, width="stretch")
    except TypeError:
        container.image(image, caption=caption, use_container_width=True)


def plot_hist(image, title):
    fig, ax = plt.subplots(figsize=(5, 2.3), dpi=90)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FAFAFA")
    if image.ndim == 2:
        ax.hist(image.ravel(), bins=256, range=(0, 255), color="#64748B", alpha=0.85)
    else:
        for i, col in enumerate(["#EF4444", "#10B981", "#3B82F6"]):
            ax.hist(image[:, :, i].ravel(), bins=256, range=(0, 255), color=col, alpha=0.55)
    ax.set_title(title, fontsize=10.5, fontweight="600", color="#1E293B", pad=8)
    ax.set_xlabel("Pixel intensity (0-255)", fontsize=8.5, color="#64748B")
    ax.set_ylabel("Count", fontsize=8.5, color="#64748B")
    ax.tick_params(colors="#64748B", labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#E2E8F0")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- sample photos definition (2 key examples)
@st.cache_data
def make_samples():
    def u8(x):
        return np.clip(x, 0, 255).astype(np.uint8)

    def rgb(a):
        return np.dstack([a, a, a]) if a.ndim == 2 else a

    coffee_raw = _load_asset("coffee.png")
    moon_raw = _load_asset("moon.png", is_gray=True)

    if coffee_raw is None or moon_raw is None:
        try:
            from skimage import data
            coffee_raw = data.coffee() if coffee_raw is None else coffee_raw
            moon_raw = data.moon() if moon_raw is None else moon_raw
        except Exception:
            h, w = 400, 400
            coffee_raw = np.full((h, w, 3), 40, dtype=np.uint8) if coffee_raw is None else coffee_raw
            moon_raw = np.full((h, w), 100, dtype=np.uint8) if moon_raw is None else moon_raw

    # 1. Low light coffee sample
    dark = u8(255 * (coffee_raw / 255.0) ** 2.6 * 0.85)
    dark_enh = E.gamma_correction(dark, 0.65)
    _, dark_buf = cv2.imencode(".png", cv2.cvtColor(dark_enh, cv2.COLOR_RGB2BGR))

    # 2. Flat hazy moon sample
    moon_f = moon_raw.astype(np.float32)
    flat = rgb(u8(110 + (moon_f - moon_f.mean()) * 0.35))
    flat_enh = E.histogram_equalization(flat)
    _, flat_buf = cv2.imencode(".png", cv2.cvtColor(flat_enh, cv2.COLOR_RGB2BGR))

    return [
        {
            "id": "dark",
            "title": "Low-Light Restoration",
            "description": "",
            "original": dark,
            "enhanced": dark_enh,
            "effect_name": "Lighten dark areas",
            "psnr": f"{E.psnr(dark, dark_enh):.2f} dB",
            "ssim": f"{E.ssim(dark, dark_enh):.3f}",
            "png_bytes": dark_buf.tobytes(),
        },
        {
            "id": "flat",
            "title": "Dynamic Range & Contrast Stretch",
            "description": "",
            "original": flat,
            "enhanced": flat_enh,
            "effect_name": "Auto contrast",
            "psnr": f"{E.psnr(flat, flat_enh):.2f} dB",
            "ssim": f"{E.ssim(flat, flat_enh):.3f}",
            "png_bytes": flat_buf.tobytes(),
        },
    ]


# ---------------------------------------------------------------- effect groups dictionary
GROUPS = {
    "Fix lighting": [
        "Brightness and contrast",
        "Lighten dark areas",
        "Reveal hidden detail in shadows",
        "Auto contrast",
        "Smart contrast",
    ],
    "Reduce noise and grain": [
        "Smooth gently",
        "Soft blur",
        "Remove specks and dots",
    ],
    "Sharpen and find edges": [
        "Quick sharpen",
        "Fine sharpen",
        "Soft outline",
        "Line drawing outline",
    ],
    "Fine detail control": [
        "Soften fine detail",
        "Keep only fine detail",
    ],
}


# ---------------------------------------------------------------- sidebar setup
sb = st.sidebar
sb.markdown("### Photo Source")
photo_mode = sb.radio(
    "Choose photo source",
    ["Sample: Coffee (Low Light)", "Sample: Moon (Contrast)", "Upload your own photo"],
    label_visibility="collapsed",
)

uploaded = None
if photo_mode == "Upload your own photo":
    uploaded = sb.file_uploader(
        "Upload custom image",
        type=["png", "jpg", "jpeg", "bmp"],
        help="Upload your own photo (PNG, JPG, BMP)",
    )

bw_mode = sb.checkbox("Convert to Black & White (Grayscale)", value=False)

# Resolve active image
samples_data = make_samples()
if uploaded is not None:
    pil_img = Image.open(uploaded).convert("RGB")
    if max(pil_img.size) > 1080:
        pil_img.thumbnail((1080, 1080))
    active_img = np.array(pil_img)
elif photo_mode == "Sample: Moon (Contrast)":
    active_img = samples_data[1]["original"].copy()
else:
    active_img = samples_data[0]["original"].copy()

if bw_mode:
    active_img = cv2.cvtColor(active_img, cv2.COLOR_RGB2GRAY) if active_img.ndim == 3 else active_img

sb.divider()

# ---------------------------------------------------------------- control panel
sb.markdown("### Adjustment Controls")
group = sb.radio("Category", list(GROUPS), key="adjust_group")
choice = sb.selectbox("Effect", GROUPS[group], key="adjust_effect")

sb.markdown("##### Parameters")

h, w = active_img.shape[:2]

if choice == "Brightness and contrast":
    contrast = sb.slider("Contrast", -50, 100, 20)
    brightness = sb.slider("Brightness", -100, 100, 20)
    out_img = E.brightness_contrast(active_img, 1 + contrast / 100, brightness)
elif choice == "Lighten dark areas":
    amount = sb.slider("Amount", 0, 100, 50)
    out_img = E.gamma_correction(active_img, 1 - 0.7 * amount / 100)
elif choice == "Reveal hidden detail in shadows":
    sb.caption("Logarithmic mapping expands darker values.")
    out_img = E.log_transform(active_img)
elif choice == "Auto contrast":
    sb.caption("Histogram equalization redistributes luminance evenly.")
    out_img = E.histogram_equalization(active_img)
elif choice == "Smart contrast":
    strength = sb.slider("Strength (clip limit)", 1.0, 10.0, 2.0, step=0.5)
    area = sb.slider("Area size (tile)", 2, 16, 8, help="Smaller values adjust smaller local patches.")
    out_img = E.clahe(active_img, strength, area)
elif choice == "Smooth gently":
    n = sb.slider("Neighborhood size", 1, 7, 1)
    out_img = E.mean_filter(active_img, 2 * n + 1)
elif choice == "Soft blur":
    n = sb.slider("Blur amount", 1, 10, 3)
    sigma = n / 2
    out_img = E.gaussian_filter(active_img, 2 * int(3 * sigma) + 1, sigma)
elif choice == "Remove specks and dots":
    n = sb.slider("Filter size", 1, 7, 1)
    out_img = E.median_filter(active_img, 2 * n + 1)
elif choice == "Quick sharpen":
    sb.caption("Laplacian 3x3 high-pass edge enhancement.")
    out_img = E.laplacian_sharpen(active_img)
elif choice == "Fine sharpen":
    strength = sb.slider("Strength", 0.0, 5.0, 1.2, step=0.1)
    size = sb.slider("Detail size", 0.5, 5.0, 1.5, step=0.25, help="Small values sharpen fine texture, larger values sharpen shapes.")
    out_img = E.unsharp_mask(active_img, size, strength)
elif choice == "Soft outline":
    sb.caption("Sobel gradient magnitude map.")
    out_img = E.sobel_edges(active_img)
elif choice == "Line drawing outline":
    sensitivity = sb.slider("Sensitivity", 0, 100, 50, help="Higher values detect more delicate outlines.")
    high = int(250 - 2 * sensitivity)
    out_img = E.canny_edges(active_img, high // 2, high)
else:  # Fine detail control
    amount = sb.slider("Cutoff frequency", 1, 99, 50)
    with sb.expander("Filter options"):
        shape = st.selectbox("Filter shape", ["gaussian", "butterworth", "ideal"])
    half = min(h, w) / 2
    if choice == "Soften fine detail":
        cutoff = max(5, int(half * (1 - amount / 100) ** 2))
        out_img = E.frequency_filter(active_img, "lowpass", shape, cutoff)
    else:
        cutoff = max(5, int(half * (amount / 100) ** 2))
        out_img = E.frequency_filter(active_img, "highpass", shape, cutoff)

sb.divider()
sb.markdown("### View Options")
show_histograms = sb.checkbox("Show Tone Histograms", value=False)
show_metrics = sb.checkbox("Show Quality Metrics (PSNR & SSIM)", value=True)


# ---------------------------------------------------------------- main interface
st.markdown('<div class="studio-title">Image Enhancement Studio</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- Section 1: Live Interactive Studio
has_same_shape = (active_img.shape == out_img.shape)
if show_metrics and has_same_shape:
    live_psnr = f"{E.psnr(active_img, out_img):.2f} dB"
    live_ssim = f"{E.ssim(active_img, out_img):.3f}"
    live_metric_tag = f"PSNR: {live_psnr} • SSIM: {live_ssim}"
else:
    live_metric_tag = choice

with st.container(border=True):
    st.markdown(
        f"""<div class="slide-header">
<div class="slide-title">{choice}</div>
</div>""",
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown(
            f"""<div class="frame-header">
<span class="frame-badge badge-orig">Original</span>
<span class="meta-tag">{w} × {h} px • {active_img.ndim}D</span>
</div>""",
            unsafe_allow_html=True,
        )
        show_image(col_l, active_img)

    with col_r:
        st.markdown(
            f"""<div class="frame-header">
<span class="frame-badge badge-enh">Enhanced</span>
<span class="metric-badge">{live_metric_tag}</span>
</div>""",
            unsafe_allow_html=True,
        )
        show_image(col_r, out_img)

    # Optional Histograms
    if show_histograms:
        st.markdown("---")
        h_c1, h_c2 = st.columns(2, gap="large")
        with h_c1:
            fig_orig = plot_hist(active_img, "Original Tone Histogram")
            st.pyplot(fig_orig)
            plt.close(fig_orig)
        with h_c2:
            fig_enh = plot_hist(out_img, "Enhanced Tone Histogram")
            st.pyplot(fig_enh)
            plt.close(fig_enh)

    # 1-click Download button
    save_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR) if out_img.ndim == 3 else out_img
    _, enc_buf = cv2.imencode(".png", save_img)
    btn_cols = st.columns([4, 1])
    with btn_cols[1]:
        st.download_button(
            label="Download Result",
            data=enc_buf.tobytes(),
            file_name="enhanced_result.png",
            mime="image/png",
            key="dl_btn_live_studio",
        )


# ---------------------------------------------------------------- Section 2: Automated Showcase (2 examples)
st.markdown("---")
st.markdown("#### Showcase Examples")

showcase_slides = []
for item in samples_data:
    orig = item["original"]
    enh = item["enhanced"]
    if bw_mode:
        orig = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY) if orig.ndim == 3 else orig
        enh = cv2.cvtColor(enh, cv2.COLOR_RGB2GRAY) if enh.ndim == 3 else enh
    slide_copy = dict(item)
    slide_copy["original"] = orig
    slide_copy["enhanced"] = enh
    showcase_slides.append(slide_copy)

for idx, slide in enumerate(showcase_slides, start=1):
    s_orig = slide["original"]
    s_enh = slide["enhanced"]
    s_h, s_w = s_orig.shape[:2]
    s_same_shape = (s_orig.shape == s_enh.shape)

    if show_metrics and s_same_shape:
        if "psnr" in slide and "ssim" in slide and not bw_mode:
            s_metric_tag = f"PSNR: {slide['psnr']} • SSIM: {slide['ssim']}"
        else:
            s_metric_tag = f"PSNR: {E.psnr(s_orig, s_enh):.2f} dB • SSIM: {E.ssim(s_orig, s_enh):.3f}"
    else:
        s_metric_tag = slide.get("effect_name", "Enhanced")

    with st.container(border=True):
        st.markdown(
            f"""<div class="slide-header">
<div class="slide-title">{slide['title']}</div>
</div>""",
            unsafe_allow_html=True,
        )

        sc_left, sc_right = st.columns(2, gap="large")

        with sc_left:
            st.markdown(
                f"""<div class="frame-header">
<span class="frame-badge badge-orig">Original</span>
<span class="meta-tag">{s_w} × {s_h} px • {s_orig.ndim}D</span>
</div>""",
                unsafe_allow_html=True,
            )
            show_image(sc_left, s_orig)

        with sc_right:
            st.markdown(
                f"""<div class="frame-header">
<span class="frame-badge badge-enh">Enhanced</span>
<span class="metric-badge">{s_metric_tag}</span>
</div>""",
                unsafe_allow_html=True,
            )
            show_image(sc_right, s_enh)

        if "png_bytes" in slide and not bw_mode:
            download_data = slide["png_bytes"]
        else:
            s_save = cv2.cvtColor(s_enh, cv2.COLOR_RGB2BGR) if s_enh.ndim == 3 else s_enh
            _, s_buf = cv2.imencode(".png", s_save)
            download_data = s_buf.tobytes()

        s_btn_cols = st.columns([4, 1])
        with s_btn_cols[1]:
            st.download_button(
                label="Download Result",
                data=download_data,
                file_name=f"enhanced_{slide['id']}.png",
                mime="image/png",
                key=f"dl_btn_showcase_{slide['id']}_{idx}",
            )

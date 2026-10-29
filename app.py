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

/* Slideshow Card Container */
.slide-section {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 18px;
    padding: 1.5rem;
    margin-bottom: 2.5rem;
    box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.05), 0 2px 6px -1px rgba(15, 23, 42, 0.02);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.slide-section:hover {
    box-shadow: 0 10px 24px -4px rgba(15, 23, 42, 0.09);
}

/* Slide Header & Metadata */
.slide-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    border-bottom: 1px solid #F1F5F9;
    padding-bottom: 0.9rem;
    margin-bottom: 1.2rem;
}
.slide-number {
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #2457F5;
    background: #EEF2FF;
    border: 1px solid #DBEAFE;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    display: inline-block;
    margin-bottom: 0.4rem;
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
.slide-algorithm {
    font-size: 0.78rem;
    color: #64748B;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    padding: 0.3rem 0.65rem;
    border-radius: 6px;
    white-space: nowrap;
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

/* Download Bar */
.slide-footer {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    margin-top: 1rem;
    padding-top: 0.75rem;
    border-top: 1px solid #F1F5F9;
}
</style>
""", unsafe_allow_html=True)


def _load_asset(filename, is_gray=False):
    asset_path = os.path.join(os.path.dirname(__file__), "assets", filename)
    if os.path.exists(asset_path):
        flag = cv2.IMREAD_GRAYSCALE if is_gray else cv2.IMREAD_COLOR
        im = cv2.imread(asset_path, flag)
        if im is not None:
            return im if is_gray else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return None


# ---------------------------------------------------------------- sample photos definition
@st.cache_data
def make_samples():
    def u8(x):
        return np.clip(x, 0, 255).astype(np.uint8)

    def rgb(a):
        return np.dstack([a, a, a]) if a.ndim == 2 else a

    # Try loading from local assets folder first
    coffee_raw = _load_asset("coffee.png")
    moon_raw = _load_asset("moon.png", is_gray=True)

    # Fallback to skimage.data if local assets are missing
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


# ---------------------------------------------------------------- helpers
def show_image(container, image, caption=None):
    try:
        container.image(image, caption=caption, width="stretch")
    except TypeError:
        container.image(image, caption=caption, use_container_width=True)


def plot_hist(image, title):
    fig, ax = plt.subplots(figsize=(5, 2.3))
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


# ---------------------------------------------------------------- sidebar setup
sb = st.sidebar
sb.markdown("### Photo Source")
uploaded = sb.file_uploader(
    "Upload custom image",
    type=["png", "jpg", "jpeg", "bmp"],
    label_visibility="collapsed",
    help="Upload your own photo (PNG, JPG, BMP)",
)

bw_mode = sb.checkbox("Convert to Black & White (Grayscale)", value=False)

sb.divider()
sb.markdown("### View Options")
show_histograms = sb.checkbox("Show Tone Histograms", value=False)
show_metrics = sb.checkbox("Show Quality Metrics (PSNR & SSIM)", value=True)

# ---------------------------------------------------------------- main interface
st.markdown('<div class="studio-title">Image Enhancement Studio</div>', unsafe_allow_html=True)

# Determine items to display
if uploaded is not None:
    # Downscale large mobile uploads to a max side of 1080px for fast responsive processing
    pil_img = Image.open(uploaded).convert("RGB")
    if max(pil_img.size) > 1080:
        pil_img.thumbnail((1080, 1080))
    custom_img = np.array(pil_img)
    if bw_mode:
        custom_img = cv2.cvtColor(custom_img, cv2.COLOR_RGB2GRAY)

    c_light = E.gamma_correction(custom_img, 0.65)
    c_contrast = E.histogram_equalization(custom_img)

    slides = [
        {
            "id": "custom_light",
            "title": "Low-Light Shadow Recovery",
            "description": "",
            "original": custom_img,
            "enhanced": c_light,
            "effect_name": "Lighten dark areas",
        },
        {
            "id": "custom_contrast",
            "title": "Contrast & Tone Redistribution",
            "description": "",
            "original": custom_img,
            "enhanced": c_contrast,
            "effect_name": "Auto contrast",
        },
    ]
else:
    # Built-in samples slideshow (2 primary examples)
    raw_slides = make_samples()
    slides = []
    for item in raw_slides:
        orig = item["original"]
        enh = item["enhanced"]
        if bw_mode:
            orig = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY) if orig.ndim == 3 else orig
            enh = cv2.cvtColor(enh, cv2.COLOR_RGB2GRAY) if enh.ndim == 3 else enh
        slide_copy = dict(item)
        slide_copy["original"] = orig
        slide_copy["enhanced"] = enh
        slides.append(slide_copy)


# ---------------------------------------------------------------- render slideshow
total_slides = len(slides)

for idx, slide in enumerate(slides, start=1):
    orig = slide["original"]
    enh = slide["enhanced"]
    h, w = orig.shape[:2]

    # Fast metrics: reuse cached if available
    has_same_shape = (orig.shape == enh.shape)
    if show_metrics and has_same_shape:
        if "psnr" in slide and "ssim" in slide and not bw_mode:
            metric_tag = f"PSNR: {slide['psnr']} • SSIM: {slide['ssim']}"
        else:
            metric_tag = f"PSNR: {E.psnr(orig, enh):.2f} dB • SSIM: {E.ssim(orig, enh):.3f}"
    else:
        metric_tag = slide.get("effect_name", "Enhanced")

    # Slide card container
    desc_html = f'<div class="slide-description">{slide["description"]}</div>' if slide.get("description") else ""
    with st.container(border=True):
        header_content = f"<div class=\"slide-title\">{slide['title']}</div>"
        if desc_html:
            header_content += f"\n{desc_html}"
        st.markdown(
            f"""<div class="slide-header">
{header_content}
</div>""",
            unsafe_allow_html=True,
        )

        # Side by side comparison columns
        col_left, col_right = st.columns(2, gap="large")

        with col_left:
            st.markdown(
                f"""<div class="frame-header">
<span class="frame-badge badge-orig">Original</span>
<span class="meta-tag">{w} × {h} px • {orig.ndim}D</span>
</div>""",
                unsafe_allow_html=True,
            )
            show_image(col_left, orig)

        with col_right:
            st.markdown(
                f"""<div class="frame-header">
<span class="frame-badge badge-enh">Enhanced</span>
<span class="metric-badge">{metric_tag}</span>
</div>""",
                unsafe_allow_html=True,
            )
            show_image(col_right, enh)

        # Optional Histograms
        if show_histograms:
            st.markdown("---")
            h_c1, h_c2 = st.columns(2, gap="large")
            with h_c1:
                fig_orig = plot_hist(orig, "Original Tone Histogram")
                st.pyplot(fig_orig)
                plt.close(fig_orig)
            with h_c2:
                fig_enh = plot_hist(enh, "Enhanced Tone Histogram")
                st.pyplot(fig_enh)
                plt.close(fig_enh)

        # Download button with fast buffer caching
        if "png_bytes" in slide and not bw_mode:
            download_data = slide["png_bytes"]
        else:
            save_img = cv2.cvtColor(enh, cv2.COLOR_RGB2BGR) if enh.ndim == 3 else enh
            _, enc_buf = cv2.imencode(".png", save_img)
            download_data = enc_buf.tobytes()

        btn_cols = st.columns([4, 1])
        with btn_cols[1]:
            st.download_button(
                label="Download Result",
                data=download_data,
                file_name=f"enhanced_{slide['id']}.png",
                mime="image/png",
                key=f"dl_btn_{slide['id']}_{idx}",
            )

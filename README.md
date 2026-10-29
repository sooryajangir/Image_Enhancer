# Image Enhancement Studio

An interactive Digital Image Processing (DIP) web application built with **Streamlit**, **OpenCV**, **NumPy**, and **scikit-image**. The studio provides side-by-side comparative visualization and quantitative quality assessment for fundamental image enhancement and restoration techniques.

---

## Features

- **Side-by-Side Comparison**: Direct visual evaluation of original versus enhanced images in unified presentation cards.
- **Scrollable Presentation**: Seamless feed layout displaying key digital image processing transformations without extra button clicks.
- **Core DIP Transformations**:
  - **Low-Light Restoration**: Non-linear power-law gamma correction ($\gamma = 0.65$) to recover shadow detail without clipping highlights.
  - **Dynamic Range & Contrast Stretch**: Luminance-channel histogram equalization to balance tonal distribution.
  - **Speckle & Sensor Noise Suppression**: $3 \times 3$ median rank filtering to eliminate impulse (salt-and-pepper) noise while preserving sharp boundaries.
  - **High-Boost Detail & Edge Sharpening**: Unsharp masking ($\sigma = 1.5, \text{amount} = 1.2$) for restoring defocus blur and edge definition.
- **Custom Image Upload**: Upload any custom photo (PNG, JPG, JPEG, BMP) to immediately run and compare all enhancement pipelines.
- **Black & White (Grayscale) Toggle**: Instant color-to-grayscale conversion to inspect luminance processing directly.
- **Quantitative Quality Metrics**:
  - **PSNR (Peak Signal-to-Noise Ratio)**: Measures pixel-level fidelity in decibels (dB).
  - **SSIM (Structural Similarity Index)**: Measures preservation of luminance, contrast, and structural texture (range 0 to 1).
- **Tone Distribution Histograms**: Optional side-by-side RGB and luminance histograms for examining tonal shifts.
- **One-Click Export**: Download any processed output directly as a high-quality PNG.

---

## Project Structure

```text
DIP PROJECT/
├── .streamlit/
│   └── config.toml       # Custom Streamlit UI theme & typography configuration
├── assets/               # Bundled sample images for offline & cloud deployment
├── app.py                # Main Streamlit web application & interface
├── enhance.py            # Core image processing algorithms & metrics library
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- `pip` package manager

### 2. Clone / Open Directory
Navigate to the project root:
```powershell
cd "c:\Users\elite\Downloads\DIP PROJECT"
```

### 3. Install Dependencies
Install all required libraries using `requirements.txt`:
```powershell
pip install -r requirements.txt
```

---

## Running the Application

Launch the Streamlit server from your terminal:
```powershell
streamlit run app.py
```

The application will start and automatically open in your default browser at:
```text
http://localhost:8501
```

---

## Algorithms Implemented in `enhance.py`

| Category | Method | Description |
| :--- | :--- | :--- |
| **Point Operations** | `brightness_contrast` | Linear point mapping: $s = \alpha \cdot r + \beta$ |
| | `gamma_correction` | Power-law mapping: $s = c \cdot r^\gamma$ |
| | `log_transform` | Logarithmic dynamic range expansion: $s = c \cdot \log(1 + r)$ |
| **Histogram Processing** | `histogram_equalization` | Equalization across luminance channel in YCrCb color space |
| | `clahe` | Contrast Limited Adaptive Histogram Equalization in LAB space |
| **Spatial Filtering** | `convolve` | 2D spatial convolution with reflect padding |
| | `mean_filter` | Linear neighborhood averaging (box filter) |
| | `gaussian_filter` | Gaussian low-pass spatial smoothing |
| | `median_filter` | Non-linear rank-order filter for salt-and-pepper noise |
| **Sharpening & Edges** | `laplacian_sharpen` | Second-order derivative edge enhancement kernel |
| | `unsharp_mask` | High-boost filtering via Gaussian unsharp mask |
| | `sobel_edges` | First-order gradient magnitude detection ($G_x, G_y$) |
| | `canny_edges` | Multi-stage edge detector with hysteresis thresholding |
| **Frequency Filtering** | `frequency_filter` | 2D Fast Fourier Transform (FFT) lowpass/highpass filtering |
| **Quality Evaluation** | `psnr` / `ssim` | Peak signal-to-noise ratio and structural similarity index |

---

## Technologies Used

- **[Streamlit](https://streamlit.io/)**: Reactive web interface framework
- **[OpenCV](https://opencv.org/)**: Computer vision and color space conversions
- **[NumPy](https://numpy.org/)**: Matrix computations and mathematical point operations
- **[scikit-image](https://scikit-image.org/)**: Standard sample datasets and metric implementations
- **[Matplotlib](https://matplotlib.org/)**: Histogram rendering and tonal distribution graphs
- **[Pillow](https://python-pillow.org/)**: Image format handling and buffer management

---

## License

This project is developed for educational and research purposes in Digital Image Processing (DIP). Sample photos are from public domain datasets bundled with `scikit-image`.


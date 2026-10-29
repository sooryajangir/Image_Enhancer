"""Image enhancement functions. All images are uint8 RGB (H x W x 3) or grayscale (H x W)."""
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ---------- 1. Point operations (from scratch with NumPy) ----------
def brightness_contrast(img, alpha=1.0, beta=0):
    """s = alpha * r + beta   (alpha = contrast, beta = brightness)"""
    return np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)


def gamma_correction(img, gamma=1.0):
    """Power-law: s = 255 * (r/255)^gamma"""
    return np.clip(255 * (img / 255.0) ** gamma, 0, 255).astype(np.uint8)


def log_transform(img):
    """s = c * log(1 + r) -- expands dark regions"""
    f = img.astype(np.float32)
    c = 255 / np.log(1 + f.max()) if f.max() > 0 else 1
    return np.clip(c * np.log(1 + f), 0, 255).astype(np.uint8)


# ---------- 2. Histogram processing ----------
def _equalize_channel(ch):
    hist = np.bincount(ch.ravel(), minlength=256)
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0].min()
    denom = cdf[-1] - cdf_min
    if denom == 0:
        return ch
    lut = np.round((cdf - cdf_min) / denom * 255).astype(np.uint8)
    return lut[ch]


def histogram_equalization(img):
    """From scratch. For colour images, equalize only the luminance (Y) channel."""
    if img.ndim == 2:
        return _equalize_channel(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = _equalize_channel(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def clahe(img, clip_limit=2.0, tile=8):
    """Contrast Limited Adaptive Histogram Equalization (OpenCV)."""
    c = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    if img.ndim == 2:
        return c.apply(img)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = c.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ---------- 3. Spatial filtering ----------
def convolve(img, kernel):
    """2-D convolution from scratch (reflect padding), works on gray or colour."""
    k = np.flipud(np.fliplr(kernel)).astype(np.float32)
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2

    def conv_ch(ch):
        padded = np.pad(ch.astype(np.float32), ((ph, ph), (pw, pw)), mode="reflect")
        out = np.zeros(ch.shape, np.float32)
        for i in range(kh):
            for j in range(kw):
                out += k[i, j] * padded[i:i + ch.shape[0], j:j + ch.shape[1]]
        return out

    if img.ndim == 2:
        out = conv_ch(img)
    else:
        out = np.dstack([conv_ch(img[:, :, c]) for c in range(img.shape[2])])
    return np.clip(out, 0, 255).astype(np.uint8)


def mean_filter(img, ksize=3):
    return convolve(img, np.ones((ksize, ksize)) / (ksize * ksize))


def gaussian_filter(img, ksize=5, sigma=1.0):
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def median_filter(img, ksize=3):
    return cv2.medianBlur(img, ksize)  # best for salt & pepper noise


# ---------- 4. Sharpening ----------
def laplacian_sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return convolve(img, kernel)


def unsharp_mask(img, sigma=1.5, amount=1.0):
    """sharp = original + amount * (original - blurred).  amount > 1 = high-boost."""
    blur = cv2.GaussianBlur(img, (0, 0), sigma).astype(np.float32)
    out = img.astype(np.float32) + amount * (img.astype(np.float32) - blur)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------- 5. Edge detection ----------
def _gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def sobel_edges(img):
    g = _gray(img).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return np.clip(np.hypot(gx, gy), 0, 255).astype(np.uint8)


def canny_edges(img, low=100, high=200):
    return cv2.Canny(_gray(img), low, high)


# ---------- 6. Frequency-domain filtering ----------
def frequency_filter(img, kind="lowpass", shape="gaussian", cutoff=30, order=2):
    """kind: lowpass/highpass.  shape: ideal/butterworth/gaussian."""
    def filt(ch):
        rows, cols = ch.shape
        u = np.arange(rows) - rows / 2
        v = np.arange(cols) - cols / 2
        V, U = np.meshgrid(v, u)
        D = np.sqrt(U ** 2 + V ** 2)
        if shape == "ideal":
            H = (D <= cutoff).astype(np.float32)
        elif shape == "butterworth":
            H = 1 / (1 + (D / cutoff) ** (2 * order))
        else:
            H = np.exp(-(D ** 2) / (2 * cutoff ** 2))
        if kind == "highpass":
            H = 1 - H
        F = np.fft.fftshift(np.fft.fft2(ch.astype(np.float32)))
        return np.real(np.fft.ifft2(np.fft.ifftshift(F * H)))

    if img.ndim == 2:
        out = filt(img)
    else:
        out = np.dstack([filt(img[:, :, c]) for c in range(img.shape[2])])
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------- Quality metrics ----------
def psnr(original, enhanced):
    return peak_signal_noise_ratio(original, enhanced, data_range=255)


def ssim(original, enhanced):
    ch_axis = -1 if original.ndim == 3 else None
    return structural_similarity(original, enhanced, channel_axis=ch_axis, data_range=255)

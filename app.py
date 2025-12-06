"""
Interactive Digital Image Processing Application
Enhancement, FFT, Color Processing, and Segmentation Berbasis Slider Parameter

Disusun Oleh:
- Fadaukas Daffa Tajuddin (5025231149)
- Davin Amadeo Wijaya (5025231204)
- Reihan Arianza (5025231274)

Institut Teknologi Sepuluh Nopember Surabaya
November 2025
"""

import streamlit as st
import numpy as np
import cv2
from skimage import filters, segmentation, color, exposure, feature, morphology
from skimage.util import img_as_float, img_as_ubyte
from scipy import ndimage
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

# ================================
# KONFIGURASI HALAMAN
# ================================
st.set_page_config(
    page_title="Interactive Digital Image Processing - ITS",
    layout="wide",
    page_icon="🎓"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 25px;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .module-header {
        background-color: #000000;
        padding: 15px;
        border-left: 5px solid #1e3c72;
        border-radius: 5px;
        margin: 20px 0 15px 0;
        font-weight: bold;
    }
    .info-box {
        background-color: #000000;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
    }
    .stButton>button {
        background-color: #1e3c72;
        color: white;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2a5298;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# HEADER
# ================================
st.markdown("""
<div class="main-header">
    <h1>🎓 Interactive Digital Image Processing Application</h1>
    <p style="font-size: 18px;">Enhancement • FFT Filtering • Color Processing • Segmentation Berbasis Slider Parameter</p>
    <p style="font-size: 14px; margin-top: 10px;">Institut Teknologi Sepuluh Nopember Surabaya</p>
</div>
""", unsafe_allow_html=True)

# ================================
# SIDEBAR - INFORMASI DAN KONFIGURASI
# ================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/ITS_Surabaya_Logo.svg/1200px-ITS_Surabaya_Logo.svg.png", width=150)
    
    st.markdown("### 👥 Tim Pengembang")
    st.markdown("""
    - Fadaukas Daffa Tajuddin (5025231149)
    - Davin Amadeo Wijaya (5025231204)
    - Reihan Arianza (5025231274)
    """)
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Pengaturan")
    
    # Mode operasi
    operation_mode = st.radio(
        "Mode Operasi",
        ["Pipeline Lengkap", "Modul Individual"],
        help="Pilih untuk menjalankan seluruh pipeline atau modul terpisah"
    )
    
    # Format download
    download_format = st.selectbox(
        "Format Download",
        ["PNG", "JPEG", "BMP"]
    )
    
    # Opsi tampilan
    st.markdown("### 📊 Opsi Tampilan")
    show_histogram = st.checkbox("Tampilkan Histogram", value=False)
    show_spectrum = st.checkbox("Tampilkan Spektrum FFT", value=True)
    show_info = st.checkbox("Tampilkan Info Citra", value=True)
    
    st.markdown("---")
    
    st.markdown("### 📚 Tentang Aplikasi")
    st.info("""
    Aplikasi ini mengimplementasikan pipeline pengolahan citra digital yang mencakup:
    
    **1. Enhancement (Domain Spasial)**
    - Transformasi Intensitas
    - Histogram Processing
    - Filtering Spasial
    
    **2. FFT Filtering (Domain Frekuensi)**
    - Low-Pass & High-Pass Filter
    - Butterworth & Gaussian Filter
    - Homomorphic Filtering
    
    **3. Color Model Processing**
    - RGB to HSI/HSV
    - Channel Extraction
    - Pseudocolor
    
    **4. Feature-Based Segmentation**
    - Edge Detection
    - Thresholding
    - Region-Based Methods
    """)

# ================================
# FUNGSI HELPER
# ================================

def create_download_link(img, filename, format_type):
    """Create download button for processed images"""
    buffered = BytesIO()
    # Handle grayscale images
    if len(img.shape) == 2:
        img_pil = Image.fromarray(img)
    else:
        img_pil = Image.fromarray(img)
    img_pil.save(buffered, format=format_type)
    return buffered.getvalue()

def display_histogram(img, title="Histogram"):
    """Display histogram"""
    fig, ax = plt.subplots(figsize=(8, 3))
    
    if len(img.shape) == 2:  # Grayscale
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        ax.plot(hist, color='black')
        ax.set_xlim([0, 256])
    else:  # RGB
        colors = ('r', 'g', 'b')
        for i, c in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            ax.plot(hist, color=c, alpha=0.7, label=f'{c.upper()} channel')
        ax.legend()
        ax.set_xlim([0, 256])
    
    ax.set_xlabel('Intensitas Piksel')
    ax.set_ylabel('Frekuensi')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig

def show_image_info(img):
    """Display image information"""
    height, width = img.shape[:2]
    channels = img.shape[2] if len(img.shape) > 2 else 1
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Lebar", f"{width}px")
    with col2:
        st.metric("Tinggi", f"{height}px")
    with col3:
        st.metric("Channel", channels)
    with col4:
        st.metric("Ukuran", f"{img.nbytes / 1024:.2f}KB")

def compute_fft(img_gray):
    """Compute FFT and return shifted spectrum"""
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    return fshift

def display_fft_spectrum(fshift, title="FFT Magnitude Spectrum"):
    """Display FFT magnitude spectrum"""
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(magnitude_spectrum, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    return fig

# ================================
# UPLOAD CITRA
# ================================
st.markdown('<div class="module-header">📁 INPUT CITRA</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Citra (PNG/JPG/BMP)",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Upload citra untuk diproses melalui pipeline"
)

if uploaded_file:
    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_float = img_as_float(img_rgb)
    
    # Pilih mode grayscale atau color
    color_mode = st.radio(
        "Pilih Mode Pemrosesan",
        ["Grayscale", "Color (RGB)"],
        horizontal=True
    )
    
    if color_mode == "Grayscale":
        img_working = img_gray.copy()
        is_color = False
    else:
        img_working = img_rgb.copy()
        is_color = True
    
    # Display original image
    col_orig1, col_orig2 = st.columns([2, 1])
    with col_orig1:
        st.image(img_working if is_color else img_gray, 
                caption=f"Citra Asli ({color_mode})", 
                use_container_width=True)
    
    with col_orig2:
        if show_info:
            st.markdown("**Informasi Citra**")
            show_image_info(img_working)
        
        if show_histogram:
            st.markdown("**Histogram Citra Asli**")
            fig_hist = display_histogram(img_working, "Histogram Asli")
            st.pyplot(fig_hist)
            plt.close()
    
    # =====================================================
    # MODUL 1: ENHANCEMENT (DOMAIN SPASIAL)
    # =====================================================
    st.markdown('<div class="module-header">✨ MODUL 1: ENHANCEMENT (DOMAIN SPASIAL)</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">Meningkatkan kualitas visual citra melalui manipulasi nilai piksel secara langsung.</div>', unsafe_allow_html=True)
    
    enhancement_tabs = st.tabs([
        "Transformasi Intensitas", 
        "Histogram Processing", 
        "Filtering Spasial"
    ])
    
    # Tab 1: Transformasi Intensitas
    with enhancement_tabs[0]:
        st.markdown("#### 🎚️ Transformasi Intensitas")
        
        transform_type = st.selectbox(
            "Pilih Metode Transformasi",
            ["Tidak Ada", "Gamma Correction", "Log Transform", "Contrast Stretching", 
             "Brightness/Contrast Adjustment"]
        )
        
        img_enhanced = img_working.copy()
        
        if transform_type == "Gamma Correction":
            st.markdown("**Gamma Correction**: $s = cr^γ$ dimana $c$ adalah konstanta dan $γ$ adalah parameter gamma")
            gamma = st.slider("Nilai Gamma (γ)", 0.1, 3.0, 1.0, 0.1,
                            help="γ < 1: meningkatkan intensitas area gelap, γ > 1: mengurangi intensitas")
            
            if is_color:
                img_enhanced = exposure.adjust_gamma(img_working, gamma)
                img_enhanced = (img_enhanced * 255).astype(np.uint8)
            else:
                img_enhanced = exposure.adjust_gamma(img_working, gamma)
                img_enhanced = (img_enhanced * 255).astype(np.uint8) if img_enhanced.dtype == np.float64 else img_enhanced
        
        elif transform_type == "Log Transform":
            st.markdown("**Log Transform**: $s = c \\log(1 + r)$ untuk meningkatkan detail pada area gelap")
            c_constant = st.slider("Konstanta c", 1, 100, 40,
                                  help="Konstanta pengali untuk transformasi log")
            
            if is_color:
                img_float_temp = img_working.astype(np.float32)
                img_enhanced = c_constant * np.log(1 + img_float_temp)
                img_enhanced = np.clip(img_enhanced, 0, 255).astype(np.uint8)
            else:
                img_float_temp = img_working.astype(np.float32)
                img_enhanced = c_constant * np.log(1 + img_float_temp)
                img_enhanced = np.clip(img_enhanced, 0, 255).astype(np.uint8)
        
        elif transform_type == "Contrast Stretching":
            st.markdown("**Contrast Stretching**: Memperluas rentang intensitas piksel")
            p_low = st.slider("Persentil Bawah (%)", 0, 50, 2)
            p_high = st.slider("Persentil Atas (%)", 50, 100, 98)
            
            if is_color:
                p2, p98 = np.percentile(img_working, (p_low, p_high))
                img_enhanced = exposure.rescale_intensity(img_working, in_range=(p2, p98))
            else:
                p2, p98 = np.percentile(img_working, (p_low, p_high))
                img_enhanced = exposure.rescale_intensity(img_working, in_range=(p2, p98))
        
        elif transform_type == "Brightness/Contrast Adjustment":
            col_bc1, col_bc2 = st.columns(2)
            with col_bc1:
                brightness = st.slider("Brightness", -100, 100, 0)
            with col_bc2:
                contrast = st.slider("Contrast", 0.5, 3.0, 1.0, 0.1)
            
            img_enhanced = img_working.astype(np.float32)
            img_enhanced = img_enhanced * contrast + brightness
            img_enhanced = np.clip(img_enhanced, 0, 255).astype(np.uint8)
        
        if transform_type != "Tidak Ada":
            col_trans1, col_trans2 = st.columns(2)
            with col_trans1:
                st.image(img_working, caption="Sebelum Transformasi", use_container_width=True)
            with col_trans2:
                st.image(img_enhanced, caption=f"Setelah {transform_type}", use_container_width=True)
            
            if show_histogram:
                col_hist1, col_hist2 = st.columns(2)
                with col_hist1:
                    fig1 = display_histogram(img_working, "Histogram Sebelum")
                    st.pyplot(fig1)
                    plt.close()
                with col_hist2:
                    fig2 = display_histogram(img_enhanced, "Histogram Setelah")
                    st.pyplot(fig2)
                    plt.close()
            
            # Update working image
            img_working = img_enhanced.copy()
    
    # Tab 2: Histogram Processing
    with enhancement_tabs[1]:
        st.markdown("#### 📊 Histogram Processing")
        
        hist_method = st.selectbox(
            "Pilih Metode Histogram",
            ["Tidak Ada", "Histogram Equalization", "CLAHE (Contrast Limited AHE)"]
        )
        
        img_hist = img_working.copy()
        
        if hist_method == "Histogram Equalization":
            st.markdown("**Histogram Equalization**: Meredistribusi intensitas untuk meningkatkan kontras global")
            
            if is_color:
                # Convert to YCrCb and equalize Y channel
                ycrcb = cv2.cvtColor(img_working, cv2.COLOR_RGB2YCrCb)
                ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                img_hist = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
            else:
                img_hist = cv2.equalizeHist(img_working)
        
        elif hist_method == "CLAHE (Contrast Limited AHE)":
            st.markdown("**CLAHE**: Adaptive Histogram Equalization dengan batas kontras untuk mencegah noise berlebih")
            
            col_clahe1, col_clahe2 = st.columns(2)
            with col_clahe1:
                clip_limit = st.slider("Clip Limit", 1.0, 10.0, 2.0, 0.5,
                                      help="Batas untuk mencegah amplifikasi noise")
            with col_clahe2:
                tile_size = st.slider("Tile Grid Size", 4, 16, 8, 2,
                                     help="Ukuran grid untuk pemrosesan lokal")
            
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
            
            if is_color:
                ycrcb = cv2.cvtColor(img_working, cv2.COLOR_RGB2YCrCb)
                ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
                img_hist = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
            else:
                img_hist = clahe.apply(img_working)
        
        if hist_method != "Tidak Ada":
            col_hist1, col_hist2 = st.columns(2)
            with col_hist1:
                st.image(img_working, caption="Sebelum", use_container_width=True)
                if show_histogram:
                    fig1 = display_histogram(img_working, "Histogram Sebelum")
                    st.pyplot(fig1)
                    plt.close()
            
            with col_hist2:
                st.image(img_hist, caption=f"Setelah {hist_method}", use_container_width=True)
                if show_histogram:
                    fig2 = display_histogram(img_hist, "Histogram Setelah")
                    st.pyplot(fig2)
                    plt.close()
            
            # Update working image
            img_working = img_hist.copy()
    
    # Tab 3: Filtering Spasial
    with enhancement_tabs[2]:
        st.markdown("#### 🔧 Filtering Spasial")
        
        filter_type = st.selectbox(
            "Pilih Filter Spasial",
            ["Tidak Ada", "Mean Filter (Smoothing)", "Gaussian Filter (Smoothing)", 
             "Median Filter (Noise Removal)", "Laplacian (Sharpening)", 
             "Sobel (Edge Enhancement)", "Unsharp Masking"]
        )
        
        img_filtered = img_working.copy()
        
        if filter_type == "Mean Filter (Smoothing)":
            st.markdown("**Mean Filter**: Filter smoothing sederhana dengan rata-rata lokal")
            kernel_size = st.slider("Ukuran Kernel", 3, 15, 5, 2)
            img_filtered = cv2.blur(img_working, (kernel_size, kernel_size))
        
        elif filter_type == "Gaussian Filter (Smoothing)":
            st.markdown("**Gaussian Filter**: Smoothing dengan bobot Gaussian untuk hasil lebih halus")
            kernel_size = st.slider("Ukuran Kernel", 3, 15, 5, 2)
            sigma = st.slider("Sigma", 0.1, 5.0, 1.0, 0.1)
            img_filtered = cv2.GaussianBlur(img_working, (kernel_size, kernel_size), sigma)
        
        elif filter_type == "Median Filter (Noise Removal)":
            st.markdown("**Median Filter**: Efektif untuk menghilangkan salt & pepper noise tanpa mengaburkan tepi")
            kernel_size = st.slider("Ukuran Kernel", 3, 15, 5, 2)
            
            if is_color:
                img_filtered = cv2.medianBlur(img_working, kernel_size)
            else:
                img_filtered = cv2.medianBlur(img_working, kernel_size)
        
        elif filter_type == "Laplacian (Sharpening)":
            st.markdown("**Laplacian**: Deteksi tepi berdasarkan turunan kedua untuk sharpening")
            
            if is_color:
                gray_temp = cv2.cvtColor(img_working, cv2.COLOR_RGB2GRAY)
            else:
                gray_temp = img_working
            
            laplacian = cv2.Laplacian(gray_temp, cv2.CV_64F)
            laplacian = np.uint8(np.absolute(laplacian))
            
            if is_color:
                img_filtered = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)
            else:
                img_filtered = laplacian
        
        elif filter_type == "Sobel (Edge Enhancement)":
            st.markdown("**Sobel**: Deteksi tepi menggunakan turunan pertama (gradien)")
            
            if is_color:
                gray_temp = cv2.cvtColor(img_working, cv2.COLOR_RGB2GRAY)
            else:
                gray_temp = img_working
            
            sobel_x = cv2.Sobel(gray_temp, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_temp, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel = np.uint8(sobel / sobel.max() * 255)
            
            if is_color:
                img_filtered = cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)
            else:
                img_filtered = sobel
        
        elif filter_type == "Unsharp Masking":
            st.markdown("**Unsharp Masking**: Meningkatkan ketajaman dengan mengurangi versi blur dari citra asli")
            
            col_unsharp1, col_unsharp2 = st.columns(2)
            with col_unsharp1:
                radius = st.slider("Radius", 0.5, 5.0, 1.0, 0.5)
            with col_unsharp2:
                amount = st.slider("Amount", 0.5, 3.0, 1.5, 0.1)
            
            if is_color:
                img_float_temp = img_as_float(img_working)
                img_filtered = filters.unsharp_mask(img_float_temp, radius=radius, amount=amount)
                img_filtered = (img_filtered * 255).astype(np.uint8)
            else:
                img_float_temp = img_as_float(img_working)
                img_filtered = filters.unsharp_mask(img_float_temp, radius=radius, amount=amount)
                img_filtered = (img_filtered * 255).astype(np.uint8)
        
        if filter_type != "Tidak Ada":
            col_filt1, col_filt2 = st.columns(2)
            with col_filt1:
                st.image(img_working, caption="Sebelum Filtering", use_container_width=True)
            with col_filt2:
                st.image(img_filtered, caption=f"Setelah {filter_type}", use_container_width=True)
            
            # Update working image
            img_working = img_filtered.copy()
    
    # =====================================================
    # MODUL 2: FFT FILTERING (DOMAIN FREKUENSI)
    # =====================================================
    st.markdown('<div class="module-header">📈 MODUL 2: FFT FILTERING (DOMAIN FREKUENSI)</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">Pemrosesan citra dalam domain frekuensi menggunakan Fast Fourier Transform untuk filtering dan analisis.</div>', unsafe_allow_html=True)
    
    fft_tabs = st.tabs(["FFT Visualization", "Frequency Filtering", "Homomorphic Filtering"])
    
    # Prepare grayscale for FFT
    if is_color:
        img_gray_fft = cv2.cvtColor(img_working, cv2.COLOR_RGB2GRAY)
    else:
        img_gray_fft = img_working.copy()
    
    # Compute FFT
    fshift = compute_fft(img_gray_fft)
    
    # Tab 1: FFT Visualization
    with fft_tabs[0]:
        st.markdown("#### 🔬 Visualisasi Spektrum FFT")
        
        col_fft1, col_fft2 = st.columns(2)
        
        with col_fft1:
            st.markdown("**Magnitude Spectrum**")
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
            st.image(magnitude_spectrum, caption="Magnitude Spectrum", clamp=True, use_container_width=True)
        
        with col_fft2:
            st.markdown("**Phase Spectrum**")
            phase_spectrum = np.angle(fshift)
            st.image(phase_spectrum, caption="Phase Spectrum", clamp=True, use_container_width=True)
        
        if show_spectrum:
            fig_spectrum = display_fft_spectrum(fshift, "FFT Magnitude Spectrum")
            st.pyplot(fig_spectrum)
            plt.close()
    
    # Tab 2: Frequency Filtering
    with fft_tabs[1]:
        st.markdown("#### 🎛️ Frequency Domain Filtering")
        
        freq_filter_type = st.selectbox(
            "Pilih Filter Frekuensi",
            ["Tidak Ada", "Ideal Low-Pass Filter", "Ideal High-Pass Filter",
             "Butterworth Low-Pass Filter", "Butterworth High-Pass Filter",
             "Gaussian Low-Pass Filter", "Gaussian High-Pass Filter"]
        )
        
        if freq_filter_type != "Tidak Ada":
            col_freq1, col_freq2 = st.columns(2)
            
            with col_freq1:
                D0 = st.slider("Cutoff Frequency (D0)", 10, 200, 30,
                             help="Frekuensi cutoff untuk filtering")
            
            with col_freq2:
                if "Butterworth" in freq_filter_type:
                    n_order = st.slider("Order (n)", 1, 10, 2,
                                       help="Order filter Butterworth")
            
            rows, cols = img_gray_fft.shape
            crow, ccol = rows // 2, cols // 2
            
            # Create filter mask
            x = np.arange(cols)
            y = np.arange(rows)
            X, Y = np.meshgrid(x - ccol, y - crow)
            D = np.sqrt(X**2 + Y**2)
            
            if "Ideal" in freq_filter_type:
                if "Low-Pass" in freq_filter_type:
                    mask = (D <= D0).astype(float)
                else:  # High-Pass
                    mask = (D > D0).astype(float)
            
            elif "Butterworth" in freq_filter_type:
                if "Low-Pass" in freq_filter_type:
                    mask = 1 / (1 + (D / D0)**(2 * n_order))
                else:  # High-Pass
                    mask = 1 / (1 + (D0 / (D + 1e-6))**(2 * n_order))
            
            elif "Gaussian" in freq_filter_type:
                if "Low-Pass" in freq_filter_type:
                    mask = np.exp(-(D**2) / (2 * (D0**2)))
                else:  # High-Pass
                    mask = 1 - np.exp(-(D**2) / (2 * (D0**2)))
            
            # Apply filter
            fshift_filtered = fshift * mask
            f_ishift = np.fft.ifftshift(fshift_filtered)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            img_back = np.clip(img_back, 0, 255).astype(np.uint8)
            
            # Display results
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                st.image(img_gray_fft, caption="Original", use_container_width=True)
            
            with col_res2:
                st.image(mask, caption="Filter Mask", clamp=True, use_container_width=True)
            
            with col_res3:
                st.image(img_back, caption="Filtered Result", use_container_width=True)
            
            # Update working image if applicable
            if not is_color:
                img_working = img_back.copy()
    
    # Tab 3: Homomorphic Filtering
    with fft_tabs[2]:
        st.markdown("#### 💡 Homomorphic Filtering")
        st.markdown("**Homomorphic Filtering**: Memisahkan komponen iluminasi dan reflektansi untuk normalisasi pencahayaan")
        
        apply_homomorphic = st.checkbox("Terapkan Homomorphic Filtering")
        
        if apply_homomorphic:
            col_homo1, col_homo2, col_homo3 = st.columns(3)
            
            with col_homo1:
                gamma_L = st.slider("Gamma L (Low Freq)", 0.1, 1.0, 0.5, 0.1,
                                   help="Gain untuk komponen iluminasi (frekuensi rendah)")
            
            with col_homo2:
                gamma_H = st.slider("Gamma H (High Freq)", 1.0, 3.0, 1.5, 0.1,
                                   help="Gain untuk komponen reflektansi (frekuensi tinggi)")
            
            with col_homo3:
                c_homo = st.slider("Cutoff Frequency", 10, 100, 30)
            
            # Apply homomorphic filtering
            img_log = np.log1p(img_gray_fft.astype(np.float64))
            fshift_homo = compute_fft(img_log)
            
            rows, cols = img_gray_fft.shape
            crow, ccol = rows // 2, cols // 2
            x = np.arange(cols)
            y = np.arange(rows)
            X, Y = np.meshgrid(x - ccol, y - crow)
            D = np.sqrt(X**2 + Y**2)
            
            # Homomorphic filter
            H = (gamma_H - gamma_L) * (1 - np.exp(-(D**2) / (2 * (c_homo**2)))) + gamma_L
            
            fshift_filtered_homo = fshift_homo * H
            f_ishift_homo = np.fft.ifftshift(fshift_filtered_homo)
            img_back_homo = np.fft.ifft2(f_ishift_homo)
            img_back_homo = np.real(img_back_homo)
            img_back_homo = np.expm1(img_back_homo)
            img_back_homo = np.clip(img_back_homo, 0, 255).astype(np.uint8)
            
            col_homo_res1, col_homo_res2 = st.columns(2)
            
            with col_homo_res1:
                st.image(img_gray_fft, caption="Sebelum Homomorphic", use_container_width=True)
            
            with col_homo_res2:
                st.image(img_back_homo, caption="Setelah Homomorphic Filtering", use_container_width=True)
            
            if not is_color:
                img_working = img_back_homo.copy()
    
    # =====================================================
    # MODUL 3: COLOR MODEL PROCESSING
    # =====================================================
    if is_color:
        st.markdown('<div class="module-header">🎨 MODUL 3: COLOR MODEL PROCESSING</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-box">Transformasi dan pemrosesan citra dalam berbagai model warna untuk analisis berbasis warna.</div>', unsafe_allow_html=True)
        
        color_tabs = st.tabs(["Color Space Transformation", "Channel Extraction", "Pseudocolor"])
        
        # Tab 1: Color Space Transformation
        with color_tabs[0]:
            st.markdown("#### 🔄 Transformasi Color Space")
            
            target_space = st.selectbox(
                "Pilih Color Space Target",
                ["RGB (Original)", "HSV", "HSI", "CMY", "YCrCb", "LAB"]
            )
            
            if target_space == "RGB (Original)":
                img_color_transformed = img_working.copy()
                st.image(img_color_transformed, caption="RGB Color Space", use_container_width=True)
            
            elif target_space == "HSV":
                st.markdown("**HSV**: Hue (warna), Saturation (kejenuhan), Value (intensitas)")
                hsv = cv2.cvtColor(img_working, cv2.COLOR_RGB2HSV)
                
                col_hsv1, col_hsv2, col_hsv3 = st.columns(3)
                with col_hsv1:
                    st.image(hsv[:, :, 0], caption="H (Hue)", use_container_width=True)
                with col_hsv2:
                    st.image(hsv[:, :, 1], caption="S (Saturation)", use_container_width=True)
                with col_hsv3:
                    st.image(hsv[:, :, 2], caption="V (Value)", use_container_width=True)
                
                img_color_transformed = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            elif target_space == "HSI":
                st.markdown("**HSI**: Hue, Saturation, Intensity - memisahkan warna dari intensitas")
                
                # Manual HSI conversion
                rgb_normalized = img_working.astype(np.float32) / 255.0
                R, G, B = rgb_normalized[:,:,0], rgb_normalized[:,:,1], rgb_normalized[:,:,2]
                
                # Intensity
                I = (R + G + B) / 3.0
                
                # Saturation
                min_rgb = np.minimum(np.minimum(R, G), B)
                S = 1 - (3 / (R + G + B + 1e-6)) * min_rgb
                
                # Hue
                num = 0.5 * ((R - G) + (R - B))
                den = np.sqrt((R - G)**2 + (R - B) * (G - B)) + 1e-6
                theta = np.arccos(np.clip(num / den, -1, 1))
                H = np.where(B <= G, theta, 2 * np.pi - theta)
                H = H / (2 * np.pi)  # Normalize to [0, 1]
                
                col_hsi1, col_hsi2, col_hsi3 = st.columns(3)
                with col_hsi1:
                    st.image(H, caption="H (Hue)", clamp=True, use_container_width=True)
                with col_hsi2:
                    st.image(S, caption="S (Saturation)", clamp=True, use_container_width=True)
                with col_hsi3:
                    st.image(I, caption="I (Intensity)", clamp=True, use_container_width=True)
            
            elif target_space == "CMY":
                st.markdown("**CMY**: Cyan, Magenta, Yellow - model subtraktif")
                cmy = 255 - img_working
                
                col_cmy1, col_cmy2, col_cmy3 = st.columns(3)
                with col_cmy1:
                    st.image(cmy[:, :, 0], caption="C (Cyan)", use_container_width=True)
                with col_cmy2:
                    st.image(cmy[:, :, 1], caption="M (Magenta)", use_container_width=True)
                with col_cmy3:
                    st.image(cmy[:, :, 2], caption="Y (Yellow)", use_container_width=True)
            
            elif target_space == "YCrCb":
                st.markdown("**YCrCb**: Luminance (Y), Chrominance Red (Cr), Chrominance Blue (Cb)")
                ycrcb = cv2.cvtColor(img_working, cv2.COLOR_RGB2YCrCb)
                
                col_ycrcb1, col_ycrcb2, col_ycrcb3 = st.columns(3)
                with col_ycrcb1:
                    st.image(ycrcb[:, :, 0], caption="Y (Luminance)", use_container_width=True)
                with col_ycrcb2:
                    st.image(ycrcb[:, :, 1], caption="Cr (Chroma Red)", use_container_width=True)
                with col_ycrcb3:
                    st.image(ycrcb[:, :, 2], caption="Cb (Chroma Blue)", use_container_width=True)
            
            elif target_space == "LAB":
                st.markdown("**LAB**: Lightness (L), A (green-red), B (blue-yellow)")
                lab = cv2.cvtColor(img_working, cv2.COLOR_RGB2LAB)
                
                col_lab1, col_lab2, col_lab3 = st.columns(3)
                with col_lab1:
                    st.image(lab[:, :, 0], caption="L (Lightness)", use_container_width=True)
                with col_lab2:
                    st.image(lab[:, :, 1], caption="A (green-red)", use_container_width=True)
                with col_lab3:
                    st.image(lab[:, :, 2], caption="B (blue-yellow)", use_container_width=True)
        
        # Tab 2: Channel Extraction
        with color_tabs[1]:
            st.markdown("#### 📤 Ekstraksi Channel")
            
            extract_mode = st.radio(
                "Mode Ekstraksi",
                ["RGB Channels", "HSV Channels", "Individual Channel Processing"],
                horizontal=True
            )
            
            if extract_mode == "RGB Channels":
                col_rgb1, col_rgb2, col_rgb3 = st.columns(3)
                
                with col_rgb1:
                    r_channel = np.zeros_like(img_working)
                    r_channel[:, :, 0] = img_working[:, :, 0]
                    st.image(r_channel, caption="Red Channel", use_container_width=True)
                
                with col_rgb2:
                    g_channel = np.zeros_like(img_working)
                    g_channel[:, :, 1] = img_working[:, :, 1]
                    st.image(g_channel, caption="Green Channel", use_container_width=True)
                
                with col_rgb3:
                    b_channel = np.zeros_like(img_working)
                    b_channel[:, :, 2] = img_working[:, :, 2]
                    st.image(b_channel, caption="Blue Channel", use_container_width=True)
            
            elif extract_mode == "HSV Channels":
                hsv_extract = cv2.cvtColor(img_working, cv2.COLOR_RGB2HSV)
                
                col_hsv_ex1, col_hsv_ex2, col_hsv_ex3 = st.columns(3)
                
                with col_hsv_ex1:
                    st.image(hsv_extract[:, :, 0], caption="Hue", use_container_width=True)
                
                with col_hsv_ex2:
                    st.image(hsv_extract[:, :, 1], caption="Saturation", use_container_width=True)
                
                with col_hsv_ex3:
                    st.image(hsv_extract[:, :, 2], caption="Value", use_container_width=True)
            
            elif extract_mode == "Individual Channel Processing":
                st.markdown("Proses channel individual untuk segmentasi berbasis warna")
                
                channel_select = st.selectbox(
                    "Pilih Channel untuk Diproses",
                    ["Hue (H)", "Saturation (S)", "Value (V)"]
                )
                
                hsv_proc = cv2.cvtColor(img_working, cv2.COLOR_RGB2HSV)
                
                if channel_select == "Hue (H)":
                    channel_idx = 0
                    min_val, max_val = st.slider("Range Hue", 0, 179, (0, 179))
                elif channel_select == "Saturation (S)":
                    channel_idx = 1
                    min_val, max_val = st.slider("Range Saturation", 0, 255, (0, 255))
                else:  # Value
                    channel_idx = 2
                    min_val, max_val = st.slider("Range Value", 0, 255, (0, 255))
                
                # Create mask
                mask = cv2.inRange(hsv_proc[:, :, channel_idx], min_val, max_val)
                result = cv2.bitwise_and(img_working, img_working, mask=mask)
                
                col_ch_proc1, col_ch_proc2 = st.columns(2)
                with col_ch_proc1:
                    st.image(hsv_proc[:, :, channel_idx], caption=f"{channel_select} Channel", use_container_width=True)
                with col_ch_proc2:
                    st.image(result, caption=f"Filtered Result ({channel_select})", use_container_width=True)
        
        # Tab 3: Pseudocolor
        with color_tabs[2]:
            st.markdown("#### 🎨 Pseudocolor")
            st.markdown("Menambahkan warna buatan pada citra grayscale untuk meningkatkan persepsi visual")
            
            # Convert to grayscale first
            gray_pseudo = cv2.cvtColor(img_working, cv2.COLOR_RGB2GRAY)
            
            colormap = st.selectbox(
                "Pilih Colormap",
                ["JET", "HOT", "COOL", "RAINBOW", "VIRIDIS", "PLASMA"]
            )
            
            colormap_dict = {
                "JET": cv2.COLORMAP_JET,
                "HOT": cv2.COLORMAP_HOT,
                "COOL": cv2.COLORMAP_COOL,
                "RAINBOW": cv2.COLORMAP_RAINBOW,
                "VIRIDIS": cv2.COLORMAP_VIRIDIS,
                "PLASMA": cv2.COLORMAP_PLASMA
            }
            
            pseudo_colored = cv2.applyColorMap(gray_pseudo, colormap_dict[colormap])
            pseudo_colored = cv2.cvtColor(pseudo_colored, cv2.COLOR_BGR2RGB)
            
            col_pseudo1, col_pseudo2 = st.columns(2)
            with col_pseudo1:
                st.image(gray_pseudo, caption="Grayscale", use_container_width=True)
            with col_pseudo2:
                st.image(pseudo_colored, caption=f"Pseudocolor ({colormap})", use_container_width=True)
    
    # =====================================================
    # MODUL 4: FEATURE-BASED SEGMENTATION
    # =====================================================
    st.markdown('<div class="module-header">🎯 MODUL 4: FEATURE-BASED SEGMENTATION</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">Segmentasi citra untuk memisahkan objek dari latar belakang menggunakan berbagai metode berbasis fitur.</div>', unsafe_allow_html=True)
    
    seg_tabs = st.tabs(["Edge-Based", "Thresholding", "Region-Based", "Advanced Methods"])
    
    # Prepare grayscale for segmentation
    if is_color:
        img_gray_seg = cv2.cvtColor(img_working, cv2.COLOR_RGB2GRAY)
    else:
        img_gray_seg = img_working.copy()
    
    # Tab 1: Edge-Based Segmentation
    with seg_tabs[0]:
        st.markdown("#### 🔍 Edge-Based Segmentation")
        
        edge_method = st.selectbox(
            "Pilih Metode Deteksi Tepi",
            ["Sobel", "Prewitt", "Roberts", "Canny", "LoG (Laplacian of Gaussian)"]
        )
        
        if edge_method == "Sobel":
            st.markdown("**Sobel**: Deteksi tepi menggunakan operator gradien Sobel")
            
            sobel_x = cv2.Sobel(img_gray_seg, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(img_gray_seg, cv2.CV_64F, 0, 1, ksize=3)
            sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel_combined = np.uint8(sobel_combined / sobel_combined.max() * 255)
            
            col_edge1, col_edge2, col_edge3 = st.columns(3)
            with col_edge1:
                st.image(np.uint8(np.abs(sobel_x) / np.abs(sobel_x).max() * 255), 
                        caption="Sobel X", use_container_width=True)
            with col_edge2:
                st.image(np.uint8(np.abs(sobel_y) / np.abs(sobel_y).max() * 255), 
                        caption="Sobel Y", use_container_width=True)
            with col_edge3:
                st.image(sobel_combined, caption="Sobel Combined", use_container_width=True)
        
        elif edge_method == "Prewitt":
            st.markdown("**Prewitt**: Deteksi tepi menggunakan operator Prewitt")
            
            kernel_prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            kernel_prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            
            prewitt_x = cv2.filter2D(img_gray_seg, -1, kernel_prewitt_x)
            prewitt_y = cv2.filter2D(img_gray_seg, -1, kernel_prewitt_y)
            prewitt_combined = np.sqrt(prewitt_x.astype(float)**2 + prewitt_y.astype(float)**2)
            prewitt_combined = np.uint8(prewitt_combined / prewitt_combined.max() * 255)
            
            st.image(prewitt_combined, caption="Prewitt Edge Detection", use_container_width=True)
        
        elif edge_method == "Roberts":
            st.markdown("**Roberts**: Deteksi tepi menggunakan operator Roberts")
            
            kernel_roberts_x = np.array([[1, 0], [0, -1]])
            kernel_roberts_y = np.array([[0, 1], [-1, 0]])
            
            roberts_x = cv2.filter2D(img_gray_seg, -1, kernel_roberts_x)
            roberts_y = cv2.filter2D(img_gray_seg, -1, kernel_roberts_y)
            roberts_combined = np.sqrt(roberts_x.astype(float)**2 + roberts_y.astype(float)**2)
            roberts_combined = np.uint8(roberts_combined / roberts_combined.max() * 255)
            
            st.image(roberts_combined, caption="Roberts Edge Detection", use_container_width=True)
        
        elif edge_method == "Canny":
            st.markdown("**Canny**: Multi-stage edge detection dengan hysteresis thresholding")
            
            col_canny1, col_canny2 = st.columns(2)
            with col_canny1:
                low_threshold = st.slider("Low Threshold", 0, 200, 50)
            with col_canny2:
                high_threshold = st.slider("High Threshold", 50, 300, 150)
            
            canny_edges = cv2.Canny(img_gray_seg, low_threshold, high_threshold)
            
            col_canny_res1, col_canny_res2 = st.columns(2)
            with col_canny_res1:
                st.image(img_gray_seg, caption="Original", use_container_width=True)
            with col_canny_res2:
                st.image(canny_edges, caption="Canny Edges", use_container_width=True)
        
        elif edge_method == "LoG (Laplacian of Gaussian)":
            st.markdown("**LoG**: Kombinasi Gaussian smoothing dan Laplacian untuk deteksi tepi yang robust terhadap noise")
            
            sigma_log = st.slider("Sigma untuk Gaussian", 0.5, 5.0, 1.0, 0.5)
            
            log_edges = filters.laplace(filters.gaussian(img_gray_seg, sigma=sigma_log))
            log_edges = np.uint8(np.abs(log_edges) / np.abs(log_edges).max() * 255)
            
            st.image(log_edges, caption="LoG Edges", use_container_width=True)
    
    # Tab 2: Thresholding
    with seg_tabs[1]:
        st.markdown("#### 🎚️ Thresholding Methods")
        
        threshold_method = st.selectbox(
            "Pilih Metode Thresholding",
            ["Manual Threshold", "Otsu's Method", "Adaptive Thresholding"]
        )
        
        if threshold_method == "Manual Threshold":
            st.markdown("**Manual Threshold**: Pemilihan threshold secara manual")
            
            threshold_value = st.slider("Threshold Value", 0, 255, 127)
            _, binary = cv2.threshold(img_gray_seg, threshold_value, 255, cv2.THRESH_BINARY)
            
            col_thresh1, col_thresh2 = st.columns(2)
            with col_thresh1:
                st.image(img_gray_seg, caption="Original", use_container_width=True)
                if show_histogram:
                    fig_thresh = display_histogram(img_gray_seg, "Histogram")
                    # Add threshold line
                    plt.axvline(x=threshold_value, color='r', linestyle='--', label=f'Threshold={threshold_value}')
                    plt.legend()
                    st.pyplot(fig_thresh)
                    plt.close()
            
            with col_thresh2:
                st.image(binary, caption=f"Binary (T={threshold_value})", use_container_width=True)
        
        elif threshold_method == "Otsu's Method":
            st.markdown("**Otsu's Method**: Pemilihan threshold optimal dengan memaksimalkan variansi antar kelas")
            
            otsu_threshold, binary_otsu = cv2.threshold(img_gray_seg, 0, 255, 
                                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            st.info(f"Threshold optimal (Otsu): {otsu_threshold:.2f}")
            
            col_otsu1, col_otsu2 = st.columns(2)
            with col_otsu1:
                st.image(img_gray_seg, caption="Original", use_container_width=True)
                if show_histogram:
                    fig_otsu = display_histogram(img_gray_seg, "Histogram with Otsu Threshold")
                    plt.axvline(x=otsu_threshold, color='r', linestyle='--', 
                              label=f'Otsu Threshold={otsu_threshold:.2f}')
                    plt.legend()
                    st.pyplot(fig_otsu)
                    plt.close()
            
            with col_otsu2:
                st.image(binary_otsu, caption="Otsu Thresholding", use_container_width=True)
        
        elif threshold_method == "Adaptive Thresholding":
            st.markdown("**Adaptive Thresholding**: Threshold berbeda untuk setiap region, cocok untuk pencahayaan tidak merata")
            
            col_adapt1, col_adapt2 = st.columns(2)
            with col_adapt1:
                block_size = st.slider("Block Size", 3, 51, 11, 2)
            with col_adapt2:
                C = st.slider("Constant C", -10, 10, 2)
            
            method_type = st.radio(
                "Metode Adaptive",
                ["Mean", "Gaussian"],
                horizontal=True
            )
            
            if method_type == "Mean":
                adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
            else:
                adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            
            binary_adaptive = cv2.adaptiveThreshold(img_gray_seg, 255, adaptive_method,
                                                   cv2.THRESH_BINARY, block_size, C)
            
            col_adapt_res1, col_adapt_res2 = st.columns(2)
            with col_adapt_res1:
                st.image(img_gray_seg, caption="Original", use_container_width=True)
            with col_adapt_res2:
                st.image(binary_adaptive, caption=f"Adaptive {method_type}", use_container_width=True)
    
    # Tab 3: Region-Based Segmentation
    with seg_tabs[2]:
        st.markdown("#### 🗺️ Region-Based Segmentation")
        
        region_method = st.selectbox(
            "Pilih Metode Region-Based",
            ["SLIC Superpixel", "Watershed", "Region Growing (Simple)"]
        )
        
        if region_method == "SLIC Superpixel":
            st.markdown("**SLIC**: Simple Linear Iterative Clustering untuk superpixel segmentation")
            
            col_slic1, col_slic2 = st.columns(2)
            with col_slic1:
                n_segments = st.slider("Jumlah Segments", 50, 500, 200, 10)
            with col_slic2:
                compactness = st.slider("Compactness", 1, 50, 10)
            
            if is_color:
                img_slic = img_as_float(img_working)
            else:
                img_slic = img_as_float(cv2.cvtColor(img_gray_seg, cv2.COLOR_GRAY2RGB))
            
            segments_slic = segmentation.slic(img_slic, n_segments=n_segments, 
                                             compactness=compactness, start_label=1)
            
            viz_type = st.radio("Visualisasi", ["Average Color", "Boundaries"], horizontal=True)
            
            if viz_type == "Average Color":
                if is_color:
                    colored_slic = color.label2rgb(segments_slic, img_working, kind="avg")
                else:
                    colored_slic = color.label2rgb(segments_slic, 
                                                   cv2.cvtColor(img_gray_seg, cv2.COLOR_GRAY2RGB), 
                                                   kind="avg")
            else:
                if is_color:
                    colored_slic = segmentation.mark_boundaries(img_working, segments_slic)
                else:
                    colored_slic = segmentation.mark_boundaries(
                        cv2.cvtColor(img_gray_seg, cv2.COLOR_GRAY2RGB), segments_slic)
            
            colored_slic = (colored_slic * 255).astype(np.uint8) if colored_slic.dtype == np.float64 else colored_slic
            
            st.image(colored_slic, caption=f"SLIC Superpixels ({n_segments} segments)", 
                    use_container_width=True)
        
        elif region_method == "Watershed":
            st.markdown("**Watershed**: Segmentasi berbasis topologi untuk memisahkan objek yang berdekatan")
            
            # Preprocessing
            _, thresh_ws = cv2.threshold(img_gray_seg, 0, 255, 
                                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Noise removal
            kernel_ws = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh_ws, cv2.MORPH_OPEN, kernel_ws, iterations=2)
            
            # Sure background
            sure_bg = cv2.dilate(opening, kernel_ws, iterations=3)
            
            # Sure foreground
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
            
            # Unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            
            # Marker labelling
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            
            # Apply watershed
            if is_color:
                markers_ws = cv2.watershed(img_working, markers)
                img_watershed = img_working.copy()
            else:
                img_temp = cv2.cvtColor(img_gray_seg, cv2.COLOR_GRAY2RGB)
                markers_ws = cv2.watershed(img_temp, markers)
                img_watershed = img_temp.copy()
            
            img_watershed[markers_ws == -1] = [255, 0, 0]
            
            col_ws1, col_ws2, col_ws3 = st.columns(3)
            with col_ws1:
                st.image(dist_transform, caption="Distance Transform", 
                        clamp=True, use_container_width=True)
            with col_ws2:
                st.image(sure_fg, caption="Sure Foreground", use_container_width=True)
            with col_ws3:
                st.image(img_watershed, caption="Watershed Result", use_container_width=True)
        
        elif region_method == "Region Growing (Simple)":
            st.markdown("**Region Growing**: Segmentasi berbasis keseragaman intensitas dari seed point")
            
            st.info("Implementasi sederhana region growing dengan threshold intensitas")
            
            seed_threshold = st.slider("Threshold Perbedaan Intensitas", 1, 50, 10)
            
            # Simple region growing
            seed_point = (img_gray_seg.shape[0] // 2, img_gray_seg.shape[1] // 2)
            
            visited = np.zeros_like(img_gray_seg, dtype=bool)
            segmented = np.zeros_like(img_gray_seg)
            
            seed_value = img_gray_seg[seed_point]
            stack = [seed_point]
            
            while stack and len(stack) < 10000:  # Limit iterations
                current = stack.pop()
                
                if visited[current]:
                    continue
                
                visited[current] = True
                
                if abs(int(img_gray_seg[current]) - int(seed_value)) < seed_threshold:
                    segmented[current] = 255
                    
                    # Add neighbors
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = current[0] + dx, current[1] + dy
                        if (0 <= nx < img_gray_seg.shape[0] and 
                            0 <= ny < img_gray_seg.shape[1] and 
                            not visited[nx, ny]):
                            stack.append((nx, ny))
            
            col_rg1, col_rg2 = st.columns(2)
            with col_rg1:
                st.image(img_gray_seg, caption="Original", use_container_width=True)
            with col_rg2:
                st.image(segmented, caption="Region Growing Result", use_container_width=True)
    
    # Tab 4: Advanced Methods
    with seg_tabs[3]:
        st.markdown("#### 🚀 Advanced Segmentation Methods")
        
        advanced_method = st.selectbox(
            "Pilih Metode Advanced",
            ["GrabCut", "K-Means Clustering", "Morphological Operations"]
        )
        
        if advanced_method == "GrabCut":
            st.markdown("**GrabCut**: Interactive foreground extraction menggunakan graph cuts")
            
            if not is_color:
                st.warning("GrabCut membutuhkan citra berwarna. Menggunakan versi RGB dari citra.")
                img_grabcut = cv2.cvtColor(img_gray_seg, cv2.COLOR_GRAY2RGB)
            else:
                img_grabcut = img_working.copy()
            
            height, width = img_grabcut.shape[:2]
            
            col_gc1, col_gc2 = st.columns(2)
            with col_gc1:
                x_start = st.slider("X Start", 0, width - 50, width // 4)
                y_start = st.slider("Y Start", 0, height - 50, height // 4)
            with col_gc2:
                roi_width = st.slider("ROI Width", 50, width - x_start, width // 2)
                roi_height = st.slider("ROI Height", 50, height - y_start, height // 2)
            
            iterations = st.slider("Iterasi GrabCut", 1, 10, 5)
            
            if st.button("Jalankan GrabCut"):
                with st.spinner("Memproses GrabCut..."):
                    mask_gc = np.zeros(img_grabcut.shape[:2], np.uint8)
                    bgd_model = np.zeros((1, 65), np.float64)
                    fgd_model = np.zeros((1, 65), np.float64)
                    
                    rect = (x_start, y_start, roi_width, roi_height)
                    cv2.grabCut(img_grabcut, mask_gc, rect, bgd_model, fgd_model, 
                               iterations, cv2.GC_INIT_WITH_RECT)
                    
                    mask2 = np.where((mask_gc == 2) | (mask_gc == 0), 0, 1).astype('uint8')
                    result_gc = img_grabcut * mask2[:, :, np.newaxis]
                    
                    col_gc_res1, col_gc_res2, col_gc_res3 = st.columns(3)
                    
                    with col_gc_res1:
                        img_with_rect = img_grabcut.copy()
                        cv2.rectangle(img_with_rect, (x_start, y_start),
                                    (x_start + roi_width, y_start + roi_height), 
                                    (255, 0, 0), 2)
                        st.image(img_with_rect, caption="ROI Selection", use_container_width=True)
                    
                    with col_gc_res2:
                        st.image(mask2 * 255, caption="Mask", use_container_width=True)
                    
                    with col_gc_res3:
                        st.image(result_gc, caption="GrabCut Result", use_container_width=True)
        
        elif advanced_method == "K-Means Clustering":
            st.markdown("**K-Means**: Clustering piksel berdasarkan kesamaan warna/intensitas")
            
            k_clusters = st.slider("Jumlah Cluster (K)", 2, 10, 3)
            
            if st.button("Jalankan K-Means"):
                with st.spinner("Memproses K-Means..."):
                    if is_color:
                        pixel_values = img_working.reshape((-1, 3)).astype(np.float32)
                    else:
                        pixel_values = img_gray_seg.reshape((-1, 1)).astype(np.float32)
                    
                    # K-means
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                    _, labels, centers = cv2.kmeans(pixel_values, k_clusters, None, 
                                                    criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                    
                    centers = np.uint8(centers)
                    segmented_image = centers[labels.flatten()]
                    
                    if is_color:
                        segmented_image = segmented_image.reshape(img_working.shape)
                    else:
                        segmented_image = segmented_image.reshape(img_gray_seg.shape)
                    
                    col_km1, col_km2 = st.columns(2)
                    with col_km1:
                        if is_color:
                            st.image(img_working, caption="Original", use_container_width=True)
                        else:
                            st.image(img_gray_seg, caption="Original", use_container_width=True)
                    
                    with col_km2:
                        st.image(segmented_image, caption=f"K-Means (K={k_clusters})", 
                                use_container_width=True)
        
        elif advanced_method == "Morphological Operations":
            st.markdown("**Morphological Operations**: Operasi morfologi untuk pemrosesan bentuk objek")
            
            morph_op = st.selectbox(
                "Pilih Operasi Morfologi",
                ["Erosion", "Dilation", "Opening", "Closing", "Gradient", "Top Hat", "Black Hat"]
            )
            
            kernel_size_morph = st.slider("Ukuran Kernel", 3, 15, 5, 2)
            iterations_morph = st.slider("Iterasi", 1, 5, 1)
            
            kernel_morph = np.ones((kernel_size_morph, kernel_size_morph), np.uint8)
            
            # Apply morphological operation
            if morph_op == "Erosion":
                result_morph = cv2.erode(img_gray_seg, kernel_morph, iterations=iterations_morph)
            elif morph_op == "Dilation":
                result_morph = cv2.dilate(img_gray_seg, kernel_morph, iterations=iterations_morph)
            elif morph_op == "Opening":
                result_morph = cv2.morphologyEx(img_gray_seg, cv2.MORPH_OPEN, kernel_morph)
            elif morph_op == "Closing":
                result_morph = cv2.morphologyEx(img_gray_seg, cv2.MORPH_CLOSE, kernel_morph)
            elif morph_op == "Gradient":
                result_morph = cv2.morphologyEx(img_gray_seg, cv2.MORPH_GRADIENT, kernel_morph)
            elif morph_op == "Top Hat":
                result_morph = cv2.morphologyEx(img_gray_seg, cv2.MORPH_TOPHAT, kernel_morph)
            elif morph_op == "Black Hat":
                result_morph = cv2.morphologyEx(img_gray_seg, cv2.MORPH_BLACKHAT, kernel_morph)
            
            col_morph1, col_morph2 = st.columns(2)
            with col_morph1:
                st.image(img_gray_seg, caption="Original", use_container_width=True)
            with col_morph2:
                st.image(result_morph, caption=f"{morph_op} Result", use_container_width=True)
    
    # =====================================================
    # HASIL PIPELINE DAN DOWNLOAD
    # =====================================================
    st.markdown('<div class="module-header">💾 HASIL AKHIR DAN DOWNLOAD</div>', unsafe_allow_html=True)
    
    st.success("✅ Pipeline processing selesai!")
    
    col_final1, col_final2 = st.columns(2)
    
    with col_final1:
        st.markdown("**Citra Asli**")
        if is_color:
            st.image(img_rgb, caption="Original RGB", use_container_width=True)
        else:
            st.image(img_gray, caption="Original Grayscale", use_container_width=True)
    
    with col_final2:
        st.markdown("**Citra Hasil Processing**")
        st.image(img_working, caption="Processed Image", use_container_width=True)
    
    # Download buttons
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    
    with col_dl1:
        st.download_button(
            label="⬇️ Download Hasil",
            data=create_download_link(img_working, "processed", download_format),
            file_name=f"hasil_processing.{download_format.lower()}",
            mime=f"image/{download_format.lower()}",
            use_container_width=True
        )
    
    with col_dl2:
        if is_color:
            st.download_button(
                label="⬇️ Download Original",
                data=create_download_link(img_rgb, "original", download_format),
                file_name=f"original.{download_format.lower()}",
                mime=f"image/{download_format.lower()}",
                use_container_width=True
            )
        else:
            st.download_button(
                label="⬇️ Download Original",
                data=create_download_link(img_gray, "original", download_format),
                file_name=f"original.{download_format.lower()}",
                mime=f"image/{download_format.lower()}",
                use_container_width=True
            )
    
    with col_dl3:
        if st.button("🔄 Reset Pipeline", use_container_width=True):
            st.rerun()

else:
    # Welcome screen
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h2>🎓 Selamat Datang di Aplikasi Image Processing ITS</h2>
        <p style='font-size: 18px; margin-top: 20px;'>
            Aplikasi interaktif untuk pembelajaran dan eksperimen pengolahan citra digital
        </p>
        <p style='color: #666; margin-top: 15px;'>
            📁 Silakan upload citra (PNG/JPG/BMP) untuk memulai
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature overview
    st.markdown("### 🎯 Fitur-Fitur Utama")
    
    col_feat1, col_feat2 = st.columns(2)
    
    with col_feat1:
        st.markdown("""
        #### Modul 1: Enhancement
        - Transformasi Intensitas (Gamma, Log)
        - Histogram Equalization & CLAHE
        - Filtering Spasial (Smoothing, Sharpening)
        
        #### Modul 2: FFT Filtering
        - FFT Visualization
        - Low-Pass & High-Pass Filters
        - Homomorphic Filtering
        """)
    
    with col_feat2:
        st.markdown("""
        #### Modul 3: Color Processing
        - Transformasi Color Space (HSV, HSI, LAB)
        - Channel Extraction
        - Pseudocolor
        
        #### Modul 4: Segmentation
        - Edge Detection (Sobel, Canny, LoG)
        - Thresholding (Manual, Otsu, Adaptive)
        - Region-Based (SLIC, Watershed, GrabCut)
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Interactive Digital Image Processing Application</strong></p>
    <p>Fadaukas Daffa Tajuddin • Davin Amadeo Wijaya • Reihan Arianza</p>
    <p>Institut Teknologi Sepuluh Nopember Surabaya - 2025</p>
    <p style='font-size: 12px; margin-top: 10px;'>
        Powered by OpenCV, scikit-image, NumPy, and Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
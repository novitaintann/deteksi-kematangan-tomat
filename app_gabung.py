import streamlit as st
import numpy as np
import cv2
import joblib
import tensorflow as tf
from PIL import Image
import os
import pandas as pd

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Tomato Maturity Analytics - AI Battle", 
    page_icon="🍅", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS UNTUK MEPERCANTIK UI ---
st.markdown("""
    <style>
    .main-title {
        font-size: 2.6rem !important;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem !important;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True) # <--- Sudah diperbaiki menjadi unsafe_allow_html

# --- HEADER APP ---
st.markdown('<div class="main-title">🍅 Tomato Maturity Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Evaluasi Komparatif Model Machine Learning (Random Forest) vs. Deep Learning (CNN MobileNetV2)</div>', unsafe_allow_html=True)

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.image("https://www.flaticon.com/free-icon/tomato_1790387?term=tomato&page=1&position=3&origin=search&related_id=1790387", width=80) # Icon pemanis
    st.title("Control Panel")
    st.write("Atur parameter pengujian model di bawah ini:")
    
    rasio = st.selectbox(
        "Variasi Data Training (Rasio):", 
        ["80:20", "75:25", "65:35", "60:40"],
        help="Pilih model berdasarkan rasio pembagian data yang digunakan saat training."
    )
    format_rasio = rasio.replace(":", "_")
    
    st.divider()
    st.markdown("### 📊 Metadata Model")
    st.caption(f"**RF Target:** `model_tomat_{format_rasio}.pkl`")
    st.caption(f"**CNN Target:** `model_cnn_tomat_{format_rasio}.keras`")

# Tentukan nama file
RF_FILE = f"model_tomat_{format_rasio}.pkl"
CNN_FILE = f"model_cnn_tomat_{format_rasio}.keras"

# --- CACHING MODEL LOADING ---
@st.cache_resource
def load_models(rf_path, cnn_path):
    rf_model = joblib.load(rf_path) if os.path.exists(rf_path) else None
    cnn_model = tf.keras.models.load_model(cnn_path) if os.path.exists(cnn_path) else None
    return rf_model, cnn_model

rf_model, cnn_model = load_models(RF_FILE, CNN_FILE)

# Validasi Keberadaan Model
if rf_model is None or cnn_model is None:
    st.error(f"⚠️ **Berkas Model Tidak Lengkap:** Ekspektasi file `{RF_FILE}` dan `{CNN_FILE}` di dalam direktori root aktif.")
    st.stop()

# Label Mapping
label_map_rf = {0: "Matang", 1: "Mentah", 2: "Setengah Matang"}  # Sesuaikan dengan encoding RF-mu jika ia outputnya angka
inv_label_map = {0: "Matang", 1: "Mentah", 2: "Setengah Matang"}

# --- INITIALIZE SESSION STATE ---
if 'prediction_triggered' not in st.session_state:
    st.session_state.prediction_triggered = False

# --- LAYOUT UTAMA ---
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.markdown("### 📸 Input Data")
    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "Unggah citra buah tomat untuk dianalisis", 
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Preprocessing alpha channel jika ada
            image = Image.open(uploaded_file)
            if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3]) 
                image = background
            else:
                image = image.convert('RGB')
                
            st.image(image, caption='Citra Input (Query Image)', use_container_width=True)
            
            # Reset state jika user upload gambar baru
            if st.button("🔍 Jalankan Analisis Komparatif", type="primary", use_container_width=True):
                st.session_state.prediction_triggered = True

with col_right:
    st.markdown("### 🔬 Hasil Komparasi Model")
    
    if uploaded_file is not None and st.session_state.prediction_triggered:
        
        # --- PROCESSING & INFERENCE ---
        with st.spinner("Melakukan inferensi pada kedua arsitektur model..."):
            # 1. Pipeline Random Forest
            img_array = np.array(image) 
            img_resized_rf = cv2.resize(img_array, (64, 64))
            img_input_rf = img_resized_rf.flatten().reshape(1, -1)
            raw_rf = rf_model.predict(img_input_rf)[0]
            # Handle jika output RF berupa integer atau string langsung
            hasil_rf = label_map_rf[raw_rf] if isinstance(raw_rf, (int, np.integer)) else raw_rf

            # 2. Pipeline CNN MobileNetV2
            img_resized_cnn = image.resize((224, 224))
            img_array_cnn = np.array(img_resized_cnn)
            img_normalized = (img_array_cnn.astype(np.float32) / 127.5) - 1.0
            img_input_cnn = np.expand_dims(img_normalized, axis=0)
            
            prediksi_cnn = cnn_model.predict(img_input_cnn)[0]
            idx_cnn = np.argmax(prediksi_cnn)
            hasil_cnn = inv_label_map[idx_cnn]
            akurasi_cnn = prediksi_cnn[idx_cnn] * 100

        # --- DISPLAY VISUALISASI ---
        res_rf, res_cnn = st.columns(2)
        
        with res_rf:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("#### 🌲 Random Forest")
            st.caption("Pendekatan Analisis Piksel Flat (64x64)")
            
            if hasil_rf == "Matang":
                st.success("🔴 **MATANG**")
            elif hasil_rf == "Setengah Matang":
                st.warning("🟠 **SETENGAH MATANG**")
            else:
                st.error("🟢 **MENTAH**")
            
            st.info("💡 *Karakteristik:* Sangat sensitif terhadap nilai rata-rata warna global, namun mengabaikan struktur spasial objek.")
            st.markdown("</div>", unsafe_allow_html=True)

        with res_cnn:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("#### 🧠 CNN MobileNetV2")
            st.caption("Pendekatan Ekstraksi Fitur Hierarkis (224x224)")
            
            if hasil_cnn == "Matang":
                st.success(f"🔴 **MATANG** ({akurasi_cnn:.1f}%)")
            elif hasil_cnn == "Setengah Matang":
                st.warning(f"🟠 **SETENGAH MATANG** ({akurasi_cnn:.1f}%)")
            else:
                st.error(f"🟢 **MENTAH** ({akurasi_cnn:.1f}%)")
                
            st.info("💡 *Karakteristik:* Mampu mengekstrak fitur tepi, tekstur, dan bentuk secara mendalam serta mengabaikan noise latar belakang.")
            st.markdown("</div>", unsafe_allow_html=True)
            
        # --- TAMBAHAN PORTFOLIO VALUE: PROBABILITY DISTRIBUTION BAR CHART ---
        st.markdown("#### 📊 Distribusi Probabilitas Prediksi (CNN MobileNetV2)")
        df_prob = pd.DataFrame({
            'Kelas': [inv_label_map[0], inv_label_map[1], inv_label_map[2]],
            'Probabilitas (%)': [prediksi_cnn[0]*100, prediksi_cnn[1]*100, prediksi_cnn[2]*100]
        })
        st.bar_chart(data=df_prob, x='Kelas', y='Probabilitas (%)', use_container_width=True)

    else:
        # State saat aplikasi baru dibuka / belum ada aksi analisis
        st.info("Silakan unggah gambar di panel kiri dan klik tombol analisis untuk melihat perbandingan performa model.")
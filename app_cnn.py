import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

st.set_page_config(page_title="Deteksi Tomat AI", page_icon="🍅", layout="centered")

st.title("🍅 Deteksi Kematangan Tomat AI (CNN)")
st.markdown("Unggah foto tomat, dan model **Deep Learning (MobileNetV2)** akan mendeteksi tingkat kematangannya secara akurat, bahkan dengan latar belakang yang kompleks!")
st.divider()

# Sidebar
st.sidebar.header("⚙️ Pengaturan Model")
opsi_model = {
    "CNN Rasio 80:20": "model_cnn_tomat_80_20.keras",
    "CNN Rasio 75:25": "model_cnn_tomat_75_25.keras",
    "CNN Rasio 65:35": "model_cnn_tomat_65_35.keras",
    "CNN Rasio 60:40": "model_cnn_tomat_60_40.keras"
}

pilihan = st.sidebar.selectbox("Variasi Model:", list(opsi_model.keys()))
MODEL_FILE = opsi_model[pilihan]

# Mapping label kembali ke teks
inv_label_map = {0: "Matang", 1: "Mentah", 2: "Setengah Matang"}

@st.cache_resource
def load_cnn_model(file_path):
    if os.path.exists(file_path):
        return tf.keras.models.load_model(file_path)
    return None

model = load_cnn_model(MODEL_FILE)

if model is None:
    st.error(f"⚠️ Model '{MODEL_FILE}' belum ada. Silakan jalankan 'cnn.py' terlebih dahulu!")
    st.stop()
else:
    st.sidebar.success(f"✅ Model aktif: {pilihan}")

# Input Gambar
st.subheader("Unggah Foto Tomat")
uploaded_file = st.file_uploader("Format yang didukung: JPG, JPEG, PNG", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file)
        
        # Tangani PNG transparan jadi background putih (Biar nggak error background hitam)
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3]) 
            image = background
        else:
            image = image.convert('RGB')
            
        st.image(image, caption='Gambar yang diunggah', use_container_width=True)

    with col2:
        st.write("**Status:**")
        
        # PREPROCESSING KHUSUS CNN 
        IMG_SIZE = 224
        img_resized = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img_resized)
        
        # Normalisasi MobileNetV2 (-1 sampai 1)
        img_normalized = (img_array.astype(np.float32) / 127.5) - 1.0
        
        # Tambahkan dimensi batch (karena CNN butuh format [batch, tinggi, lebar, channel])
        img_input = np.expand_dims(img_normalized, axis=0)

        if st.button("🔍 Deteksi dengan CNN", type="primary", use_container_width=True):
            with st.spinner("Menganalisis fitur visual dengan AI..."):
                # Prediksi
                prediksi = model.predict(img_input)
                
                # Ambil index nilai tertinggi (0, 1, atau 2)
                predicted_class_idx = np.argmax(prediksi[0])
                hasil = inv_label_map[predicted_class_idx]
                
                # Menghitung persentase keyakinan AI
                akurasi_tebakan = np.max(prediksi[0]) * 100

                st.markdown("### Hasil Prediksi:")
                if hasil == "Matang":
                    st.success(f"🍅 **MATANG** ({akurasi_tebakan:.1f}%)")
                    st.info("Tomat ini sudah siap untuk dipanen atau dikonsumsi!")
                elif hasil == "Setengah Matang":
                    st.warning(f"🟠 **SETENGAH MATANG** ({akurasi_tebakan:.1f}%)")
                    st.info("Beri waktu beberapa hari lagi agar matang sempurna.")
                else:
                    st.error(f"🟢 **MENTAH** ({akurasi_tebakan:.1f}%)")
                    st.info("Tomat masih mentah, biarkan di pohon atau peram terlebih dahulu.")
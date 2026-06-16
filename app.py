import streamlit as st
import numpy as np
import cv2
import joblib
import os
import subprocess
from PIL import Image

# 1. Konfigurasi Halaman (Biar UI lebih rapi dan ada iconnya)
st.set_page_config(page_title="Deteksi Tomat AI", page_icon="🍅", layout="centered")

# 2. Judul dan Deskripsi UI
st.title("🍅 Deteksi Kematangan Tomat AI")
st.markdown("""
Aplikasi ini menggunakan algoritma **Random Forest** untuk mendeteksi kematangan tomat. 
Silakan unggah foto tomat, dan AI akan menentukan apakah tomat tersebut **Matang**, **Setengah Matang**, atau **Mentah**.
""")
st.divider() # Garis pembatas biar rapi

# 3. Sidebar untuk Pengaturan Model
st.sidebar.header("⚙️ Pengaturan Model")
st.sidebar.write("Pilih rasio dataset yang digunakan untuk melatih model:")

opsi_model = {
    "Rasio 80:20": "model_tomat_80_20.pkl",
    "Rasio 75:25": "model_tomat_75_25.pkl",
    "Rasio 65:35": "model_tomat_65_35.pkl",
    "Rasio 60:40": "model_tomat_60_40.pkl"
}

pilihan = st.sidebar.selectbox("Variasi Model:", list(opsi_model.keys()))
MODEL_FILE = opsi_model[pilihan]

# 4. Logika Pengecekan dan Pembuatan Model Otomatis (Fitur dari dosen!)
if not os.path.exists(MODEL_FILE):
    st.warning(f"⚠️ Model '{MODEL_FILE}' belum ada. Memulai pelatihan model secara otomatis...")
    with st.spinner("Menjalankan model.py... Harap tunggu sekitar 1-2 menit ⏳"):
        try:
            # Pastikan nama file training kamu adalah 'model.py'
            subprocess.run(["python", "model.py"], check=True) 
            st.success("Pelatihan selesai! Silakan muat ulang (refresh) halaman.")
            st.rerun() # Otomatis refresh UI setelah model selesai dibuat
        except subprocess.CalledProcessError:
            st.error("Gagal menjalankan model.py. Pastikan file tersebut ada dan tidak ada error di dalamnya.")
            st.stop()

# 5. Memuat Model
@st.cache_resource
def load_model(file_path):
    return joblib.load(file_path)

try:
    model = load_model(MODEL_FILE)
    st.sidebar.success(f"✅ Model aktif: {pilihan}")
except Exception as e:
    st.error("Gagal memuat model. Pastikan file tidak korup.")
    st.stop()

# 6. Komponen Input: Upload Gambar
st.subheader("Unggah Foto Tomat")
uploaded_file = st.file_uploader("Format yang didukung: JPG, JPEG, PNG", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membagi layout menjadi 2 kolom: Kiri untuk gambar, Kanan untuk hasil
    col1, col2 = st.columns(2)
    
    with col1:
        # Tampilkan gambar yang diupload
        image = Image.open(uploaded_file)
        # Memastikan gambar selalu format RGB (menghindari error jika upload PNG transparan/RGBA)
        image = image.convert('RGB')
        st.image(image, caption='Gambar yang diunggah', use_container_width=True)

    with col2:
        st.write("**Status:**")
        # 7. Preprocessing Gambar (SANGAT KRUSIAL)
        img_array = np.array(image) # Image PIL sudah dalam format RGB
        
        # Resize ke 64x64 piksel (TIDAK PERLU DIJADIKAN GRAYSCALE LAGI)
        img_resized = cv2.resize(img_array, (64, 64))

        # Ratakan (flatten) jadi array 1 Dimensi
        img_flattened = img_resized.flatten().reshape(1, -1)

        # 8. Lakukan Prediksi
        if st.button("🔍 Deteksi Kematangan", type="primary", use_container_width=True):
            with st.spinner("Menganalisis warna dan tekstur..."):
                prediksi = model.predict(img_flattened)
                hasil = prediksi[0]

                # 9. UI Hasil yang Menarik
                st.markdown("### Hasil Prediksi:")
                
                if hasil == "Matang":
                    st.success("🍅 **MATANG**")
                    st.info("Tomat ini sudah siap untuk dipanen atau dikonsumsi!")
                elif hasil == "Setengah Matang":
                    st.warning("🟠 **SETENGAH MATANG**")
                    st.info("Beri waktu beberapa hari lagi agar matang sempurna.")
                else:
                    st.error("🟢 **MENTAH**")
                    st.info("Tomat masih mentah, biarkan di pohon atau peram terlebih dahulu.")
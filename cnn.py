import pandas as pd
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# --- KONFIGURASI ---
BASE_DIR = "dataset"
IMG_SIZE = 224 # Ukuran standar untuk MobileNetV2

# Daftar CSV rasio yang sudah kamu buat
daftar_csv = [
    "tomat_detection_80_20.csv",
    "tomat_detection_75_25.csv",
    "tomat_detection_60_40.csv",
    "tomat_detection_65_35.csv"
]

# Mapping label teks ke angka untuk CNN
label_map = {"Matang": 0, "Mentah": 1, "Setengah Matang": 2}

print("\n" + "="*40)
print("MULAI PELATIHAN CNN (MOBILENET-V2) TOMAT")
print("="*40)

ringkasan_hasil = [] # Variabel untuk menampung hasil

for csv_file in daftar_csv:
    if not os.path.exists(csv_file):
        print(f"File {csv_file} tidak ditemukan, lewati...")
        continue
        
    print(f"\n[1] Memproses Data: {csv_file}...")
    df = pd.read_csv(csv_file)
    
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading Images"):
        img_path = os.path.join(BASE_DIR, row['file'])
        
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                # Resize ke ukuran MobileNetV2
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                
                # PENTING: Konversi BGR ke RGB agar tidak buta warna tomat!
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Normalisasi ke rentang -1 sampai 1 (standar MobileNetV2)
                img_normalized = (img.astype(np.float32) / 127.5) - 1.0
                
                label = label_map[row['label']]
                
                # Pisahkan berdasarkan kolom 'split' dari CSV
                if row['split'] == 'train':
                    X_train.append(img_normalized)
                    y_train.append(label)
                elif row['split'] == 'test':
                    X_test.append(img_normalized)
                    y_test.append(label)

    # Konversi ke array Numpy
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    print(f"\n[2] Membangun Model untuk {csv_file}...")
    # Load MobileNetV2 tanpa lapisan atas (Top)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False # "Membekukan" otak model dasar agar tidak rusak
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax') # 3 Output untuk klasifikasi kematangan
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy', # Khusus untuk klasifikasi kategori
        metrics=['accuracy'] # Pakai Akurasi, bukan MSE/R2
    )
    
    print(f"\n[3] Melatih Model (Harap tunggu, proses CNN cukup berat)...")
    history = model.fit(
        X_train, y_train,
        epochs=10, # Diset 10 Epoch agar tidak terlalu lama
        batch_size=32,
        validation_data=(X_test, y_test)
    )
    
    print("\n[4] Mengevaluasi & Menyimpan Model...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    
    # Menyimpan model (contoh nama: model_cnn_tomat_80_20.keras)
    nama_rasio = csv_file.replace('tomat_detection_', '').replace('.csv', '')
    model_save_name = f"model_cnn_tomat_{nama_rasio}.keras"
    
    model.save(model_save_name)
    print(f"✅ Selesai! Akurasi: {acc:.4f} | Disimpan: {model_save_name}")
    
    # Masukkan hasil ke dalam list
    ringkasan_hasil.append({"Rasio": nama_rasio, "Akurasi": acc})

print("\n" + "="*40)
print("📊 RINGKASAN AKURASI CNN TOMAT")
print("="*40)
for hasil in ringkasan_hasil:
    print(f"Model Rasio {hasil['Rasio'].replace('_', ':')} -> Akurasi: {hasil['Akurasi']*100:.2f}%")
print("="*40)
print("🚀 SEMUA MODEL CNN BERHASIL DIBUAT!")
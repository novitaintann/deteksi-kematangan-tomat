import pandas as pd
import cv2
import numpy as np
import os
import warnings
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

warnings.filterwarnings("ignore")

BASE_DIR = "dataset"

daftar_csv = [
    "tomat_detection_80_20.csv",
    "tomat_detection_75_25.csv",
    "tomat_detection_60_40.csv",
    "tomat_detection_65_35.csv"
]

print("\n" + "="*40)
print("MULAI PELATIHAN 4 RASIO (RANDOM FOREST)")
print("="*40)

ringkasan_rf = [] # Variabel untuk menampung hasil RF
for csv_file in daftar_csv:
    if not os.path.exists(csv_file):
        print(f"File {csv_file} tidak ditemukan, lewati...")
        continue
        
    print(f"\nMemproses {csv_file}...")
    df = pd.read_csv(csv_file)
    
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    # Membaca gambar berdasarkan CSV
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading Images"):
        img_path = os.path.join(BASE_DIR, row['file'])
        
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                # Resize dan konversi ke RGB
                img = cv2.resize(img, (64, 64))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                features = img.flatten()
                
                # Pemisahan berdasarkan kolom 'split' dari CSV
                if row['split'] == 'train':
                    X_train.append(features)
                    y_train.append(row['label'])
                elif row['split'] == 'test':
                    X_test.append(features)
                    y_test.append(row['label'])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    # Melatih Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluasi Akurasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Simpan Model (.pkl)
    # Nama file CSV contohnya: tomat_detection_80_20.csv -> ambil '80_20'
    nama_rasio = csv_file.replace('tomat_detection_', '').replace('.csv', '')
    nama_file_model = f"model_tomat_{nama_rasio}.pkl"
    
    joblib.dump(model, nama_file_model)
    print(f"Akurasi: {acc:.2f} | Disimpan: {nama_file_model}")
    
    # Masukkan hasil ke dalam list
    ringkasan_rf.append({"Rasio": nama_rasio, "Akurasi": acc})
    
    joblib.dump(model, nama_file_model)
    print(f"Akurasi: {acc:.2f} | Disimpan: {nama_file_model}")

print("\n" + "="*40)
print("📊 RINGKASAN AKURASI RANDOM FOREST TOMAT")
print("="*40)
for hasil in ringkasan_rf:
    print(f"Model Rasio {hasil['Rasio'].replace('_', ':')} -> Akurasi: {hasil['Akurasi']*100:.2f}%")
print("="*40)
print("✅ Semua proses selesai! Model .pkl siap digunakan di UI.")
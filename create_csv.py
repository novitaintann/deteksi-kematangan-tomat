import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = "dataset"
kategori = ['Matang', 'Mentah', 'Setengah Matang']

# 1. Mengumpulkan semua file gambar dan labelnya
all_files = []
all_labels = []

print("Mendata file gambar dari folder utama...")
for label in kategori:
    folder_path = os.path.join(BASE_DIR, label)
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Format yang disimpan: Matang/nama_gambar.jpg
                all_files.append(f"{label}/{file}") 
                all_labels.append(label)

print(f"Total gambar ditemukan: {len(all_files)}\n")

# 2. Buat csv untuk berbagai rasio eksperimen
rasio_eksperimen = [
    (0.20, "80_20"),
    (0.25, "75_25"),
    (0.40, "60_40"),
    (0.35, "65_35")
]

for test_size, nama in rasio_eksperimen:
    # Membagi status train dan test
    X_train, X_test, y_train, y_test = train_test_split(
        all_files, all_labels, 
        test_size=test_size, 
        random_state=42, 
        stratify=all_labels # Biar seimbang
    )
    
    # Siapkan data untuk csv
    data_csv = []
    for file, label in zip(X_train, y_train):
        data_csv.append([file, label, 'train'])
    for file, label in zip(X_test, y_test):
        data_csv.append([file, label, 'test'])
        
    # Ubah ke DataFrame dan simpan
    df = pd.DataFrame(data_csv, columns=['file', 'label', 'split'])
    csv_name = f'tomat_detection_{nama}.csv'
    df.to_csv(csv_name, index=False)
    
    print(f"✅ Berhasil membuat {csv_name} (Train: {len(X_train)}, Test: {len(X_test)})")
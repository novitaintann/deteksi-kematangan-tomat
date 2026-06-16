# GABUNG FOLDER 1 DAN 2 DI SETENGAH MATANG

import os
import shutil

# Menentukan path ke folder Setengah Matang dan sub-foldernya
target_dir = 'dataset/Setengah Matang'
sub_folders = ['1', '2']

for sub in sub_folders:
    sub_path = os.path.join(target_dir, sub)
    
    # Cek apakah folder 1 atau 2 ada
    if os.path.exists(sub_path):
        for filename in os.listdir(sub_path):
            
            # Awalan unik di nama file untuk menghindari duplikasi
            new_name = f"folder{sub}_{filename}"
            
            old_file_path = os.path.join(sub_path, filename)
            new_file_path = os.path.join(target_dir, new_name)
            
            # Pindahkan file ke folder induk
            shutil.move(old_file_path, new_file_path)
        
        # Hapus sub-folder yang sudah kosong
        os.rmdir(sub_path)

print("Folder Setengah Matang sekarang sudah setara!")
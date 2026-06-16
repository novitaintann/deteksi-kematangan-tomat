import splitfolders

input_folder = "dataset"
folder_name = "tomat_80_20"

print("Sedang memproses pembagian data 80:20...")

splitfolders.ratio(
    input_folder, 
    output=folder_name, 
    seed=42, 
    ratio=(0.80, 0.20), 
    group_prefix=None
)

print("✅ Pembagian dataset selesai! Silakan cek folder tomat_80_20")
import re
import os
import matplotlib
matplotlib.use("Agg")  # Mode non-interaktif agar aman dijalankan otomatis
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================================
# === [1] KONFIGURASI STOPWORDS
# ==========================================================
# Stopwords adalah kata-kata umum yang biasanya tidak memiliki makna penting
# dalam proses pencarian atau analisis teks (contoh: "dan", "atau", "di", "ke").
# Daftar berikut berisi kata-kata yang akan dihapus selama preprocessing.
STOPWORDS = set([
    "judul", "dan", "atau", "serta",
    "di", "ke", "dari", "pada", "dalam", "antara",
    "ini", "itu", "tersebut",
    "adalah", "sebagai", "untuk", "sebuah", "seorang",
    "juga", "lebih", "tidak", "bukan", "saat", "hingga",
    "adanya", "agar", "karenanya", "sehingga", "per"
])


# ==========================================================
# === [2] PENENTUAN PATH FOLDER INPUT & OUTPUT
# ==========================================================
# Folder input  : data/raw         → berisi dokumen mentah
# Folder output : data/processed   → berisi hasil tokenisasi dan pembersihan
RAW_PATH = Path("data/raw")
PROCESSED_PATH = Path("data/processed")

# Pastikan folder output sudah ada (buat otomatis jika belum)
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)


# ==========================================================
# === [3] FUNGSI PREPROCESSING TEKS
# ==========================================================
# Fungsi ini melakukan pembersihan dan tokenisasi teks:
# - Case folding (huruf kecil semua)
# - Menghapus angka, tanda baca, dan simbol
# - Memisahkan kata berdasarkan spasi
# - Menghapus stopwords
def preprocess_text(text: str):
    # Case folding → ubah semua huruf jadi huruf kecil
    text = text.lower()

    # Hapus angka, tanda baca, dan simbol
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenisasi → pisahkan berdasarkan spasi
    tokens = text.split()

    # Hapus stopwords dan token kosong
    tokens = [t for t in tokens if t and t not in STOPWORDS]

    return tokens


# ==========================================================
# === [4] FUNGSI UTAMA UNTUK MEMPROSES SEMUA FILE
# ==========================================================
# Fungsi ini akan membaca semua file .txt di folder `data/raw`,
# lalu menjalankan fungsi `preprocess_text()` untuk setiap dokumen.
# Hasilnya disimpan ke folder `data/processed` dan dibuat grafik distribusi panjang dokumen.
def process_all_files():
    files = list(RAW_PATH.glob("*.txt"))
    if not files:
        print("Folder data/raw kosong atau tidak ditemukan.")
        return

    doc_lengths = {}

    # Proses setiap file
    for file in files:
        raw_text = file.read_text(encoding="utf-8", errors="ignore")
        tokens = preprocess_text(raw_text)
        doc_lengths[file.name] = len(tokens)

        # Simpan hasil tokenisasi ke folder processed
        out_path = PROCESSED_PATH / file.name
        out_path.write_text(" ".join(tokens), encoding="utf-8")

    # ==========================================================
    # === [5] TAMPILKAN RINGKASAN HASIL PREPROCESSING
    # ==========================================================
    print(f"{len(files)} dokumen berhasil diproses & disimpan di '{PROCESSED_PATH}/'")
    print("Distribusi panjang dokumen (jumlah token):")
    for name, n_tok in doc_lengths.items():
        print(f"  {name:<25} : {n_tok:>5} token")

    # ==========================================================
    # === [6] VISUALISASI DISTRIBUSI PANJANG DOKUMEN
    # ==========================================================
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(doc_lengths)), list(doc_lengths.values()), tick_label=list(doc_lengths.keys()))
    plt.xticks(rotation=45, ha="right")
    plt.title("Distribusi Panjang Dokumen (Setelah Preprocessing)")
    plt.ylabel("Jumlah Token")
    plt.tight_layout()

    # Simpan grafik ke file .png
    out_plot = PROCESSED_PATH / "distribusi_dokumen.png"
    plt.savefig(out_plot)
    print(f"Grafik disimpan ke {out_plot}")


# ==========================================================
# === [7] MAIN RUNNER
# ==========================================================
# Bagian ini akan otomatis dijalankan jika file dipanggil langsung
# (misalnya lewat terminal: `python src/preprocess.py`)
if __name__ == "__main__":
    process_all_files()

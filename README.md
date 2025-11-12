# 🧠 Mini Search Engine – Sistem Temu Kembali Informasi (STKI)

Proyek ini merupakan implementasi **Sistem Temu Kembali Informasi (STKI)** menggunakan model **Boolean Retrieval** dan **Vector Space Model (VSM)**.
Sistem dikembangkan menggunakan Python dan Streamlit, dengan 15 dokumen teks universitas di Indonesia sebagai korpus.

---

## 📚 1. Deskripsi Singkat

Proyek ini terdiri atas dua bagian utama:

1. **Modul Backend (src/)**
   Berisi seluruh fungsi utama sistem — mulai dari *preprocessing*, *indexing*, *retrieval*, hingga *evaluasi*.
2. **Aplikasi Frontend (app/)**
   Berisi antarmuka pengguna berbasis **Streamlit**, yang memungkinkan pencarian dan visualisasi hasil melalui web.

---

## 🗂️ 2. Struktur Folder dan File

```
stki-uts-<nim>-<nama>/ 
├─ data/ 
│  ├─ processed  # orchestrator + CLI/
│  └─ rar/               
├─ src/ 
# .txt/.pdf yang Anda konversi ke .txt 
│  ├─ preprocess.py 
│  ├─ boolean_ir.py 
│  ├─ vsm_ir.py 
│  ├─ search_engine.py  # orchestrator + CLI 
│  └─ eval.py 
├─ app/ 
│  └─ main.py
├─ notebooks/ 
        # 
contoh : Interface (CLI/UI) 
│  └─ UTS_STKI_<nim>.ipynb 
├─ reports/ 
│  ├─ laporan.pdf       
│  └─ readme.md         
└─ requirements.txt 

```

---

## ⚙️ 3. Penjelasan Fungsi Setiap File

| File                   | Fungsi Utama                                                                                                                                  |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **`preprocess.py`**    | Melakukan *text preprocessing* (case folding, tokenisasi, stopword removal, stemming). Menyimpan hasil ke `data/processed/`.                  |
| **`boolean_ir.py`**    | Membangun *inverted index* dan mendukung operasi logika: `AND`, `OR`, `NOT`. Menghitung *precision* & *recall* tiap query.                    |
| **`vsm_ir.py`**        | Mengimplementasikan model *Vector Space* dengan pembobotan **TF-IDF** dan **TF-IDF Sublinear**. Menghitung *cosine similarity* antar vektor.  |
| **`eval.py`**          | Berisi fungsi evaluasi: `precision`, `recall`, `F1-score`, `MAP@k`, dan `nDCG@k`.                                                             |
| **`search_engine.py`** | Mengintegrasikan Boolean & VSM dalam satu program. Menampilkan hasil, peringkat, dan evaluasi di terminal.                                    |
| **`app/main.py`**      | Menjadi **index utama (entry point)** untuk *deployment* di Streamlit. Menyediakan antarmuka pencarian, tabel hasil, dan evaluasi interaktif. |

---

## 🚀 4. Cara Menjalankan Proyek

### 💻 A. Menjalankan di Terminal

1. Pastikan sudah membuat *virtual environment* dan menginstall dependensi:

   ```bash
   pip install -r requirements.txt
   ```
2. Jalankan preprocessing:

   ```bash
   python src/preprocess.py
   ```
3. Jalankan model Boolean Retrieval:

   ```bash
   python src/boolean_ir.py
   ```
4. Jalankan model VSM:

   ```bash
   python src/vsm_ir.py
   ```
5. Jalankan integrasi (gabungan sistem):

   ```bash
   python src/search_engine.py
   ```

---

### 🌐 B. Menjalankan Aplikasi Streamlit (Antarmuka Web)

File utama untuk deployment adalah `app/main.py`.

#### ▶️ Menjalankan Lokal:

```bash
streamlit run app/main.py
```

Akan muncul tampilan interaktif di browser berisi:

* Input query teks
* Pilihan model: **Boolean**, **VSM TF-IDF**, atau **VSM Sublinear**
* Tabel hasil pencarian
* Nilai evaluasi: Precision, Recall, F1, MAP@k, nDCG@k

#### ☁️ Deployment Online:

> Aplikasi ini dapat di-deploy ke [Streamlit Cloud](https://streamlit.io/cloud) atau platform serupa.
> Link hasil deploy dapat ditulis di bawah ini:

🔗 **Link Deployment:** https://stki-uts-a11202315018-suryani-ayu-dewanti-nfgr37xfbpamuqcmjndc.streamlit.app/

---

## 🧩 5. Dependensi Utama

Pastikan library berikut sudah terinstall:

```bash
pip install streamlit numpy scipy pandas tabulate
```

---

## 🔍 6. Evaluasi Sistem

Model diuji menggunakan query seperti:

* `nama AND universitas`
* `teknik OR semarang`
* `NOT bandung`

dan model **VSM** menggunakan:

* `universitas`
* `fakultas teknik`
* `bandung`

Evaluasi mencakup:

* **Precision**
* **Recall**
* **F1-Score**
* **MAP@5**
* **nDCG@5**

---

## 🧱 7. Cara Push ke GitHub

### 🪜 A. Inisialisasi Repository

```bash
git init
git add .
git commit -m "Initial commit - STKI UTS project"
```

### 🌍 B. Hubungkan ke Repository GitHub

1. Buat repository baru di GitHub (tanpa README).
2. Salin URL repository, misalnya:

   ```
   https://github.com/username/stki-uts.git
   ```
3. Jalankan perintah:

   ```bash
   git remote add origin https://github.com/username/stki-uts.git
   git branch -M main
   git push -u origin main
   ```

---

## 📥 8. Cara Clone Repository

Jika ingin mendownload atau menjalankan di komputer lain:

```bash
git clone https://github.com/username/stki-uts.git
cd stki-uts
pip install -r requirements.txt
streamlit run app/main.py
```

---

## 🧩 9. Lisensi

Proyek ini dibuat untuk keperluan akademik mata kuliah **Sistem Temu Kembali Informasi (STKI)**, Universitas Dian Nuswantoro (UDINUS).
Bebas digunakan untuk pembelajaran dan pengembangan lanjutan.


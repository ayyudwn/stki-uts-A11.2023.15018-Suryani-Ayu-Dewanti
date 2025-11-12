import os
import re
import math
from collections import Counter, defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from numpy.linalg import norm


# ==========================================================
# [1] LOAD DOKUMEN HASIL PREPROCESSING
# ==========================================================
def load_processed_docs(path="data/processed"):
    """
    Membaca dokumen hasil preprocessing dari folder `data/processed`.
    Setiap file teks (*.txt) dibaca dan diubah menjadi list token.

    Output:
    -------
    docs : dict
        { nama_dokumen.txt : [token1, token2, ...] }
    """
    docs = {}
    if not os.path.exists(path):
        print("Folder tidak ditemukan:", path)
        return docs

    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
                docs[filename] = f.read().split()
    return docs


# ==========================================================
# [2] PEMBANGUNAN MODEL TF-IDF (SPARSE MATRIX)
# ==========================================================
def build_tfidf_sparse(docs):
    """
    Membangun representasi TF-IDF dalam bentuk sparse matrix (hemat memori).

    Tahapan:
    --------
    1. Hitung Document Frequency (df)
    2. Hitung Inverse Document Frequency (idf)
    3. Hitung TF-IDF untuk setiap term di setiap dokumen
    4. Simpan ke dalam bentuk matriks sparse

    Output:
    -------
    - doc_ids      : daftar nama dokumen
    - vocab        : daftar term unik
    - term_index   : posisi indeks tiap term dalam vektor
    - idf          : bobot idf per term
    - tfidf_matrix : matriks TF-IDF bentuk sparse
    """
    doc_ids = list(docs.keys())
    N = len(doc_ids)

    # --- Hitung Document Frequency (DF) ---
    df = defaultdict(int)
    for tokens in docs.values():
        for term in set(tokens):
            df[term] += 1

    # --- Hitung Inverse Document Frequency (IDF) ---
    idf = {term: math.log10(N / df[term]) for term in df}

    # --- Buat Vocabulary dan Mapping Index ---
    vocab = sorted(idf.keys())
    term_index = {t: i for i, t in enumerate(vocab)}

    # --- Hitung TF-IDF tiap dokumen ---
    rows, cols, values = [], [], []

    for d_i, doc in enumerate(doc_ids):
        tf = Counter(docs[doc])
        max_tf = max(tf.values())
        for term, freq in tf.items():
            rows.append(d_i)
            cols.append(term_index[term])
            values.append((freq / max_tf) * idf[term])

    # --- Bentuk Matriks Sparse ---
    tfidf_matrix = csr_matrix((values, (rows, cols)),
                              shape=(len(doc_ids), len(vocab)))

    return doc_ids, vocab, term_index, idf, tfidf_matrix


# ==========================================================
# [3] VEKTORISASI QUERY
# ==========================================================
def vectorize_query(query, idf, term_index, vocab_size):
    """
    Mengubah query pengguna menjadi vektor TF-IDF.

    Tahapan:
    --------
    1. Tokenisasi query
    2. Hitung frekuensi (TF)
    3. Kalikan dengan IDF term yang ada di vocab
    4. Hasil berupa vektor sparse berdimensi sama seperti dokumen
    """
    tokens = re.findall(r"\b\w+\b", query.lower())
    tf = Counter(tokens)

    rows, cols, values = [], [], []
    if not tf:
        return csr_matrix((1, vocab_size)), tokens

    max_tf = max(tf.values())

    for term, freq in tf.items():
        if term in term_index:
            rows.append(0)
            cols.append(term_index[term])
            values.append((freq / max_tf) * idf[term])

    q_vec = csr_matrix((values, (rows, cols)), shape=(1, vocab_size))
    return q_vec, tokens


# ==========================================================
# [4] COSINE SIMILARITY (VSM RETRIEVAL)
# ==========================================================
def cosine_similarity_sparse(q_vec, d_matrix):
    """
    Menghitung Cosine Similarity antara query dan setiap dokumen.

    Rumus:
    -------
    sim(q, d) = (q ⋅ d) / (|q| * |d|)

    Output:
    -------
    scores : list nilai kesamaan cosine untuk setiap dokumen
    """
    scores = []
    q_norm = norm(q_vec.toarray())
    for i in range(d_matrix.shape[0]):
        d_vec = d_matrix.getrow(i)
        dot = q_vec.multiply(d_vec).sum()
        d_norm = norm(d_vec.toarray())
        scores.append(dot / (q_norm * d_norm) if q_norm and d_norm else 0)
    return scores


# ==========================================================
# [5] PEMBUATAN SNIPPET (CUKILAN TEKS)
# ==========================================================
def get_snippet(tokens, n=100):
    """
    Mengambil sebagian teks (n karakter pertama)
    untuk ditampilkan sebagai ringkasan hasil pencarian.
    """
    text = " ".join(tokens)
    return text[:n] + "..." if len(text) > n else text


# ==========================================================
# [6] METRIK EVALUASI (P@K, MAP, NDCG)
# ==========================================================
def precision_at_k(results, gold, k):
    """
    Precision@K = proporsi dokumen relevan di antara K hasil teratas.
    """
    r = [doc for doc, _ in results[:k]]
    return sum(1 for d in r if d in gold) / k


def average_precision(results, gold, k):
    """
    MAP (Mean Average Precision)
    Mengukur rata-rata precision kumulatif hingga posisi K.
    """
    hits = 0
    score = 0
    for i, (doc, _) in enumerate(results[:k], start=1):
        if doc in gold:
            hits += 1
            score += hits / i
    return score / len(gold) if gold else 0


def ndcg_at_k(results, gold, k):
    """
    nDCG@K = Normalized Discounted Cumulative Gain
    Mengukur kualitas urutan hasil pencarian dengan mempertimbangkan posisi relevansi.
    """
    dcg, idcg = 0, 0
    for i, (doc, _) in enumerate(results[:k], start=1):
        rel = 1 if doc in gold else 0
        dcg += rel / math.log2(i + 1)

    ideal_hits = min(len(gold), k)
    for i in range(1, ideal_hits + 1):
        idcg += 1 / math.log2(i + 1)

    return dcg / idcg if idcg else 0


# ==========================================================
# [7] MAIN PROGRAM (EVALUASI SISTEM)
# ==========================================================
def main():
    """
    Menjalankan pipeline evaluasi sistem pencarian berbasis VSM (TF-IDF).
    Langkah:
    --------
    1. Load dokumen hasil preprocessing
    2. Bangun representasi TF-IDF (dokumen)
    3. Definisikan gold standard (dokumen relevan per query)
    4. Hitung kesamaan cosine antara query dan dokumen
    5. Evaluasi dengan metrik: Precision@K, MAP@K, nDCG@K
    """
    print("=== Evaluasi Sistem VSM (TF-IDF) ===")

    # --- Step 1: Load dokumen ---
    docs = load_processed_docs("data/processed")
    if not docs:
        print("Tidak ada dokumen pada folder data/processed")
        return

    print(f"{len(docs)} dokumen terbaca")

    # --- Step 2: Bangun TF-IDF Sparse Matrix ---
    doc_ids, vocab, term_index, idf, tfidf = build_tfidf_sparse(docs)

    # --- Step 3: Tentukan Gold Standard untuk evaluasi ---
    file_list = set(doc_ids)
    gold_sets = {
        "universitas":
            set(f for f in file_list if "universitas" in " ".join(docs[f]).lower()),

        "fasilitas":
            set(f for f in file_list if "fasilitas" in " ".join(docs[f]).lower()),

        "fakultas teknik":
            set(f for f in file_list if "fakultas" in " ".join(docs[f]).lower()
                and "teknik" in " ".join(docs[f]).lower()),
    }

    k = 5  # ambil 5 hasil teratas
    sum_p = sum_ap = sum_ndcg = 0

    print("\n=== HASIL PER QUERY ===")
    for query, gold in gold_sets.items():
        print(f"\n Query: {query}")

        # --- Step 4: Vektorisasi dan Hitung Cosine Similarity ---
        q_vec, _ = vectorize_query(query, idf, term_index, len(vocab))
        scores = cosine_similarity_sparse(q_vec, tfidf)

        # --- Urutkan hasil berdasarkan skor kesamaan ---
        ranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
        top_k = ranked[:k]

        # --- Tampilkan hasil pencarian ---
        for rank, (doc, score) in enumerate(top_k, 1):
            print(f"{rank}. {doc} | cos={score:.4f} | {get_snippet(docs[doc])}")

        # --- Step 5: Hitung metrik evaluasi ---
        p = precision_at_k(top_k, gold, k)
        ap = average_precision(top_k, gold, k)
        nd = ndcg_at_k(top_k, gold, k)

        sum_p += p
        sum_ap += ap
        sum_ndcg += nd

        print(f"P@{k}: {p:.2f}, MAP@{k}: {ap:.2f}, nDCG@{k}: {nd:.2f}")
        print("-" * 90)

    # --- Step 6: Rata-rata sistem secara keseluruhan ---
    n = len(gold_sets)
    print("\n=== RATA-RATA SISTEM ===")
    print(f"P@{k}: {sum_p / n:.2f}")
    print(f"MAP@{k}: {sum_ap / n:.2f}")
    print(f"nDCG@{k}: {sum_ndcg / n:.2f}")
    print("Evaluasi selesai.")


# ==========================================================
# [8] EKSEKUSI PROGRAM
# ==========================================================
if __name__ == "__main__":
    main()

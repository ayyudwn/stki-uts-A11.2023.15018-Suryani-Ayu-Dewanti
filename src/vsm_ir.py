import os
import math
import re
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
from numpy.linalg import norm
from tabulate import tabulate


# =============================================================================
# === [1] LOAD DOKUMEN PROSES (HASIL PREPROCESSING) ===
# =============================================================================
def load_processed_docs(path="data/processed"):
    """
    Membaca seluruh dokumen teks dari folder hasil preprocessing.
    Setiap file dianggap sebagai satu dokumen dan dipisahkan menjadi token-token.

    Args:
        path (str): Lokasi folder 'data/processed'

    Returns:
        dict: {nama_file: [token1, token2, ...]}
    """
    docs = {}
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
                docs[filename] = f.read().split()
    return docs


# =============================================================================
# === [2] PEMBANGUNAN MODEL TF-IDF ===
# =============================================================================
def build_tfidf(docs, scheme="standard"):
    """
    Membentuk representasi vektor dokumen dalam bentuk matriks TF-IDF.
    Dua skema bobot didukung: 'standard' dan 'sublinear'.

    Args:
        docs (dict): Kumpulan dokumen hasil tokenisasi.
        scheme (str): Pilihan skema TF ('standard' atau 'sublinear').

    Returns:
        tuple: (doc_ids, vocab, tfidf_matrix, idf, term_index)
    """
    doc_ids = list(docs.keys())
    N = len(doc_ids)

    # --- Hitung Document Frequency (DF) dan Inverse Document Frequency (IDF)
    df = defaultdict(int)
    for tokens in docs.values():
        for term in set(tokens):
            df[term] += 1
    idf = {t: math.log10(N / df[t]) for t in df}

    # --- Bentuk Vocabulary dan Indeks Term
    vocab = sorted(idf.keys())
    term_index = {t: i for i, t in enumerate(vocab)}

    # --- Siapkan struktur sparse matrix untuk TF-IDF
    rows, cols, values = [], [], []

    for d_i, doc in enumerate(doc_ids):
        tf = Counter(docs[doc])

        if scheme == "standard":
            max_tf = max(tf.values()) if tf else 1
            # Normalisasi TF dan kalikan dengan IDF
            for term, freq in tf.items():
                rows.append(d_i)
                cols.append(term_index[term])
                values.append((freq / max_tf) * idf[term])

        elif scheme == "sublinear":
            # Gunakan skema logaritmik pada TF
            for term, freq in tf.items():
                rows.append(d_i)
                cols.append(term_index[term])
                values.append((1 + math.log(freq)) * idf[term])

    # --- Bentuk matriks sparse TF-IDF
    tfidf_matrix = csr_matrix((values, (rows, cols)), shape=(len(doc_ids), len(vocab)))

    return doc_ids, vocab, tfidf_matrix, idf, term_index


# =============================================================================
# === [3] VEKTORISASI QUERY ===
# =============================================================================
def vectorize_query(query, idf, term_index, vocab_size, scheme="standard"):
    """
    Mengubah query teks menjadi vektor TF-IDF menggunakan skema tertentu.

    Args:
        query (str): Query pencarian.
        idf (dict): Nilai IDF untuk setiap term.
        term_index (dict): Pemetaan term ke indeks kolom.
        vocab_size (int): Jumlah total term unik (ukuran vektor).
        scheme (str): Skema TF ('standard' atau 'sublinear').

    Returns:
        csr_matrix: Representasi vektor query.
        list: Token-token query.
    """
    tokens = re.findall(r"\b\w+\b", query.lower())
    tf = Counter(tokens)
    rows, cols, values = [], [], []
    max_tf = max(tf.values()) if tf else 1

    for term, freq in tf.items():
        if term in term_index:
            rows.append(0)
            cols.append(term_index[term])
            if scheme == "standard":
                values.append((freq / max_tf) * idf[term])
            elif scheme == "sublinear":
                values.append((1 + math.log(freq)) * idf[term])

    return csr_matrix((values, (rows, cols)), shape=(1, vocab_size)), tokens


# =============================================================================
# === [4] COSINE SIMILARITY UNTUK PERINGKAT DOKUMEN ===
# =============================================================================
def cosine_similarity(q_vec, tfidf_matrix):
    """
    Menghitung kemiripan kosinus antara query vector dan setiap dokumen.

    Args:
        q_vec (csr_matrix): Vektor query.
        tfidf_matrix (csr_matrix): Matriks TF-IDF dokumen.

    Returns:
        list: Skor cosine similarity untuk setiap dokumen.
    """
    scores = []
    q_array = q_vec.toarray()

    for i in range(tfidf_matrix.shape[0]):
        d_vec = tfidf_matrix.getrow(i).toarray()
        denom = norm(q_array) * norm(d_vec)
        score = (q_array @ d_vec.T)[0, 0] / denom if denom else 0.0
        scores.append(score)

    return scores


# =============================================================================
# === [5] PEMBUATAN SNIPPET ===
# =============================================================================
def get_snippet(tokens, n=120):
    """
    Membuat potongan singkat dari dokumen untuk ditampilkan di hasil pencarian.

    Args:
        tokens (list): Token dokumen.
        n (int): Panjang maksimal snippet.

    Returns:
        str: Cuplikan teks dokumen.
    """
    text = " ".join(tokens)
    return text[:n] + "..." if len(text) > n else text


# =============================================================================
# === [6] MAIN PROGRAM (TESTING PENCARIAN TF-IDF STANDARD & SUBLINEAR) ===
# =============================================================================
if __name__ == "__main__":
    # --- Muat dokumen hasil preprocessing
    docs = load_processed_docs("data/processed")
    if not docs:
        exit("Folder data/processed kosong.")
    print(f"Dokumen terbaca: {len(docs)}")

    # --- Bangun model TF-IDF standard dan sublinear
    doc_ids, vocab, tfidf_std, idf, term_index = build_tfidf(docs, scheme="standard")
    _, _, tfidf_sub, _, _ = build_tfidf(docs, scheme="sublinear")

    # --- Daftar query uji dan gold set (dokumen relevan)
    queries = ["universitas", "fasilitas", "fakultas teknik"]
    gold_sets = {
        "universitas": set(f for f in doc_ids if "itb" in f or "ub" in f),
        "fasilitas": set(f for f in doc_ids if "ugm" in f or "unima" in f),
        "fakultas teknik": set(f for f in doc_ids if "itb" in f or "ui" in f),
    }

    # --- Jalankan pencarian untuk setiap query
    for q in queries:
        print(f"\n=== QUERY: {q} ===")

        # Vectorisasi query
        q_vec_std, _ = vectorize_query(q, idf, term_index, len(vocab), scheme="standard")
        q_vec_sub, _ = vectorize_query(q, idf, term_index, len(vocab), scheme="sublinear")

        # Hitung skor cosine similarity
        scores_std = cosine_similarity(q_vec_std, tfidf_std)
        scores_sub = cosine_similarity(q_vec_sub, tfidf_sub)

        # Urutkan hasil (ranking)
        ranking_std = sorted(zip(doc_ids, scores_std), key=lambda x: x[1], reverse=True)
        ranking_sub = sorted(zip(doc_ids, scores_sub), key=lambda x: x[1], reverse=True)

        # Tampilkan hasil dalam bentuk tabel
        table_std = [[i+1, doc, f"{score:.4f}", get_snippet(docs[doc])]
                     for i, (doc, score) in enumerate(ranking_std)]
        table_sub = [[i+1, doc, f"{score:.4f}", get_snippet(docs[doc])]
                     for i, (doc, score) in enumerate(ranking_sub)]

        print("\nTF-IDF Standard:")
        print(tabulate(table_std, headers=["Rank", "Doc ID", "Cosine", "Snippet"], tablefmt="grid"))

        print("\nTF-IDF Sublinear:")
        print(tabulate(table_sub, headers=["Rank", "Doc ID", "Cosine", "Snippet"], tablefmt="grid"))

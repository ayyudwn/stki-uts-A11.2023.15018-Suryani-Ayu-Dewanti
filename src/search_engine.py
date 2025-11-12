import math
import re
from collections import Counter, defaultdict
from boolean_ir import load_docs, build_inverted_index, eval_boolean, precision_recall
from tabulate import tabulate  # pip install tabulate
from pathlib import Path


# ==========================================================
# === [1] PREPROCESSING (TOKENISASI + STEMMING + STOPWORDS)
# ==========================================================
# Bagian ini bertanggung jawab untuk mempersiapkan teks mentah agar siap diproses
# dalam model pencarian (VSM atau Boolean).
# Langkah-langkah:
#  - Lowercasing (huruf kecil semua)
#  - Menghapus simbol, tanda baca, dan angka
#  - Menghapus stopwords (kata umum seperti “the”, “is”, “of”)
#  - Melakukan stemming (mengembalikan kata ke bentuk dasarnya)
try:
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    STOPWORDS = set(stopwords.words("english"))
except:
    import nltk
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    STOPWORDS = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [ps.stem(w) for w in text.split() if w not in STOPWORDS]
    return tokens


# ==========================================================
# === [2] PEMBANGUNAN MODEL VSM (VECTOR SPACE MODEL)
# ==========================================================
# Fungsi ini menghitung TF (term frequency), DF (document frequency),
# dan IDF (inverse document frequency) dari seluruh dokumen.
# Rumus IDF yang digunakan: log((N+1)/(df+1)) + 1 untuk menghindari pembagian 0.
def build_vsm(docs):
    tf = {}
    df = Counter()
    for doc_id, text in docs.items():
        tokens = preprocess(" ".join(text)) if isinstance(text, list) else preprocess(text)
        tf[doc_id] = Counter(tokens)
        for term in set(tokens):
            df[term] += 1
    N = len(docs)
    idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in df}
    return {"tf": tf, "df": df, "idf": idf, "docs": docs}


# ==========================================================
# === [3] PEMBOBOTAN TF-IDF (STANDAR & SUBLINEAR)
# ==========================================================
# Dua versi pembobotan:
#  (a) Standard TF-IDF   → w = tf * idf
#  (b) Sublinear TF-IDF  → w = (1 + log(tf)) * idf
# Sublinear TF-IDF digunakan untuk meredam pengaruh kata dengan frekuensi sangat tinggi.
def weight_tfidf_standard(vsm):
    return {doc: {term: freq * vsm["idf"].get(term, 0) for term, freq in terms.items()}
            for doc, terms in vsm["tf"].items()}

def weight_tfidf_sublinear(vsm):
    return {doc: {term: (1 + math.log(freq)) * vsm["idf"].get(term, 0) for term, freq in terms.items()}
            for doc, terms in vsm["tf"].items()}


# ==========================================================
# === [4] PERHITUNGAN SIMILARITY & RANKING (COSINE SIMILARITY)
# ==========================================================
# Fungsi ini melakukan pencarian dengan Vector Space Model (VSM):
# - Query diproses sama seperti dokumen
# - Setiap dokumen dihitung skor cosine similarity terhadap query
# - Hasil diurutkan dari skor tertinggi ke terendah
def rank_vsm(query, vsm, wm, scheme="tfidf", top_k=5):
    q_tokens = preprocess(query)
    q_tf = Counter(q_tokens)

    # Pilih metode pembobotan query (standard / sublinear)
    q_vec = {t: (1 + math.log(f)) * vsm["idf"].get(t, 0) if scheme=="sublinear" else f * vsm["idf"].get(t,0)
             for t,f in q_tf.items()}

    doc_scores = []
    for doc, weights in wm.items():
        dot = sum(weights[t] * q_vec[t] for t in q_vec if t in weights)
        d_norm = sum(w**2 for w in weights.values())
        q_norm = sum(wq**2 for wq in q_vec.values())
        if d_norm==0 or q_norm==0: 
            continue
        score = dot / math.sqrt(d_norm * q_norm)
        if score > 0:
            doc_scores.append((doc, score))

    # Urutkan berdasarkan skor tertinggi
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    return doc_scores[:top_k]


# ==========================================================
# === [5] PEMBUATAN SNIPPET (CUKILAN ISI DOKUMEN)
# ==========================================================
# Digunakan untuk menampilkan potongan kecil teks hasil pencarian,
# agar pengguna bisa melihat konteks dokumen tanpa membuka seluruh isi.
def snippet(text, length=80):
    txt = " ".join(text) if isinstance(text, list) else text
    txt = txt.replace("\n"," ")
    return txt[:length]+"..." if len(txt)>length else txt


# ==========================================================
# === [6] PENJELASAN TERM DOMINAN DALAM DOKUMEN
# ==========================================================
# Fungsi ini menampilkan kata-kata yang memiliki bobot tertinggi dalam dokumen
# berdasarkan model TF-IDF, untuk keperluan interpretasi.
def explain_terms(doc, vsm, wm, top_n=3):
    if doc not in wm: return []
    weighted = wm[doc]
    return sorted(weighted.items(), key=lambda x:x[1], reverse=True)[:top_n]


# ==========================================================
# === [7] BOOLEAN SEARCH ENGINE
# ==========================================================
# Menggunakan model Boolean IR yang menafsirkan query logika seperti:
#  - AND  → irisan dokumen
#  - OR   → gabungan dokumen
#  - NOT  → komplemen dokumen
def search_boolean(query, inverted_index):
    return eval_boolean(query, inverted_index)


# ==========================================================
# === [8] EVALUASI KINERJA (PRECISION, RECALL, F1, MAP, NDCG)
# ==========================================================
# Fungsi ini menghitung berbagai metrik evaluasi pencarian:
#  - Precision@k   : proporsi hasil yang relevan di top-k
#  - Recall@k      : proporsi dokumen relevan yang berhasil ditemukan
#  - F1-score      : harmonisasi precision dan recall
#  - MAP@k         : mean average precision
#  - nDCG@k        : normalized discounted cumulative gain
def evaluate_vsm(results, gold_set, k=5):
    r = [doc for doc,_ in results[:k]]
    hits = [1 if doc in gold_set else 0 for doc in r]

    precision = sum(hits)/k
    recall = sum(hits)/len(gold_set) if gold_set else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0

    ap = 0
    num_hits = 0
    for i, h in enumerate(hits,1):
        if h==1:
            num_hits+=1
            ap += num_hits/i
    ap = ap/len(gold_set) if gold_set else 0

    dcg = sum(h/math.log2(i+1) for i,h in enumerate(hits,start=1))
    ideal_hits = min(len(gold_set),k)
    idcg = sum(1/math.log2(i+1) for i in range(1,ideal_hits+1))
    ndcg = dcg/idcg if idcg>0 else 0

    return round(precision,2), round(recall,2), round(f1,2), round(ap,2), round(ndcg,2)


# ==========================================================
# === [9] MAIN PROGRAM (UJI VSM vs BOOLEAN)
# ==========================================================
# Bagian ini menjalankan eksperimen mini search engine:
#  - Memuat dokumen hasil preprocessing
#  - Membangun model Inverted Index & VSM
#  - Menguji performa VSM (TF-IDF & Sublinear)
#  - Menguji pencarian Boolean (AND, OR, NOT)
#  - Menghitung metrik evaluasi untuk tiap query
if __name__=="__main__":
    print("=== MINI SEARCH ENGINE (VSM & BOOLEAN) ===")

    # --- Load dokumen hasil preprocessing
    docs = load_docs("data/processed")
    if not docs:
        exit("Folder data/processed kosong!")

    # --- Bangun index dan model
    inverted = build_inverted_index(docs)
    vsm = build_vsm(docs)
    wm_std = weight_tfidf_standard(vsm)
    wm_sub = weight_tfidf_sublinear(vsm)

    # ==========================================================
    # === [A] UJI DENGAN MODEL VSM (TF-IDF)
    # ==========================================================
    vsm_queries = ["universitas", "fakultas teknik", "bandung"]
    vsm_gold_sets = {
        "universitas": set(f for f in docs if "universitas" in " ".join(docs[f]).lower()),
        "fakultas teknik": set(f for f in docs if "fakultas" in " ".join(docs[f]).lower() and "teknik" in " ".join(docs[f]).lower()),
        "bandung": set(f for f in docs if "bandung" in " ".join(docs[f]).lower()),
    }

    for q in vsm_queries:
        print(f"\n=== VSM Query: {q} ===")
        res_std = rank_vsm(q,vsm,wm_std)
        res_sub = rank_vsm(q,vsm,wm_sub,scheme="sublinear")

        # --- Tabel hasil VSM TF-IDF
        table_std = [[i+1, doc, f"{score:.4f}", snippet(docs[doc])] for i,(doc,score) in enumerate(res_std)]
        print("\nVSM TF-IDF:")
        print(tabulate(table_std, headers=["Rank","Doc ID","Cosine","Snippet"], tablefmt="grid"))

        # --- Tabel hasil VSM Sublinear TF-IDF
        table_sub = [[i+1, doc, f"{score:.4f}", snippet(docs[doc])] for i,(doc,score) in enumerate(res_sub)]
        print("\nVSM TF-IDF Sublinear:")
        print(tabulate(table_sub, headers=["Rank","Doc ID","Cosine","Snippet"], tablefmt="grid"))

        # --- Evaluasi performa
        gold = vsm_gold_sets.get(q,set())
        p,r,f,mapk,ndcg = evaluate_vsm(res_std, gold)
        print(f"\nVSM TF-IDF | Precision: {p}, Recall: {r}, F1: {f}, MAP@5: {mapk}, nDCG@5: {ndcg}")
        p,r,f,mapk,ndcg = evaluate_vsm(res_sub, gold)
        print(f"VSM Sublinear | Precision: {p}, Recall: {r}, F1: {f}, MAP@5: {mapk}, nDCG@5: {ndcg}")


    # ==========================================================
    # === [B] UJI DENGAN BOOLEAN MODEL
    # ==========================================================
    boolean_queries = [
        "universitas AND fakultas",
        "fakultas AND teknik",
        "NOT bandung",
        "universitas OR bandung"
    ]
    boolean_gold_sets = {
        "universitas AND fakultas": set(f for f in docs if "universitas" in " ".join(docs[f]).lower() and "fakultas" in " ".join(docs[f]).lower()),
        "fakultas AND teknik": set(f for f in docs if "fakultas" in " ".join(docs[f]).lower() and "teknik" in " ".join(docs[f]).lower()),
        "NOT bandung": set(docs.keys()) - set(f for f in docs if "bandung" in " ".join(docs[f]).lower()),
        "universitas OR bandung": set(f for f in docs if "universitas" in " ".join(docs[f]).lower() or "bandung" in " ".join(docs[f]).lower())
    }

    for q in boolean_queries:
        print(f"\n=== Boolean Query: {q} ===")
        bool_res = search_boolean(q, inverted)
        if bool_res:
            bool_table = [[i+1, doc, snippet(docs[doc])] for i,doc in enumerate(sorted(bool_res))]
            print(tabulate(bool_table, headers=["No","Doc ID","Snippet"], tablefmt="grid"))
        else:
            print("Hasil Dokumen : Tidak tersedia")

        # --- Evaluasi Boolean Search
        gold = boolean_gold_sets.get(q,set())
        p,r,f,_,_ = evaluate_vsm([(d,1) for d in bool_res], gold)
        print(f"\nBoolean | Precision: {p}, Recall: {r}, F1: {f}")

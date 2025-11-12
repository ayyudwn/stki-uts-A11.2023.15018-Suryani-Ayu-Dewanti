from collections import defaultdict
from pathlib import Path
import re


# ============================================================
# === [1] LOAD DOKUMEN HASIL PREPROCESSING ===================
# ============================================================
def load_docs(path="data/processed"):
    """
    Membaca semua file teks hasil preprocessing dari folder 'data/processed'.
    Setiap dokumen disimpan sebagai list token.

    Output:
    -------
    docs : dict
        { nama_file.txt : [token1, token2, ...] }
    """
    docs = {}
    p = Path(path)
    if not p.exists():
        print("Folder data/processed tidak ditemukan!")
        return docs

    for f in p.glob("*.txt"):
        docs[f.name] = f.read_text(encoding="utf-8").split()

    print(f"{len(docs)} dokumen dimuat.")
    return docs


# ============================================================
# === [2A] MEMBANGUN INCIDENCE MATRIX (SPARSE) ===============
# ============================================================
def build_incidence_matrix(docs):
    """
    Membangun *incidence matrix* (versi sparse).
    Tiap term menyimpan himpunan dokumen yang mengandung term tersebut.

    Output:
    -------
    matrix : dict
        { term : {dok1, dok2, ...} }
    """
    matrix = defaultdict(set)
    for doc, toks in docs.items():
        for term in set(toks):
            matrix[term].add(doc)
    return dict(matrix)


# ============================================================
# === [2B] MEMBANGUN INVERTED INDEX ==========================
# ============================================================
def build_inverted_index(docs):
    """
    Membangun inverted index (struktur pencarian cepat untuk Boolean retrieval).
    Sama seperti incidence matrix, tapi lebih ringkas untuk lookup.

    Output:
    -------
    inverted : dict
        { term : {dok1, dok2, ...} }
    """
    inverted = defaultdict(set)
    for doc, toks in docs.items():
        for t in set(toks):
            inverted[t].add(doc)
    return dict(inverted)


# ============================================================
# === [3] BOOLEAN QUERY PARSER (dengan perbaikan NOT) ========
# ============================================================
def eval_boolean(query, inverted_index):
    """
    Mengevaluasi query Boolean sederhana (tanpa kurung).
    Mendukung operator: AND, OR, NOT.

    Contoh:
    -------
    "bandung AND itb"
    "teknik OR semarang"
    "NOT bandung"

    Jika pengguna mengetik "NOT bandung", maka sistem akan
    mengembalikan semua dokumen yang *tidak* mengandung kata 'bandung'.
    """
    # --- Normalisasi & tokenisasi query ---
    q = query.lower().strip()
    tokens = re.findall(r'\bnot\b|\band\b|\bor\b|[\w-]+', q)
    tokens = [t.upper() if t in ('and', 'or', 'not') else t for t in tokens]

    # --- Prioritas operator ---
    prec = {"NOT": 3, "AND": 2, "OR": 1}
    output, stack = [], []

    # --- Infix → Postfix (Reverse Polish Notation) ---
    for tok in tokens:
        if tok in ("AND", "OR", "NOT"):
            while stack and prec.get(stack[-1], 0) >= prec[tok]:
                output.append(stack.pop())
            stack.append(tok)
        else:
            output.append(tok)
    while stack:
        output.append(stack.pop())

    # --- Evaluasi Postfix ---
    all_docs = set().union(*inverted_index.values()) if inverted_index else set()
    eval_stack = []

    for tok in output:
        if tok == "NOT":
            # Jika query diawali NOT, anggap operand kiri = semua dokumen
            if not eval_stack:
                eval_stack.append(all_docs)
            A = eval_stack.pop()
            eval_stack.append(all_docs - A)
        elif tok == "AND":
            if len(eval_stack) >= 2:
                B = eval_stack.pop()
                A = eval_stack.pop()
                eval_stack.append(A & B)
        elif tok == "OR":
            if len(eval_stack) >= 2:
                B = eval_stack.pop()
                A = eval_stack.pop()
                eval_stack.append(A | B)
        else:
            eval_stack.append(inverted_index.get(tok, set()))

    return eval_stack[-1] if eval_stack else set()


# ============================================================
# === [4] PERHITUNGAN PRECISION & RECALL =====================
# ============================================================
def precision_recall(result, gold):
    """
    Menghitung Precision & Recall antara hasil pencarian dan truth set (jawaban benar).

    Precision = dokumen relevan / semua dokumen hasil
    Recall    = dokumen relevan / semua dokumen seharusnya relevan
    """
    if not result and not gold:
        return 1, 1
    if not result:
        return 0, 0

    tp = len(result & gold)
    precision = tp / len(result)
    recall = tp / len(gold) if gold else 0

    return round(precision, 2), round(recall, 2)


# ============================================================
# === [5] PENJELASAN OPERATOR BOOLEAN ========================
# ============================================================
def explain_ops(query):
    """
    Menjelaskan makna operator Boolean yang digunakan pada query.
    """
    ops = []
    if "AND" in query.upper():
        ops.append("AND = Interseksi himpunan dokumen (harus mengandung semua term)")
    if "OR" in query.upper():
        ops.append("OR  = Union dokumen (mengandung salah satu term)")
    if "NOT" in query.upper():
        ops.append("NOT = Komplemen dokumen (semua dokumen kecuali yang mengandung term)")
    return ops


# ============================================================
# === [6] MAIN TESTING & EVALUASI ============================
# ============================================================
if __name__ == "__main__":
    # --- Step 1: Load dokumen hasil preprocessing ---
    docs = load_docs()

    if not docs:
        exit()

    # --- Step 2: Bangun struktur index ---
    inverted = build_inverted_index(docs)
    incidence = build_incidence_matrix(docs)

    print("\n=== BOOLEAN RETRIEVAL – UJI TUGAS STKI ===")

    # --- Step 3: Definisi truth set untuk uji akurasi ---
    truth_sets = {
        "nama AND universitas": {"stmik_bm_palu.txt", "itb.txt", "ugm.txt", "unpad.txt"},
        "teknik OR semarang": set(docs.keys()),
        "NOT bandung": set(f for f in docs if "bandung" not in docs[f])
    }

    # --- Step 4: Evaluasi tiap query ---
    for query, gold in truth_sets.items():
        print("\n----------------------------------------------------")
        print(f"QUERY : {query}")

        result = eval_boolean(query, inverted)
        print(f"Hasil Dokumen : {sorted(result)}")

        # Penjelasan operasi Boolean
        ops = explain_ops(query)
        for o in ops:
            print("-", o)

        # Hitung precision & recall
        P, R = precision_recall(result, gold)
        print(f"Precision : {P}")
        print(f"Recall    : {R}")

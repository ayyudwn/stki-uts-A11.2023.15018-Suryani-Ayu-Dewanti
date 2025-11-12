# app/main.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from search_engine import build_vsm, weight_tfidf_standard, weight_tfidf_sublinear, rank_vsm, snippet, search_boolean, evaluate_vsm
from boolean_ir import load_docs, build_inverted_index
import streamlit as st

# ==== STREAMLIT INTERFACE ====
st.title("Mini Search Engine (UTS)")

docs = load_docs("data/processed")
if not docs:
    st.error("Folder data/processed kosong!")
    st.stop()

inverted = build_inverted_index(docs)
vsm = build_vsm(docs)
wm_std = weight_tfidf_standard(vsm)
wm_sub = weight_tfidf_sublinear(vsm)

query = st.text_input("Masukkan query:")

scheme = st.selectbox("Skema pencarian", ["Boolean", "VSM TF-IDF", "VSM Sublinear"])
top_k = st.slider("Top-k dokumen", 1, 10, 5)

if query:
    st.subheader("Hasil Pencarian")
    if scheme=="Boolean":
        res = search_boolean(query, inverted)
        res_sorted = sorted(res)[:top_k]  # Batasi sesuai top_k
        table = [[i+1, doc, snippet(docs[doc])] for i, doc in enumerate(res_sorted)]
        st.table(table)

    else:
        wm = wm_std if scheme=="VSM TF-IDF" else wm_sub
        res = rank_vsm(query, vsm, wm, "sublinear" if scheme=="VSM Sublinear" else "tfidf", top_k)
        table = [[i+1, doc, f"{score:.4f}", snippet(docs[doc])] for i,(doc,score) in enumerate(res)]
        st.table(table)

    gold_set = set(f for f in docs if any(w in " ".join(docs[f]).lower() for w in query.lower().split()))
    ev_res = [(d,1) for d in res] if scheme=="Boolean" else res
    p,r,f,mapk,ndcg = evaluate_vsm(ev_res, gold_set)
    st.subheader("Evaluasi")
    st.write(f"Precision: {p}, Recall: {r}, F1: {f}, MAP@{top_k}: {mapk}, nDCG@{top_k}: {ndcg}")

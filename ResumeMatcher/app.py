# app.py
"""
Streamlit frontend for RAG Resume Matcher (calls backend.run_resume_match)
"""

import streamlit as st
from backend import extract_text_from_upload, run_resume_match, simple_token_set
import json

st.set_page_config(page_title="RAG Resume Matcher (Detailed)", layout="wide")
st.title("📄 RAG Resume Matcher — Detailed 8-criteria evaluation")

st.markdown("Paste or upload JD and Resume. The backend will index the resume, retrieve relevant snippets for the JD, and ask the LLM to return a structured JSON evaluation.")

col1, col2 = st.columns(2)

with col1:
    jd_text = st.text_area("Paste Job Description (JD)", height=200)
    jd_file = st.file_uploader("Or upload JD file (txt/pdf/docx)", type=["txt", "pdf", "docx"])
    if jd_file and not jd_text:
        jd_text = extract_text_from_upload(jd_file)

    top_k = st.slider("Number of snippets to retrieve (k)", min_value=1, max_value=12, value=6)
    chunk_size = st.number_input("Resume chunk size (chars)", min_value=200, max_value=2000, value=800, step=50)
    chunk_overlap = st.number_input("Resume chunk overlap (chars)", min_value=0, max_value=800, value=150, step=25)

with col2:
    resume_text = st.text_area("Paste Resume text", height=200)
    resume_file = st.file_uploader("Or upload Resume file (txt/pdf/docx)", type=["txt", "pdf", "docx"])
    if resume_file and not resume_text:
        resume_text = extract_text_from_upload(resume_file)

    llm_model = st.text_input("Ollama LLM model name (local)", value="llama2")
    embed_model = st.text_input("Ollama embedding model name", value="nomic-embed-text")

run_btn = st.button("Run Detailed RAG Match")

if run_btn:
    if not jd_text or not resume_text:
        st.warning("Please provide both JD and Resume (paste or upload).")
    else:
        with st.spinner("Running RAG pipeline and asking LLM..."):
            result = run_resume_match(jd_text=jd_text,
                                      resume_text=resume_text,
                                      llm_model=llm_model,
                                      embed_model=embed_model,
                                      chunk_size=int(chunk_size),
                                      chunk_overlap=int(chunk_overlap),
                                      top_k=int(top_k))

        if not result.raw_llm_output:
            st.error("No LLM output returned. Check Ollama availability and model names.")
        else:
            st.success(f"Overall fit score: {result.overall_fit_score} / 5")
            st.write(f"Recommendation: **{result.recommendation}**")
            st.write(f"Summary: {result.summary}")

            st.subheader("Full parsed fit_evaluation")
            try:
                st.json(result.fit_evaluation)
            except Exception:
                st.write("Unable to display structured fit_evaluation; raw LLM JSON below.")
                st.code(result.raw_llm_output, language="json")

            st.subheader("Retrieved resume snippets (used as evidence)")
            for i, r in enumerate(result.retrieved, start=1):
                st.markdown(f"**Snippet {i}**")
                st.write(r.get("text")[:1500] + ("..." if len(r.get("text","")) > 1500 else ""))

            # quick token overlap signal
            jd_tokens = simple_token_set(jd_text)
            resume_tokens = simple_token_set(resume_text)
            overlap = jd_tokens.intersection(resume_tokens)
            st.subheader("Quick token overlap (skills-like)")
            st.write(f"Overlap ratio: {len(overlap) / max(1, len(jd_tokens)):.3f}")
            st.write(f"Top overlap tokens: {list(overlap)[:30]}")

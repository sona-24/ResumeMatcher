# AI Resume Matcher

An AI-powered resume-to-job description matching tool that uses **LangChain**, **RAG (Retrieval-Augmented Generation)**, **OllamaEmbeddings**, and **FAISS** to evaluate candidate fit. The system scores candidates (0–5) based on skills, experience, and domain relevance, and provides a recommendation (Shortlist / Consider / Reject) with explanations.

## ✨ Features
- **Resume & Job Description Matching** – Uses semantic search for accurate alignment.
- **Candidate Scoring** – Based on multiple evaluation parameters (skills, experience, education, etc.).
- **RAG Pipeline** – Combines vector search with LLM reasoning.
- **Interactive UI** – Built with Streamlit for easy usage.
- **Customizable Parameters** – Easily adjust scoring logic.

## 🛠 Tech Stack
- **Python 3.10+**
- **LangChain**
- **FAISS**
- **OllamaEmbeddings**
- **Streamlit**
- **PyPDF2** (for resume parsing)
- **python-docx** (for DOCX parsing)

## 📂 Project Structure
```
AI-Resume-Matcher/
│-- app.py               # Streamlit app entry point
│-- resume_matcher.py    # Core matching & scoring logic
│-- utils.py             # Helper functions
│-- requirements.txt     # Dependencies
│-- sample_data/         # Example resumes & job descriptions
│-- README.md            # Project documentation
```

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/ai-resume-matcher.git
cd ai-resume-matcher
```

### 2️⃣ Create & Activate a Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Mac/Linux
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Install & Run Ollama (for local embeddings)
- Download from: [https://ollama.ai/download](https://ollama.ai/download)  
- Start Ollama in background:
```bash
ollama run llama2
```

### 5️⃣ Run the Application
```bash
streamlit run app.py
```

---

✅ **You can now open the Streamlit URL (usually http://localhost:8501) in your browser and start matching resumes with job descriptions!**

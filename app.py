import streamlit as st

# ✅ MUST BE FIRST Streamlit command
st.set_page_config(page_title="RAG QA from URL", layout="centered")

# Now imports
from newspaper import Article
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import numpy as np
import faiss
import os

@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "tiiuae/falcon-7b-instruct", device_map="auto", torch_dtype="auto"
    )
    rag_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return rag_pipeline, embedder

rag_pipeline, embedder = load_models()

def extract_article_text(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.title, article.text

def embed_text(text, chunk_size=512, overlap=100):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]
    embeddings = embedder.encode(chunks)
    return chunks, embeddings

def build_faiss_index(embeddings):
    matrix = np.vstack(embeddings).astype("float32")
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    return index

def search(query, chunks, index, k=3):
    q_vec = embedder.encode([query]).astype("float32")
    _, I = index.search(q_vec, k)
    return [chunks[i] for i in I[0]]

def answer_query(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"[INST] Use the context below to answer the question:\n\n{context}\n\nQuestion: {query} [/INST]"
    output = rag_pipeline(prompt, max_new_tokens=100, temperature=0.7)
    return output[0]['generated_text'].replace(prompt, "").strip()

# ========== Streamlit UI ==========

st.title("🧠 Ask Questions from a Web Article")

url = st.text_input("🔗 Enter the article URL:")
if st.button("📥 Extract and Index"):
    try:
        title, text = extract_article_text(url)
        st.session_state.chunks, embeddings = embed_text(text)
        st.session_state.index = build_faiss_index(embeddings)
        st.success(f"✅ Indexed: {title}")
    except Exception as e:
        st.error(f"❌ Error: {e}")

if "index" in st.session_state:
    question = st.text_input("❓ Ask your question:")
    if st.button("🤖 Get Answer") and question:
        top_chunks = search(question, st.session_state.chunks, st.session_state.index)
        answer = answer_query(question, top_chunks)
        st.success(answer)

        with st.expander("🔍 Retrieved Chunks"):
            for i, chunk in enumerate(top_chunks):
                st.markdown(f"**Chunk {i+1}**\n> {chunk[:400]}...")

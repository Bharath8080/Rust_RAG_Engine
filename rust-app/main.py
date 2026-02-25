import streamlit as st
import os
import time
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import base64
from PyPDF2 import PdfReader
from fastembed import TextEmbedding
from groq import Groq
import rag  # Native Rust Extension

# ------------------ CONFIG ------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"
CHUNK_SIZE = 1000
OVERLAP = 200
TOP_K = 5

st.set_page_config(page_title="Rust RAG", page_icon="🦀", layout="centered")

# Custom CSS for a premium feel
st.markdown("""
<style>
    .stMetric {
        background-color: #0e1117;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #333;
    }
    div.stButton > button:first-child {
        background-color: #f25724;
        color: white;
        font-weight: bold;
    }
    .latency-pill {
        display: inline-flex;
        align-items: center;
        background-color: #111;
        border: 1px solid #1a1a1a;
        border-left: 3px solid #00ff00;
        border-radius: 5px;
        padding: 8px 16px;
        color: #999;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
    }
    .latency-pill span {
        color: #ddd;
        font-weight: 600;
        margin: 0 4px;
    }
    .latency-pill .icon {
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ------------------ SESSION SETUP ------------------
if "engine" not in st.session_state:
    st.session_state.engine = rag.RagEngine()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed" not in st.session_state:
    st.session_state.indexed = False

# ------------------ INITIALIZE MODELS ------------------
@st.cache_resource
def load_embedder():
    return TextEmbedding(model_name=MODEL_NAME)

@st.cache_resource
def load_llm():
    return Groq(api_key=GROQ_API_KEY)

embedder = load_embedder()
llm = load_llm()

# ------------------ LLM WARM-UP ------------------
if "warmed_up" not in st.session_state:
    # Silent warm-up call to eliminate first-response latency
    try:
        llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "."}],
            max_tokens=1
        )
        st.session_state.warmed_up = True
    except:
        pass

# ------------------ RAG WORKFLOW ------------------

def process_question(question, top_k):
    start_total = time.perf_counter()
    
    # 1. Embedding
    t0 = time.perf_counter()
    query_vec = list(embedder.embed([question]))[0] # No prefix for all-MiniLM-L6-v2
    embed_time = (time.perf_counter() - t0) * 1000

    # 2. Native Rust Search
    t1 = time.perf_counter()
    results = st.session_state.engine.search(query_vec.tolist(), top_k)
    search_time = (time.perf_counter() - t1) * 1000

    context = "\n".join([r[0] for r in results])

    # 3. LLM Generation
    t2 = time.perf_counter()
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer accurately based only on context:"
    
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    gen_time = (time.perf_counter() - t2) * 1000
    answer = response.choices[0].message.content
    
    total_latency = (time.perf_counter() - start_total) * 1000
    
    return {
        "content": answer,
        "context": context,
        "timings": {
            "Embedding": f"{embed_time:.1f}ms",
            "Rust Search": f"{search_time:.3f}ms",
            "LLM Generation": f"{gen_time:.1f}ms",
            "Total Pipeline": f"{total_latency:.1f}ms"
        }
    }

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.title("📂 Rust Knowledge Base")

    uploaded_files = st.file_uploader("Upload Documents (PDF/Text)", type=["pdf", "txt"], accept_multiple_files=True)

    if uploaded_files and st.button("Index Documents", use_container_width=True):
        t0 = time.perf_counter()
        with st.spinner("Rust Engine: Indexing..."):
            all_text = ""
            for file in uploaded_files:
                if file.type == "application/pdf":
                    reader = PdfReader(file)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            all_text += extracted + "\n"
                else:
                    all_text += file.read().decode("utf-8") + "\n"

            # 1. Smarter Chunking in Rust
            chunks = st.session_state.engine.chunk_text(all_text, CHUNK_SIZE, OVERLAP)
            
            # 2. Parallel Embedding
            embeddings = list(embedder.embed(chunks))
            flat_embeddings = np.array(embeddings).flatten().tolist()
            
            # 3. Native Storage & Optimization
            st.session_state.engine.load_embeddings(flat_embeddings, len(embeddings[0]))
            st.session_state.indexed = True
            index_time = time.perf_counter() - t0
            st.success(f"Indexed {len(chunks)} chunks in {index_time:.2f}s")


# ------------------ MAIN INTERFACE ------------------
def display_chat_history():
    """Renders the chat history with custom metrics badge."""
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                # Compact Premium Badge with all metrics
                total_sec = float(msg["timings"]["Total Pipeline"].replace("ms", "")) / 1000
                tok_per_sec = int((len(msg["content"].split()) * 1.3) / total_sec) if total_sec > 0 else 0
                
                emb = msg["timings"]["Embedding"]
                src = msg["timings"]["Rust Search"]
                llm_gen = msg["timings"]["LLM Generation"]

                st.markdown(f"""
                    <div class="latency-pill">
                        <div class="icon">🔥</div>
                        <span>{total_sec:.3f}s total</span> • 
                        <span>{emb}</span> embed • 
                        <span>{src}</span> search • 
                        <span>{llm_gen}</span> llm • 
                        <span>{tok_per_sec:,} tok/s</span>
                    </div>
                    """, unsafe_allow_html=True)

def handle_user_input():
    """Handles the chat input and RAG processing."""
    question = st.chat_input("Ask any question from your knowledge base...")

    if question:
        if not st.session_state.indexed:
            st.error("Please index documents first!")
        else:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.spinner("Rust Engine Searching..."):
                result = process_question(question, TOP_K)
                st.session_state.chat_history.append({"role": "assistant", **result})
            st.rerun()

def main():
    """Main chat layout."""
    python_logo = base64.b64encode(open("assets/python.gif", "rb").read()).decode()
    rust_logo = base64.b64encode(open("assets/RustLogo.jpg", "rb").read()).decode()

    st.markdown(f"""
    <style>
        .gradient-text {{
            background: linear-gradient(to right, #EC227A, #F77334);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: inherit;
            font-weight: bold;
            display: inline-block;
        }}
    </style>
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h2 style='margin-bottom: 0.5rem;'>
            <img src="data:image/gif;base64,{python_logo}" width="42" style="vertical-align: middle; margin-right: 10px;">
            Rust & Python Optimized RAG Engine
            <img src="data:image/jpeg;base64,{rust_logo}" width="45" style="vertical-align: middle; margin-left: 10px;">
        </h2>
        <h3 style='margin-top: 0; color: #888; font-weight: 400; font-size: 1.2rem;'>
            Sub-Millisecond Retrieval
        </h3>
    </div>
    """, unsafe_allow_html=True)

    display_chat_history()
    handle_user_input()

if __name__ == "__main__":
    main()


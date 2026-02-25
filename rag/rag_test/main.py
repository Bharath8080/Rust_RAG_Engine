import streamlit as st
import os
import time
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from pypdf import PdfReader
from fastembed import TextEmbedding
from groq import Groq
import rag  # Native Rust Extension
from go_client import GoRagClient # Go TCP Client

# ------------------ CONFIG ------------------
MODEL_NAME = "snowflake/snowflake-arctic-embed-xs"
LLM_MODEL = "llama-3.1-8b-instant"

st.set_page_config(page_title="FastRAG ⚡", page_icon="⚡", layout="wide")

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
        background-color: #00ff00;
        color: black;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ------------------ SESSION SETUP ------------------
if "engine" not in st.session_state:
    st.session_state.engine = rag.RagEngine()
if "go_engine" not in st.session_state:
    # Use absolute path to the Go binary
    go_exe = os.path.join(os.getcwd(), "go-rag", "go_rag.exe")
    st.session_state.go_engine = GoRagClient(go_exe)
    # The server starts automatically on the first command if not running
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "current_engine_type" not in st.session_state:
    st.session_state.current_engine_type = "Rust"

# ------------------ INITIALIZE MODELS ------------------
@st.cache_resource
def load_embedder():
    return TextEmbedding(model_name=MODEL_NAME)

@st.cache_resource
def load_llm():
    return Groq(api_key=GROQ_API_KEY)

embedder = load_embedder()
llm = load_llm()

# ------------------ RAG WORKFLOW ------------------

def process_question(question):
    start_total = time.perf_counter()
    
    # 1. Embedding
    t0 = time.perf_counter()
    query_vec = list(embedder.embed([f"query: {question}"]))[0]
    embed_time = (time.perf_counter() - t0) * 1000

    # 2. Search
    t1 = time.perf_counter()
    if st.session_state.current_engine_type == "Rust":
        results = st.session_state.engine.search(query_vec.tolist(), top_k)
        context_list = [r[0] for r in results]
    else:
        # Go Search
        context_list = st.session_state.go_engine.search(query_vec.tolist(), top_k)
    
    search_time = (time.perf_counter() - t1) * 1000

    context = "\n\n---\n\n".join(context_list)

    # 3. LLM Generation
    t2 = time.perf_counter()
    prompt = f"""Use the following medical/document context to answer the question. 
Be precise and thorough. If the information is not in the context, say so.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
    
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    gen_time = (time.perf_counter() - t2) * 1000
    answer = response.choices[0].message.content
    
    total_latency = (time.perf_counter() - start_total) * 1000
    
    return {
        "content": answer,
        "context": context,
        "timings": {
            "Embedding": f"{embed_time:.1f}ms",
            "Native Search": f"{search_time:.3f}ms",
            "LLM Generation": f"{gen_time:.1f}ms",
            "Total Pipeline": f"{total_latency:.1f}ms"
        }
    }

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.title("📂 Knowledge Base")
    
    # Engine Selection
    st.session_state.current_engine_type = st.radio(
        "Optimization Engine",
        ["Rust", "Go"],
        index=0 if st.session_state.current_engine_type == "Rust" else 1,
        horizontal=True,
        help="Rust: Native PyO3. Go: HNSW over TCP."
    )

    uploaded_files = st.file_uploader("Upload Medical Docs (PDF/Text)", type=["pdf", "txt"], accept_multiple_files=True)

    c1, c2, c3 = st.columns(3)
    chunk_size = c1.number_input("Chunk Size", value=1000, step=100)
    overlap = c2.number_input("Overlap", value=200, step=50)
    top_k = c3.number_input("Top K", value=5, min_value=1, max_value=10)

    if uploaded_files and st.button("Index Documents", use_container_width=True):
        with st.spinner(f"{st.session_state.current_engine_type} Engine: Indexing..."):
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

            # Use Rust for chunking for consistency (the logic is identical)
            chunks = st.session_state.engine.chunk_text(all_text, chunk_size, overlap)
            
            # Embed
            embeddings = list(embedder.embed(chunks))
            flat_embeddings = np.array(embeddings).flatten().tolist()
            dim = len(embeddings[0])
            
            # Load into the selected engine (or both for easy switching)
            if st.session_state.current_engine_type == "Rust":
                st.session_state.engine.load_embeddings(flat_embeddings, dim)
            else:
                # Go needs the chunks locally if we want to retrieve them
                # Our go_client already handles this if we call process_documents
                # But since we already have chunks here, let's just use the load_embeddings
                # Wait, Go engine stores chunks in its own memory.
                st.session_state.go_engine.process_documents(all_text, chunk_size, overlap)
                st.session_state.go_engine.load_embeddings(flat_embeddings, dim)

            st.session_state.indexed = True
            st.success(f"Indexed {len(chunks)} contextual chunks in {st.session_state.current_engine_type}.")

    if st.button("Clear Cache", use_container_width=True):
        st.session_state.engine.clear()
        st.session_state.go_engine.clear()
        st.session_state.chat_history = []
        st.session_state.indexed = False
        st.rerun()

    st.divider()
    st.info("⚡ **Smart Splitter Active**: Rust now respects paragraph and sentence boundaries for higher accuracy.")

# ------------------ MAIN INTERFACE ------------------
st.title("⚡ FastRAG: Accuracy & Speed Edition")
st.markdown("Native Rust storage + Sentence-aware chunking.")

# Chat Input
question = st.chat_input("Ask about treatment, diagnosis, or risk factors...")

if question:
    if not st.session_state.indexed:
        st.error("Please index documents first!")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.spinner("Native Engine Searching..."):
            result = process_question(question)
            st.session_state.chat_history.append({"role": "assistant", **result})

# Render Chat
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            # Display metrics
            cols = st.columns(len(msg["timings"]))
            for i, (label, val) in enumerate(msg["timings"].items()):
                cols[i].metric(label, val)
            
            with st.expander("🔍 View Retrieved Context"):
                st.code(msg["context"], wrap_lines=True)

st.divider()
st.caption("Optimized RAG Engine • Sub-millisecond Search • Accurate Contextual Retrieval")
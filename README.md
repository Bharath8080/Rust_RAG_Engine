# Rust-Optimized RAG Engine: Setup Guide

This guide covers the end-to-end setup for the high-performance Rust-based RAG application. We use **uv** for Python package management and **maturin** to bridge the Rust engine with Python.

---

## 🛠 Prerequisites

Ensure you have the following installed:

1. **Rust Toolchain**: [Install via rustup](https://rustup.rs/) (Required to build the native engine).
2. **Python 3.9+**: Recommended for compatibility with FastEmbed.
3. **uv**: [Install via curl/pip](https://github.com/astral-sh/uv) (Extremely fast Python package manager).
4. **Groq API Key**: Get one from [Groq Console](https://console.groq.com/).
5. **Qdrant**: (Optional) For advanced vector storage.

---

## 🚀 Quick Start (From Scratch)

### 1. Initialize Python Environment

From the project root (`Rust-RAG`), create and activate a virtual environment:

```powershell
# Create venv using uv
uv venv

# Activate (Windows)
.venv\Scripts\activate
```

### 2. Install Development Tools

Install `maturin` globally using `uv` tools for easier building:

```powershell
uv tool install maturin
uv tool update-shell

# Add local bin to Path if not already there
$env:Path += ";C:\Users\homeu\.local\bin"
```

### 3. Install Dependencies

Install all required libraries, including `qdrant-client`:

```powershell
uv pip install streamlit numpy PyPDF2 fastembed groq python-dotenv maturin qdrant-client
```

---

## 🦀 Building the Rust Engine

The core logic (chunking & vector search) is written in Rust for sub-millisecond performance. You must build and install it as a Python module:

```powershell
# Navigate to the rust engine directory
cd rag

# Build and install the module into your venv
maturin develop --release

# Return to root
cd ..
```

---

## 🔑 Environment Configuration

Create a `.env` file in the root directory and add your Groq API key:

```env
GROQ_API_KEY=gsk_your_key_here
```

---

## 🏃‍♂️ Running the Application

Once the Rust core is built and dependencies are installed, you can launch the Streamlit interface:

```powershell
streamlit run rust-app/main.py
```

---

## 📁 Project Structure

- `rag/`: Native Rust source code (High-performance Core).
- `rust-app/`: Streamlit frontend and Python glue logic.
- `assets/`: Logos and UI media.
- `.venv/`: Managed Python environment.

---

## 💡 Troubleshooting

- **Rust Build Errors**: Ensure `cargo` is in your PATH and your Rust version is up to date (`rustup update`).
- **Module Not Found**: If Python says `import rag` fails, ensure you ran `maturin develop --release` inside the active venv.
- **API Errors**: Double-check your `.env` file for typos in the variable name.

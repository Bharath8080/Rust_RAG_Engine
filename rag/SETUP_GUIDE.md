# Rust-RAG Setup and Troubleshooting Guide

This guide documents the complete setup process for the `rag` Rust extension, including the fixes for common environment and build issues on Windows.

## 1. Initial Prerequisites

Before building, ensure you have the following installed:

1.  **Rust**: Installed via [rustup.rs](https://rustup.rs/).
2.  **Maturin**: Installed via `uv` or `pip`.
    ```powershell
    uv tool install maturin
    uv tool update-shell
    ```

## 2. Environment Configuration (PATH Fix)

On Windows, tools installed by Rust (`.cargo/bin`) and `uv` (`.local/bin`) may not be automatically added to your shell's `PATH`.

### Manual Session Fix

Run this in your PowerShell session to temporarily add the directories:

```powershell
$env:Path += ";C:\Users\homeu\.cargo\bin;C:\Users\homeu\.local\bin"
```

### Permanent Fix

1.  Open **Start Search**, type "env", and select **Edit the system environment variables**.
2.  Click **Environment Variables**.
3.  Under **User variables**, select `Path` and click **Edit**.
4.  Add these two entries:
    - `C:\Users\homeu\.cargo\bin`
    - `C:\Users\homeu\.local\bin`
5.  Restart your terminal.

## 3. Resolving Linker Issues (MSVC to GNU)

If you see an error like `linker link.exe not found`, it means the Visual Studio Build Tools are missing. A faster alternative is to use the GNU toolchain:

### Install GNU Toolchain

```powershell
rustup toolchain install stable-x86_64-pc-windows-gnu
rustup default stable-x86_64-pc-windows-gnu
```

## 4. Project Initialization & Code Fixes

### Creating the Project

```powershell
maturin new rag --bindings pyo3
cd rag
```

## 5. Building and Installing

`maturin develop` requires a Python virtual environment to manage dependencies safely.

### Setup and Build

```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install maturin in the venv (optional but recommended)
pip install maturin

# Build and install the extension as an editable package
maturin develop --release
```

## 6. Verification

Test the installation in Python:

```python
import rag
print(rag.find_primes(10))  # Should output: [2, 3, 5, 7]
```

## Summary of Fixes Applied

1.  **PATH Update**: Added `.cargo/bin` and `.local/bin` to the session environment.
2.  **GNU Fallback**: Switched to the GNU toolchain to avoid missing MSVC dependencies.
3.  **Venv Requirement**: Used a virtual environment for the build process.
4.  **Venv Requirement**: Used a virtual environment for the build process.

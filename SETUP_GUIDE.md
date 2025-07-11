# 🤖 Local RAG System - Setup Guide

A complete guide to set up your privacy-first, local RAG (Retrieval-Augmented Generation) system.

## 📋 Prerequisites

- **Python 3.8+** (recommended: Python 3.10 or 3.11)
- **8GB+ RAM** (for embedding models and local LLM)
- **5GB+ free disk space** (for models and dependencies)
- **Internet connection** (for initial setup only)

## 🚀 Quick Start

### 1. Clone or Download the Project

```bash
# Create project directory
mkdir local-rag-system
cd local-rag-system

# Copy all the provided Python files into this directory
```

### 2. Install Python Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Alternative: Use virtual environment (recommended)
python -m venv rag-env
source rag-env/bin/activate  # On Windows: rag-env\Scripts\activate
pip install -r requirements.txt
```

### 3. Install and Setup Ollama

#### For Windows:

1. Download from [https://ollama.ai/](https://ollama.ai/)
2. Run the installer
3. Open Command Prompt and run:

```cmd
ollama pull mistral
```

#### For macOS:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the Mistral model
ollama pull mistral
```

#### For Linux:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
sudo systemctl start ollama
sudo systemctl enable ollama

# Pull the Mistral model
ollama pull mistral
```

### 4. Verify Installation

```bash
# Test the RAG utilities
python test_rag_utils.py

# Verify Ollama is working
ollama list
```

## 📁 Project Structure

After setup, your directory should look like this:

```
local-rag-system/
├── rag_utils.py          # Core RAG functionality
├── embed_documents.py    # Document embedding script
├── query_rag.py         # Command-line query interface
├── app.py               # Streamlit web interface
├── test_rag_utils.py    # Test suite
├── requirements.txt     # Python dependencies
├── sample.txt           # Sample document
├── SETUP_GUIDE.md       # This file
└── vectorstore/         # Created after first embedding
```

## 🎯 Usage Workflows

### Method 1: Command Line Interface

1. **Embed documents:**

```bash
python embed_documents.py
```

2. **Query documents:**

```bash
python query_rag.py
```

### Method 2: Web Interface (Recommended)

1. **Start the web app:**

```bash
streamlit run app.py
```

2. **Open your browser** to the displayed URL (usually `http://localhost:8501`)

3. **Upload documents** using the sidebar

4. **Ask questions** in the chat interface

## 📄 Supported Document Formats

- **📝 Text files** (`.txt`)
- **📄 PDF files** (`.pdf`) - with OCR support for scanned documents
- **📊 Word documents** (`.docx`)
- **📈 PowerPoint presentations** (`.pptx`)

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. "Import error" when running scripts

**Solution:**

```bash
pip install -r requirements.txt
```

#### 2. "Ollama connection failed"

**Solutions:**

- Ensure Ollama is installed and running
- Try: `ollama serve` (if not running as service)
- Verify model: `ollama list`
- Pull model: `ollama pull mistral`

#### 3. "PDF processing failed"

**Solutions:**

- Install additional dependencies:

```bash
# For Ubuntu/Debian
sudo apt-get install poppler-utils

# For macOS
brew install poppler

# For Windows - pdf2image should work out of the box
```

#### 4. "ChromaDB errors"

**Solution:**

```bash
# Clear any corrupted vectorstore
rm -rf vectorstore/
# Re-run embedding
python embed_documents.py
```

#### 5. "Unicode/Font errors in PDF generation"

**Solution:**

- Download DejaVu fonts:

```bash
# Create fonts directory
mkdir fonts
# Download DejaVuSans.ttf to fonts/ directory
# Or the system will fall back to basic fonts
```

### Performance Optimization

#### For Better Speed:

- Use an SSD for vectorstore storage
- Increase chunk size for fewer, larger chunks
- Use GPU-enabled embeddings (if available)

#### For Lower Memory Usage:

- Reduce chunk size and overlap
- Process fewer documents at once
- Close other applications

## 🔒 Privacy & Security

This system is designed to be completely private:

- ✅ **No data leaves your machine**
- ✅ **No API calls to external services**
- ✅ **No internet required after setup**
- ✅ **Local LLM processing**
- ✅ **Local vector storage**

## ⚙️ Advanced Configuration

### Changing the LLM Model

Edit the model name in your scripts:

```python
# In rag_utils.py and query_rag.py
OLLAMA_MODEL = "llama2"  # or "codellama", "neural-chat", etc.
```

Then pull the new model:

```bash
ollama pull llama2
```

### Adjusting Chunk Sizes

Edit in `rag_utils.py`:

```python
# Larger chunks = more context, fewer chunks
chunk_size = 1000
chunk_overlap = 200

# Smaller chunks = more precise retrieval
chunk_size = 300
chunk_overlap = 50
```

### Custom Embedding Models

Edit in `rag_utils.py`:

```python
# Faster, smaller model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Better quality, larger model
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
```

## 🆘 Getting Help

### If you encounter issues:

1. **Run the test suite:**

   ```bash
   python test_rag_utils.py
   ```

2. **Check the logs** in your terminal for specific error messages

3. **Verify all components:**

   - Python version: `python --version`
   - Ollama status: `ollama list`
   - Dependencies: `pip list | grep langchain`

4. **Try the minimal example:**
   ```bash
   # Just test basic functionality
   python -c "from rag_utils import load_documents; print('✅ Import successful')"
   ```

### System Requirements Check

**Minimum Requirements:**

- 4GB RAM (for basic functionality)
- 2GB free disk space
- Python 3.8+

**Recommended Setup:**

- 8GB+ RAM
- 5GB+ free disk space
- Python 3.10+
- SSD storage

## 🎉 You're Ready!

Once setup is complete, you can:

1. **Upload any supported documents**
2. **Ask questions about their content**
3. **Get answers with source citations**
4. **Export answers as PDFs**
5. **Keep everything completely private**

Enjoy your local RAG system! 🚀

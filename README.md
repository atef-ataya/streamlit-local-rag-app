# 🧠 Local RAG App - Your Private AI Document Assistant

<div align="center">

![Local RAG App Banner](https://github.com/atef-ataya/streamlit-local-rag-app/assets/your-image.png)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

**Build your own ChatGPT-like system that runs 100% offline. No API keys. No data leaves your machine.**

[**📺 Watch the Tutorial**](https://youtube.com/your-link) | [**🚀 Live Demo**](https://your-demo-link) | [**📝 Blog Post**](https://your-blog-link)

</div>

---

## ✨ Features

- 🔒 **100% Private** - Your data never leaves your machine
- 📄 **Multi-Format Support** - PDFs, Word docs, PowerPoints
- 🔍 **OCR Capabilities** - Search even scanned documents
- 💬 **Natural Language Q&A** - Ask questions in plain English
- 📊 **Source Citations** - Know exactly where answers come from
- 💾 **Export Results** - Download Q&A sessions as formatted PDFs
- 🚀 **Fast & Lightweight** - Runs smoothly on consumer hardware

## 🎥 Demo

<div align="center">
  <img src="https://github.com/atef-ataya/streamlit-local-rag-app/assets/demo.gif" alt="Demo GIF" width="600">
</div>

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Ollama (Mistral/Llama2)
- **Embeddings**: Sentence Transformers
- **Vector Store**: ChromaDB
- **Document Processing**: LangChain
- **OCR**: Unstructured

## 📋 Prerequisites

Before you begin, ensure you have:

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed
- At least 8GB RAM (16GB recommended)
- 10GB free disk space

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/atef-ataya/streamlit-local-rag-app.git
cd streamlit-local-rag-app
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install and start Ollama
```bash
# Pull the Mistral model (or any model of your choice)
ollama pull mistral
```

### 5. Run the application
```bash
streamlit run app.py
```

Your app should now be running at `http://localhost:8501` 🎉

## 📖 Usage

1. **Upload Documents**: Drag and drop your PDFs, Word docs, or PowerPoints
2. **Wait for Processing**: Documents are chunked and embedded locally
3. **Ask Questions**: Type natural language questions about your documents
4. **Get Answers**: Receive detailed answers with source citations
5. **Export Results**: Download your Q&A session as a PDF

### Example Questions

- "What are the key findings in this research?"
- "Summarize the main points from the meeting notes"
- "Find all mentions of budget in these documents"
- "What are the action items from the presentation?"

## 🔧 Configuration

You can customize the app by modifying `config.py`:

```python
# Model settings
MODEL_NAME = "mistral"  # or "llama2", "codellama"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# UI settings
MAX_FILE_SIZE = 200  # MB
SUPPORTED_FORMATS = ['.pdf', '.docx', '.pptx']
```

## 🏗️ Project Structure

```
streamlit-local-rag-app/
├── app.py                 # Main Streamlit application
├── components/
│   ├── document_processor.py  # Document handling logic
│   ├── embeddings.py          # Embedding generation
│   ├── qa_chain.py           # Question-answering chain
│   └── ui_components.py      # Reusable UI elements
├── utils/
│   ├── config.py            # Configuration settings
│   └── helpers.py           # Helper functions
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── .gitignore             # Git ignore rules
```

## 🤝 Contributing

Contributions are what make the open source community amazing! Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📝 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

## 🙏 Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain) for the amazing framework
- [Ollama](https://ollama.ai/) for making local LLMs accessible
- [Streamlit](https://streamlit.io/) for the intuitive web framework
- The open-source community for continuous inspiration

## 📬 Contact

Atef Ataya - [@your_twitter](https://twitter.com/your_twitter) - your.email@example.com

Project Link: [https://github.com/atef-ataya/streamlit-local-rag-app](https://github.com/atef-ataya/streamlit-local-rag-app)

---

<div align="center">

**If you found this helpful, please ⭐ this repository!**

</div>

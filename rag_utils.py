
import os
import tempfile
import logging
from typing import List, Tuple, Dict, Any
from io import BytesIO
import textwrap

# Core LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Document loaders
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
)

# PDF processing
try:
    from pdf2image import convert_from_path
    PDF_PREVIEW_AVAILABLE = True
except ImportError:
    PDF_PREVIEW_AVAILABLE = False
    logging.warning("pdf2image not available. PDF previews disabled.")

# PDF generation
try:
    from fpdf import FPDF
    PDF_GENERATION_AVAILABLE = True
except ImportError:
    PDF_GENERATION_AVAILABLE = False
    logging.warning("fpdf2 not available. PDF export disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"

# Constants
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = "vectorstore"
SUPPORTED_EXTENSIONS = [".txt", ".pdf", ".docx", ".pptx"]

class SafePDF(FPDF):
    """Enhanced FPDF class with better Unicode and error handling"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_auto_page_break(auto=True, margin=15)
        self.unicode_font_loaded = False
        self._setup_font()
    
    def _setup_font(self):
        """Setup Unicode-compatible font with fallback"""
        try:
            # Try to use DejaVu font for Unicode support
            font_path = self._find_font_path()
            if font_path:
                self.add_font("DejaVu", "", font_path, uni=True)
                # Try to add bold variant if available
                bold_path = font_path.replace("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")
                if os.path.exists(bold_path):
                    self.add_font("DejaVu", "B", bold_path, uni=True)
                    logger.info("Bold DejaVu font loaded successfully")
                else:
                    logger.info("Bold DejaVu font not found, will use regular font for bold text")
                
                self.set_font("DejaVu", size=11)
                self.unicode_font_loaded = True
                logger.info("Unicode font loaded successfully")
            else:
                raise FileNotFoundError("DejaVu font not found")
        except Exception as e:
            logger.warning(f"Unicode font loading failed: {e}. Using Helvetica fallback.")
            self.set_font("Helvetica", size=11)
            self.unicode_font_loaded = False
    
    def _find_font_path(self):
        """Find DejaVu font in common locations"""
        possible_paths = [
            "DejaVuSans.ttf",  # Root directory
            "fonts/DejaVuSans.ttf",  # fonts subdirectory
            "./DejaVuSans.ttf",  # Current directory
            "./fonts/DejaVuSans.ttf",  # Current directory fonts
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "/System/Library/Fonts/DejaVuSans.ttf",  # macOS
            "C:\\Windows\\Fonts\\DejaVuSans.ttf"  # Windows
        ]
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found DejaVu font at: {path}")
                return path
        logger.warning("DejaVu font not found in any expected location")
        return None
    
    def safe_text(self, text: str) -> str:
        """Clean and encode text safely for PDF rendering"""
        if not text:
            return ""
        
        # Unicode replacements for common problematic characters
        replacements = {
            '\u2018': "'", '\u2019': "'",  # Smart quotes
            '\u201c': '"', '\u201d': '"',  # Smart double quotes
            '\u2013': '-', '\u2014': '--', # En/Em dashes
            '\u2026': '...', # Ellipsis
            '\u2028': ' ', '\u2029': ' ',  # Line/paragraph separators
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Handle encoding based on font capability
        if self.unicode_font_loaded:
            return text
        else:
            # Fallback to ASCII-safe encoding
            return text.encode('ascii', 'ignore').decode('ascii')
    
    def add_title(self, title: str):
        """Add a title to the PDF"""
        try:
            self.set_font(self.font_family, 'B', 16)
        except:
            # Fallback if bold not available
            self.set_font(self.font_family, '', 16)
        self.cell(0, 10, self.safe_text(title), ln=True, align='C')
        self.ln(5)
    
    def add_section(self, title: str, content: str):
        """Add a section with title and content"""
        # Section title
        try:
            self.set_font(self.font_family, 'B', 14)
        except:
            # Fallback if bold not available
            self.set_font(self.font_family, '', 14)
        self.cell(0, 10, self.safe_text(title), ln=True)
        self.ln(2)
        
        # Section content
        self.set_font(self.font_family, '', 11)
        self._write_wrapped_text(content)
        self.ln(5)
    
    def _write_wrapped_text(self, text: str):
        """Write text with proper wrapping and page breaks"""
        safe_text = self.safe_text(text)
        paragraphs = safe_text.split('\n')
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                self.ln()
                continue
            
            # Use multi_cell with automatic wrapping instead of manual textwrap
            # Check if we need a new page before writing
            if self.get_y() > 250:
                self.add_page()
            
            try:
                # Let FPDF handle the wrapping automatically
                self.multi_cell(0, 6, paragraph, align='L')
            except Exception as e:
                logger.warning(f"Failed to write paragraph: {e}")
                # Ultra-safe fallback with shorter text
                try:
                    # Try with ASCII-only text
                    safe_paragraph = ''.join(c for c in paragraph if ord(c) < 128)
                    if safe_paragraph.strip():
                        self.multi_cell(0, 6, safe_paragraph, align='L')
                    else:
                        self.multi_cell(0, 6, "[Content contains unsupported characters]", align='L')
                except Exception as e2:
                    logger.error(f"Even fallback failed: {e2}")
                    # Last resort - just add a placeholder
                    try:
                        self.cell(0, 6, "[Content unavailable]", ln=True)
                    except:
                        pass  # Give up on this content

def load_documents(file_path: str) -> List:
    """
    Load documents from a file path.
    Supports .txt, .pdf, .docx, .pptx files.
    
    Args:
        file_path (str): Path to the document file
        
    Returns:
        List: List of loaded documents
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file type is not supported
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_extension = os.path.splitext(file_path.lower())[1]
    
    if file_extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {file_extension}. Supported: {SUPPORTED_EXTENSIONS}")
    
    try:
        if file_extension == ".txt":
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_extension == ".pdf":
            # Try OCR-enabled loader first, fallback to basic PDF
            try:
                loader = UnstructuredPDFLoader(file_path)
                docs = loader.load()
                logger.info(f"Loaded PDF with OCR: {file_path}")
                return docs
            except Exception as e:
                logger.warning(f"OCR failed for {file_path}: {e}. Trying basic PDF loader.")
                loader = PyPDFLoader(file_path)
        elif file_extension == ".docx":
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_extension == ".pptx":
            loader = UnstructuredPowerPointLoader(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
        docs = loader.load()
        logger.info(f"Successfully loaded {len(docs)} document(s) from {file_path}")
        return docs
        
    except Exception as e:
        logger.error(f"Error loading document {file_path}: {e}")
        raise

def split_documents(docs: List, chunk_size: int = 500, chunk_overlap: int = 100) -> List:
    """
    Split documents into chunks for better retrieval.
    
    Args:
        docs (List): List of documents to split
        chunk_size (int): Maximum size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        List: List of document chunks
    """
    if not docs:
        logger.warning("No documents provided for splitting")
        return []
    
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(docs)
        logger.info(f"Split {len(docs)} documents into {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        logger.error(f"Error splitting documents: {e}")
        raise

def process_uploaded_files(uploaded_files) -> Tuple[List, Dict, Dict, Dict, Any]:
    """
    Process uploaded files from Streamlit and return processed data.
    
    Returns:
        Tuple: (docs, chunk_map, ocr_status, preview_images, vectorstore)
    """
    docs = []
    chunk_map = {}
    ocr_status = {}
    preview_images = {}
    vectorstore = None
    
    if not uploaded_files:
        logger.warning("No files uploaded")
        return docs, chunk_map, ocr_status, preview_images, vectorstore
    
    for file in uploaded_files:
        try:
            filename = file.name
            file_extension = os.path.splitext(filename.lower())[1]
            
            if file_extension not in SUPPORTED_EXTENSIONS:
                logger.warning(f"Skipping unsupported file: {filename}")
                ocr_status[filename] = f"❌ Unsupported format: {file_extension}"
                continue
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name
            
            try:
                # Process based on file type
                if file_extension == ".pdf":
                    file_docs, ocr_stat, preview = process_pdf(tmp_path, filename)
                    ocr_status[filename] = ocr_stat
                    preview_images[filename] = preview
                elif file_extension == ".docx":
                    loader = UnstructuredWordDocumentLoader(tmp_path)
                    file_docs = loader.load()
                    ocr_status[filename] = "✅ Word Document"
                    preview_images[filename] = None
                elif file_extension == ".pptx":
                    loader = UnstructuredPowerPointLoader(tmp_path)
                    file_docs = loader.load()
                    ocr_status[filename] = "✅ PowerPoint"
                    preview_images[filename] = None
                elif file_extension == ".txt":
                    loader = TextLoader(tmp_path, encoding='utf-8')
                    file_docs = loader.load()
                    ocr_status[filename] = "✅ Text File"
                    preview_images[filename] = None
                else:
                    continue
                
                # Split into chunks
                chunks = split_documents(file_docs)
                
                # Add source metadata
                for chunk in chunks:
                    chunk.metadata["source"] = filename
                
                docs.extend(chunks)
                chunk_map[filename] = [chunk.page_content for chunk in chunks]
                
                logger.info(f"Processed {filename}: {len(chunks)} chunks")
                
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            ocr_status[filename] = f"❌ Error: {str(e)}"
            continue
    
    # Create vectorstore if we have documents
    if docs:
        try:
            embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=None  # Use in-memory for uploaded files
            )
            logger.info(f"Created vectorstore with {len(docs)} document chunks")
        except Exception as e:
            logger.error(f"Error creating vectorstore: {e}")
            raise
    else:
        logger.warning("No documents processed successfully")
    
    return docs, chunk_map, ocr_status, preview_images, vectorstore

def process_pdf(tmp_path: str, filename: str) -> Tuple[List, str, Any]:
    """Process PDF file with OCR fallback and preview generation"""
    preview = None
    
    # Try OCR-enabled processing first
    try:
        loader = UnstructuredPDFLoader(tmp_path)
        file_docs = loader.load()
        ocr_stat = "✅ OCR Enabled"
        logger.info(f"Processed {filename} with OCR")
    except Exception as e:
        logger.warning(f"OCR failed for {filename}: {e}")
        try:
            # Fallback to basic PDF extraction
            loader = PyPDFLoader(tmp_path)
            file_docs = loader.load()
            ocr_stat = "⚠️ Text Extraction Only"
            logger.info(f"Processed {filename} with basic PDF loader")
        except Exception as e2:
            logger.error(f"Both PDF loaders failed for {filename}: {e2}")
            raise
    
    # Try to generate preview
    if PDF_PREVIEW_AVAILABLE:
        try:
            images = convert_from_path(tmp_path, first_page=1, last_page=1, dpi=150)
            if images:
                preview = images[0]
                logger.info(f"Generated preview for {filename}")
        except Exception as e:
            logger.warning(f"Could not generate preview for {filename}: {e}")
    
    return file_docs, ocr_stat, preview

def get_vectorstore(persist_directory: str = VECTORSTORE_DIR):
    """Get existing vectorstore from disk"""
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"Vectorstore directory not found: {persist_directory}")
    
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vectorstore

def query_documents(query: str, vectorstore) -> Tuple[str, List]:
    """
    Query documents using the vectorstore and return answer with sources.
    
    Args:
        query (str): User question
        vectorstore: ChromaDB vectorstore instance
        
    Returns:
        Tuple: (answer, source_documents)
    """
    if not vectorstore:
        return "❌ No vectorstore available. Please upload documents first.", []
    
    if not query.strip():
        return "❌ Please enter a valid question.", []
    
    try:
        # Initialize LLM
        llm = OllamaLLM(model="mistral", temperature=0.3)
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful assistant analyzing uploaded documents. 
Use ONLY the following context from the uploaded documents to answer the question.
Do not use your general knowledge unless the context is insufficient.
If the answer cannot be found in the provided context, state this clearly.

Context from uploaded documents:
{context}

Question: {question}

Answer based on the uploaded documents:"""
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )
        
        # Execute query
        result = qa_chain.invoke({"query": query})
        answer = result.get("result", "No answer found.")
        sources = result.get("source_documents", [])
        
        logger.info(f"Query processed successfully. Found {len(sources)} source documents.")
        return answer, sources
        
    except Exception as e:
        logger.error(f"Error querying documents: {e}")
        return f"❌ Error processing query: {str(e)}", []

def save_answer_as_file(answer: str, sources: List, query: str = "") -> bytes:
    """
    Save answer and sources as a PDF file.
    
    Args:
        answer (str): The generated answer
        sources (List): Source documents used
        query (str): Original user query
        
    Returns:
        bytes: PDF file content as bytes
    """
    if not PDF_GENERATION_AVAILABLE:
        # Fallback to text format
        text_content = f"QUERY: {query}\n\n"
        text_content += f"ANSWER:\n{answer}\n\n"
        text_content += "SOURCES:\n"
        for i, doc in enumerate(sources, 1):
            text_content += f"\n--- Source {i} ---\n"
            text_content += f"File: {doc.metadata.get('source', 'Unknown')}\n"
            text_content += f"Content: {doc.page_content[:500]}...\n"
        return text_content.encode('utf-8')
    
    try:
        pdf = SafePDF(orientation='P', unit='mm', format='A4')
        pdf.add_page()
        
        # Set proper margins (left, top, right)
        pdf.set_margins(20, 20, 20)
        pdf.set_auto_page_break(auto=True, margin=20)
        
        # Add title
        pdf.add_title("RAG Query Results")
        
        # Add query if provided
        if query:
            pdf.add_section("Query:", query)
        
        # Add answer
        pdf.add_section("Answer:", answer)
        
        # Add sources
        if sources:
            pdf.add_section("Source Documents:", "")
            
            for i, doc in enumerate(sources, 1):
                if pdf.get_y() > 240:  # Check if we need a new page
                    pdf.add_page()
                
                source_name = doc.metadata.get("source", "Unknown Source")
                content = doc.page_content
                
                # Limit content length for readability
                if len(content) > 800:
                    content = content[:800] + "..."
                
                pdf.add_section(f"Source {i} - {source_name}:", content)
        
        # Generate PDF with correct output method
        return bytes(pdf.output())
            
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        # Fallback to text
        fallback_content = f"Query: {query}\n\nAnswer: {answer}\n\nSources:\n"
        for i, doc in enumerate(sources, 1):
            fallback_content += f"\nSource {i}: {doc.page_content[:300]}...\n"
        return fallback_content.encode('utf-8', errors='ignore')

def get_file_info(vectorstore_dir: str = VECTORSTORE_DIR) -> Dict[str, Any]:
    """Get information about the stored vectorstore"""
    info = {
        "vectorstore_exists": os.path.exists(vectorstore_dir),
        "total_chunks": 0,
        "status": "Unknown"
    }
    
    if info["vectorstore_exists"]:
        try:
            vectorstore = get_vectorstore(vectorstore_dir)
            collection_data = vectorstore._collection.get()
            info["total_chunks"] = len(collection_data["ids"])
            info["status"] = "✅ Ready"
            logger.info(f"Vectorstore contains {info['total_chunks']} chunks")
        except Exception as e:
            logger.error(f"Error accessing vectorstore: {e}")
            info["status"] = f"❌ Error: {str(e)}"
    else:
        info["status"] = "⚠️ Not Found"
    
    return info



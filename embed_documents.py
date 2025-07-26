#!/usr/bin/env python3
"""
Document Embedding Script for Local RAG System

This script loads documents, splits them into chunks, and creates a persistent
vectorstore using ChromaDB and HuggingFace embeddings.

Usage:
    python embed_documents.py

Requirements:
    - Document file specified in DOC_PATH
    - All dependencies from requirements.txt installed
"""

import os
import sys
from pathlib import Path

# Suppress LangChain tracking
os.environ["LANGCHAIN_ENDPOINT"] = "none"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

try:
    from rag_utils import load_documents, split_documents
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)

# Configuration
DOC_PATH = "sample.txt"  # Update this with your document path
VECTORSTORE_DIR = "vectorstore"  # Directory to store the vector store
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def validate_document_path(doc_path: str) -> bool:
    """Validate that the document path exists and is supported"""
    if not os.path.exists(doc_path):
        print(f"❌ Document not found: {doc_path}")
        return False
    
    supported_extensions = [".txt", ".pdf", ".docx", ".pptx"]
    file_extension = os.path.splitext(doc_path.lower())[1]
    
    if file_extension not in supported_extensions:
        print(f"❌ Unsupported file type: {file_extension}")
        print(f"Supported formats: {', '.join(supported_extensions)}")
        return False
    
    return True

def create_vectorstore_directory(directory: str) -> None:
    """Create vectorstore directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"📁 Created vectorstore directory: {directory}")
    else:
        print(f"📁 Using existing vectorstore directory: {directory}")

def main():
    """Main embedding process"""
    print("🚀 Starting document embedding process...")
    print(f"📄 Document: {DOC_PATH}")
    print(f"📁 Vectorstore: {VECTORSTORE_DIR}")
    print("-" * 50)
    
    # Step 1: Validate document path
    if not validate_document_path(DOC_PATH):
        print("\n💡 Tips:")
        print("- Check that the file path is correct")
        print("- Ensure the file exists in the current directory")
        print("- Supported formats: .txt, .pdf, .docx, .pptx")
        return False
    
    # Step 2: Load the document
    print(f"📖 Loading document: {DOC_PATH}")
    try:
        docs = load_documents(DOC_PATH)
        print(f"✅ Successfully loaded {len(docs)} document(s)")
        
        # Display document info
        total_chars = sum(len(doc.page_content) for doc in docs)
        print(f"📊 Total characters: {total_chars:,}")
        
    except Exception as e:
        print(f"❌ Failed to load document: {e}")
        print("\n💡 Troubleshooting:")
        print("- Check file permissions")
        print("- For PDFs: ensure they contain extractable text")
        print("- For Office docs: ensure they're not corrupted")
        return False

    # Step 3: Split the document into chunks
    print(f"\n🔪 Splitting documents into chunks...")
    try:
        chunks = split_documents(docs)
        print(f"✅ Document split into {len(chunks)} chunks")
        
        # Display chunk statistics
        if chunks:
            avg_chunk_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
            print(f"📊 Average chunk size: {avg_chunk_size:.0f} characters")
            
            # Show sample chunk
            print(f"\n📝 Sample chunk preview:")
            sample_content = chunks[0].page_content[:200]
            print(f"'{sample_content}{'...' if len(chunks[0].page_content) > 200 else ''}'")
        
    except Exception as e:
        print(f"❌ Failed to split documents: {e}")
        return False

    # Step 4: Create vectorstore directory
    print(f"\n📁 Preparing vectorstore directory...")
    try:
        create_vectorstore_directory(VECTORSTORE_DIR)
    except Exception as e:
        print(f"❌ Failed to create directory: {e}")
        return False

    # Step 5: Initialize embeddings
    print(f"\n🧠 Initializing embeddings model: {MODEL_NAME}")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        print("✅ Embeddings model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load embeddings model: {e}")
        print("\n💡 This might be a first-time download. Please wait...")
        return False

    # Step 6: Create and persist the vector store
    print(f"\n💾 Creating vectorstore...")
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=VECTORSTORE_DIR
        )
        
        # Persist the vectorstore
        vectorstore.persist()
        
        print(f"✅ Vectorstore created and persisted to '{VECTORSTORE_DIR}'")
        
        # Verify the vectorstore
        collection_data = vectorstore._collection.get()
        stored_count = len(collection_data["ids"])
        print(f"📊 Verified: {stored_count} document chunks stored")
        
    except Exception as e:
        print(f"❌ Failed to create vectorstore: {e}")
        print("\n💡 Possible issues:")
        print("- Insufficient disk space")
        print("- Permission issues with the directory")
        print("- ChromaDB installation problems")
        return False

    # Step 7: Success summary
    print("\n" + "=" * 50)
    print("🎉 EMBEDDING PROCESS COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"📄 Document processed: {DOC_PATH}")
    print(f"📊 Total chunks created: {len(chunks)}")
    print(f"💾 Vectorstore location: {VECTORSTORE_DIR}")
    print(f"🧠 Embedding model: {MODEL_NAME}")
    
    print(f"\n🚀 Next steps:")
    print(f"- Run 'python query_rag.py' to query your documents")
    print(f"- Or run 'streamlit run app.py' for the web interface")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print(f"\n✅ Embedding process completed successfully!")
            sys.exit(0)
        else:
            print(f"\n❌ Embedding process failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print("Please check the error message above and try again.")
        sys.exit(1)
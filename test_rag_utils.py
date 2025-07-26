#!/usr/bin/env python3
"""
Test Script for RAG Utils

This script tests the core functionality of the RAG system including
document loading, splitting, and basic processing.

Usage:
    python test_rag_utils.py

Requirements:
    - Sample document file (sample.txt by default)
    - All dependencies from requirements.txt installed
"""

import os
import sys
from pathlib import Path

# Suppress LangChain tracking
os.environ["LANGCHAIN_ENDPOINT"] = "none"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

try:
    from rag_utils import load_documents, split_documents, get_file_info
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)

def test_file_loading():
    """Test document loading functionality"""
    print("🧪 Testing document loading...")
    
    # Test files to try
    test_files = [
        "sample.txt",
        "sample.pdf",
        "test_document.txt",
        "README.md"
    ]
    
    loaded_files = []
    
    for file_path in test_files:
        if os.path.exists(file_path):
            try:
                print(f"📄 Testing: {file_path}")
                docs = load_documents(file_path)
                
                if docs:
                    print(f"✅ Successfully loaded {len(docs)} document(s)")
                    
                    # Show document info
                    total_chars = sum(len(doc.page_content) for doc in docs)
                    print(f"   📊 Total characters: {total_chars:,}")
                    
                    # Show sample content
                    if docs[0].page_content:
                        sample = docs[0].page_content[:100]
                        print(f"   📝 Sample: '{sample}{'...' if len(docs[0].page_content) > 100 else ''}'")
                    
                    loaded_files.append((file_path, docs))
                else:
                    print(f"⚠️ No content loaded from {file_path}")
                    
            except Exception as e:
                print(f"❌ Failed to load {file_path}: {e}")
        else:
            print(f"⏭️ Skipping {file_path} (not found)")
    
    return loaded_files

def test_document_splitting(loaded_files):
    """Test document splitting functionality"""
    print(f"\n🧪 Testing document splitting...")
    
    if not loaded_files:
        print("⚠️ No loaded files to test splitting")
        return []
    
    split_results = []
    
    for file_path, docs in loaded_files:
        try:
            print(f"🔪 Splitting: {file_path}")
            
            # Test different chunk sizes
            test_configs = [
                {"chunk_size": 500, "chunk_overlap": 100},
                {"chunk_size": 1000, "chunk_overlap": 200},
                {"chunk_size": 200, "chunk_overlap": 50}
            ]
            
            for config in test_configs:
                chunks = split_documents(docs, **config)
                
                if chunks:
                    print(f"   ✅ Config {config}: {len(chunks)} chunks")
                    
                    # Analyze chunk sizes
                    chunk_sizes = [len(chunk.page_content) for chunk in chunks]
                    avg_size = sum(chunk_sizes) / len(chunk_sizes)
                    max_size = max(chunk_sizes)
                    min_size = min(chunk_sizes)
                    
                    print(f"      📊 Avg: {avg_size:.0f}, Max: {max_size}, Min: {min_size}")
                    
                    # Store best result (medium chunk size)
                    if config["chunk_size"] == 500:
                        split_results.append((file_path, chunks))
                        
                        # Show sample chunks
                        print(f"      📝 Sample chunks:")
                        for i, chunk in enumerate(chunks[:3]):
                            preview = chunk.page_content[:80]
                            print(f"         Chunk {i+1}: '{preview}{'...' if len(chunk.page_content) > 80 else ''}'")
                        
                        if len(chunks) > 3:
                            print(f"         ... and {len(chunks)-3} more chunks")
                else:
                    print(f"   ❌ Config {config}: No chunks created")
                    
        except Exception as e:
            print(f"❌ Failed to split {file_path}: {e}")
    
    return split_results

def test_vectorstore_info():
    """Test vectorstore information retrieval"""
    print(f"\n🧪 Testing vectorstore info...")
    
    try:
        info = get_file_info()
        print(f"📊 Vectorstore Info:")
        print(f"   Exists: {info['vectorstore_exists']}")
        print(f"   Status: {info.get('status', 'Unknown')}")
        
        if 'total_chunks' in info:
            print(f"   Total chunks: {info['total_chunks']}")
        
        if info['vectorstore_exists']:
            print("✅ Vectorstore is accessible")
        else:
            print("ℹ️ No existing vectorstore found (run embed_documents.py to create one)")
            
    except Exception as e:
        print(f"❌ Error testing vectorstore info: {e}")

def test_supported_formats():
    """Test supported file format detection"""
    print(f"\n🧪 Testing supported formats...")
    
    test_files = [
        "test.txt", "test.pdf", "test.docx", "test.pptx",
        "test.xlsx", "test.jpg", "test.mp4", "test.unknown"
    ]
    
    from rag_utils import SUPPORTED_EXTENSIONS
    
    print(f"📋 Supported extensions: {SUPPORTED_EXTENSIONS}")
    
    for test_file in test_files:
        extension = os.path.splitext(test_file.lower())[1]
        is_supported = extension in SUPPORTED_EXTENSIONS
        status = "✅" if is_supported else "❌"
        print(f"   {status} {test_file} ({extension})")

def create_test_document():
    """Create a test document if none exists"""
    test_content = """# Test Document for RAG System

This is a test document created automatically for testing the RAG system functionality.

## Section 1: Introduction
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.

## Section 2: Technical Details
The RAG (Retrieval-Augmented Generation) system combines:
- Document processing and chunking
- Vector embeddings using sentence transformers
- ChromaDB for vector storage
- Local LLM inference via Ollama

## Section 3: Features
Key features include:
1. Multi-format document support (PDF, DOCX, PPTX, TXT)
2. OCR capability for scanned documents
3. Local processing for privacy
4. Web interface with Streamlit
5. PDF export of answers

## Section 4: Conclusion
This test document contains enough text to demonstrate chunking, embedding, and retrieval capabilities of the RAG system. The content is structured to test various query types.
"""
    
    test_file = "test_document.txt"
    
    if not os.path.exists(test_file):
        try:
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_content)
            print(f"📝 Created test document: {test_file}")
            return test_file
        except Exception as e:
            print(f"❌ Failed to create test document: {e}")
            return None
    else:
        print(f"📄 Test document already exists: {test_file}")
        return test_file

def run_comprehensive_test():
    """Run all tests in sequence"""
    print("🚀 Starting comprehensive RAG utils test...")
    print("=" * 60)
    
    # Test 1: Check supported formats
    test_supported_formats()
    
    # Test 2: Create test document if needed
    create_test_document()
    
    # Test 3: Test file loading
    loaded_files = test_file_loading()
    
    # Test 4: Test document splitting
    split_results = test_document_splitting(loaded_files)
    
    # Test 5: Test vectorstore info
    test_vectorstore_info()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Files successfully loaded: {len(loaded_files)}")
    print(f"✅ Files successfully split: {len(split_results)}")
    
    if loaded_files:
        print(f"\n📄 Processed files:")
        for file_path, docs in loaded_files:
            print(f"   - {file_path}: {len(docs)} documents")
    
    if split_results:
        print(f"\n🔪 Split results:")
        for file_path, chunks in split_results:
            print(f"   - {file_path}: {len(chunks)} chunks")
    
    # Recommendations
    print(f"\n💡 Next steps:")
    if loaded_files:
        print("   - Run 'python embed_documents.py' to create vectorstore")
        print("   - Run 'python query_rag.py' to test querying")
        print("   - Run 'streamlit run app.py' for web interface")
    else:
        print("   - Ensure you have a valid document file (sample.txt, etc.)")
        print("   - Check file permissions and formats")
    
    return len(loaded_files) > 0 and len(split_results) > 0

def main():
    """Main test function"""
    try:
        success = run_comprehensive_test()
        
        if success:
            print(f"\n🎉 All tests completed successfully!")
            return True
        else:
            print(f"\n⚠️ Some tests failed. Check the output above.")
            return False
            
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        return False

if __name__ == "__main__":
    try:
        print("🧪 RAG Utils Test Suite")
        print("=" * 60)
        
        success = main()
        
        if success:
            print(f"\n✅ Test suite completed successfully!")
            sys.exit(0)
        else:
            print(f"\n❌ Test suite completed with issues!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
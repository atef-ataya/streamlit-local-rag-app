# import os
# os.environ["LANGCHAIN_ENDPOINT"] = "none"

# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from langchain_ollama import OllamaLLM
# from langchain.chains import RetrievalQA

# VECTORSTORE_DIR = "vectorstore"

# def main():
#     print("[*] Loading local vectorstore...")
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vectorstore = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)

#     print("[*] Launching local LLM (Mistral via Ollama)...")
#     llm = OllamaLLM(model="mistral")

#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         return_source_documents=True
#     )

#     while True:
#         query = input("\n🔍 Ask a question (or 'q' to quit): ")
#         if query.lower() in ["q", "quit", "exit"]:
#             break

#         result = qa_chain.invoke({"query": query})
#         print("\n🧠 Answer:\n", result.get("result", "No answer returned."))

#         sources = result.get("source_documents", [])
#         if sources:
#             print("\n📄 Source Documents:")
#             for i, doc in enumerate(sources, 1):
#                 print(f"\n--- Source {i} ---")
#                 print(doc.page_content)
#         else:
#             print("\n⚠️ No source documents found.")

# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
RAG Query Interface - Command Line Version

This script provides a command-line interface for querying documents
using the pre-built vectorstore and local LLM via Ollama.

Usage:
    python query_rag.py

Requirements:
    - Vectorstore created by embed_documents.py
    - Ollama running with Mistral model
    - All dependencies from requirements.txt installed
"""

import os
import sys
from pathlib import Path

# Suppress LangChain tracking
os.environ["LANGCHAIN_ENDPOINT"] = "none"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

try:
    from rag_utils import get_vectorstore, query_documents, get_file_info
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaLLM
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)

# Configuration
VECTORSTORE_DIR = "vectorstore"
OLLAMA_MODEL = "mistral"

def check_ollama_connection() -> bool:
    """Check if Ollama is running and model is available"""
    try:
        print("🔍 Checking Ollama connection...")
        llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.1)
        
        # Test with a simple query
        test_response = llm.invoke("Hello")
        print(f"✅ Ollama connection successful")
        print(f"🤖 Using model: {OLLAMA_MODEL}")
        return True
        
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("\n💡 Troubleshooting steps:")
        print("1. Install Ollama: https://ollama.ai/")
        print("2. Start Ollama service")
        print(f"3. Pull the model: ollama pull {OLLAMA_MODEL}")
        print("4. Verify: ollama list")
        return False

def check_vectorstore() -> bool:
    """Check if vectorstore exists and is accessible"""
    try:
        print(f"📁 Checking vectorstore: {VECTORSTORE_DIR}")
        
        if not os.path.exists(VECTORSTORE_DIR):
            print(f"❌ Vectorstore directory not found: {VECTORSTORE_DIR}")
            print("\n💡 Please run 'python embed_documents.py' first to create the vectorstore")
            return False
        
        # Get vectorstore info
        info = get_file_info(VECTORSTORE_DIR)
        
        if info["vectorstore_exists"]:
            print(f"✅ Vectorstore found")
            print(f"📊 Status: {info['status']}")
            print(f"📄 Total chunks: {info['total_chunks']}")
            return True
        else:
            print(f"❌ Vectorstore not accessible")
            return False
            
    except Exception as e:
        print(f"❌ Error checking vectorstore: {e}")
        return False

def display_help():
    """Display help information for the query interface"""
    print("\n" + "=" * 60)
    print("🔍 RAG QUERY INTERFACE - HELP")
    print("=" * 60)
    print("Commands:")
    print("  • Type your question and press Enter")
    print("  • 'help' or '?' - Show this help")
    print("  • 'info' - Show vectorstore information")
    print("  • 'examples' - Show example queries")
    print("  • 'clear' - Clear the screen")
    print("  • 'quit', 'exit', or 'q' - Exit the program")
    print("\nTips:")
    print("  • Be specific in your questions")
    print("  • Ask about content from your uploaded documents")
    print("  • The system will show source documents used")
    print("=" * 60)

def display_examples():
    """Display example queries"""
    print("\n" + "=" * 60)
    print("💡 EXAMPLE QUERIES")
    print("=" * 60)
    print("Here are some example questions you can ask:")
    print("\n📄 Content Questions:")
    print("  • What is the main topic of this document?")
    print("  • Summarize the key points")
    print("  • What does the document say about [topic]?")
    print("\n🔍 Specific Information:")
    print("  • Who are the authors mentioned?")
    print("  • What are the main conclusions?")
    print("  • List the important dates/numbers mentioned")
    print("\n📊 Analysis Questions:")
    print("  • What are the pros and cons discussed?")
    print("  • How does this relate to [concept]?")
    print("  • What evidence is provided for [claim]?")
    print("=" * 60)

def format_sources(sources):
    """Format source documents for display"""
    if not sources:
        return "⚠️ No source documents found."
    
    output = []
    for i, doc in enumerate(sources, 1):
        source_name = doc.metadata.get("source", "Unknown")
        content_preview = doc.page_content[:200]
        if len(doc.page_content) > 200:
            content_preview += "..."
        
        output.append(f"\n📄 Source {i} - {source_name}:")
        output.append(f"   {content_preview}")
        output.append("-" * 50)
    
    return "\n".join(output)

def main():
    """Main query interface"""
    print("🚀 RAG QUERY INTERFACE")
    print("=" * 60)
    
    # Step 1: Check Ollama connection
    if not check_ollama_connection():
        return False
    
    # Step 2: Check vectorstore
    if not check_vectorstore():
        return False
    
    # Step 3: Load vectorstore
    print(f"\n💾 Loading vectorstore...")
    try:
        vectorstore = get_vectorstore(VECTORSTORE_DIR)
        print("✅ Vectorstore loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load vectorstore: {e}")
        return False
    
    # Step 4: Start query loop
    print("\n" + "=" * 60)
    print("🎉 READY FOR QUERIES!")
    print("=" * 60)
    print("Type 'help' for commands or 'examples' for sample queries")
    print("Type 'quit' to exit")
    
    query_count = 0
    
    while True:
        try:
            # Get user input
            print(f"\n🔍 Query #{query_count + 1}")
            query = input("Ask a question (or 'q' to quit): ").strip()
            
            # Handle commands
            if query.lower() in ["q", "quit", "exit"]:
                print("👋 Goodbye!")
                break
            elif query.lower() in ["help", "?"]:
                display_help()
                continue
            elif query.lower() == "info":
                info = get_file_info(VECTORSTORE_DIR)
                print(f"\n📊 Vectorstore Info:")
                print(f"   Directory: {VECTORSTORE_DIR}")
                print(f"   Status: {info['status']}")
                print(f"   Total chunks: {info['total_chunks']}")
                continue
            elif query.lower() == "examples":
                display_examples()
                continue
            elif query.lower() == "clear":
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            elif not query:
                print("⚠️ Please enter a question")
                continue
            
            # Process query
            print("🤔 Processing your question...")
            
            try:
                answer, sources = query_documents(query, vectorstore)
                query_count += 1
                
                # Display results
                print("\n" + "=" * 60)
                print("🧠 ANSWER:")
                print("=" * 60)
                print(answer)
                
                if sources:
                    print("\n" + "=" * 60)
                    print("📚 SOURCE DOCUMENTS:")
                    print("=" * 60)
                    print(format_sources(sources))
                else:
                    print("\n⚠️ No source documents found for this query")
                
                # Query statistics
                print(f"\n📊 Query completed in response #{query_count}")
                print(f"📄 Sources used: {len(sources)}")
                
            except Exception as e:
                print(f"❌ Error processing query: {e}")
                print("💡 Try rephrasing your question or check the logs")
                
        except KeyboardInterrupt:
            print(f"\n⚠️ Query interrupted by user")
            continue
        except EOFError:
            print(f"\n👋 Session ended")
            break
    
    print(f"\n📊 Session Summary:")
    print(f"   Total queries processed: {query_count}")
    print(f"   Vectorstore: {VECTORSTORE_DIR}")
    print(f"   Model used: {OLLAMA_MODEL}")
    
    return True

if __name__ == "__main__":
    try:
        print("🎯 Starting RAG Query Interface...")
        success = main()
        
        if success:
            print(f"\n✅ Query session completed successfully!")
            sys.exit(0)
        else:
            print(f"\n❌ Query session failed to start!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Program interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print("Please check the error message above and try again.")
        sys.exit(1)
#!/usr/bin/env python3
"""
RAG System - Streamlit Web Interface

A complete web interface for the local RAG system with document upload,
processing, querying, and answer export capabilities.

Usage:
    streamlit run app.py

Requirements:
    - Ollama running with Mistral model
    - All dependencies from requirements.txt installed
"""

import streamlit as st
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Any, Optional

# Suppress warnings and tracking
import warnings
warnings.filterwarnings("ignore")
os.environ["LANGCHAIN_ENDPOINT"] = "none"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

try:
    from rag_utils import (
        process_uploaded_files, query_documents, save_answer_as_file,
        get_vectorstore, get_file_info
    )
    from langchain_ollama import OllamaLLM
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.error("Please ensure all dependencies are installed: `pip install -r requirements.txt`")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Local RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
OLLAMA_MODEL = "mistral"
SUPPORTED_FORMATS = [".pdf", ".docx", ".pptx", ".txt"]
MAX_FILE_SIZE = 10  # MB

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = {}
    if "ocr_status" not in st.session_state:
        st.session_state.ocr_status = {}
    if "preview_images" not in st.session_state:
        st.session_state.preview_images = {}
    if "chunk_map" not in st.session_state:
        st.session_state.chunk_map = {}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "ollama_status" not in st.session_state:
        st.session_state.ollama_status = None

def check_ollama_status() -> bool:
    """Check if Ollama is running and cache the result"""
    if st.session_state.ollama_status is None:
        try:
            llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.1)
            # Quick test
            response = llm.invoke("Hi")
            st.session_state.ollama_status = True
            return True
        except Exception as e:
            st.session_state.ollama_status = False
            return False
    return st.session_state.ollama_status

def display_sidebar():
    """Display sidebar with system status and controls"""
    with st.sidebar:
        st.title("🤖 Local RAG System")
        st.markdown("---")
        
        # System Status
        st.subheader("📊 System Status")
        
        # Ollama Status
        if check_ollama_status():
            st.success(f"✅ Ollama ({OLLAMA_MODEL}) - Ready")
        else:
            st.error("❌ Ollama - Not Available")
            with st.expander("🔧 Ollama Setup Guide"):
                st.markdown("""
                **Install Ollama:**
                1. Visit https://ollama.ai/
                2. Download and install Ollama
                3. Run: `ollama pull mistral`
                4. Verify: `ollama list`
                """)
        
        # Vectorstore Status
        if st.session_state.vectorstore:
            try:
                collection_data = st.session_state.vectorstore._collection.get()
                chunk_count = len(collection_data["ids"])
                st.success(f"✅ Vectorstore - {chunk_count} chunks")
            except:
                st.warning("⚠️ Vectorstore - Error")
        else:
            st.info("📄 No documents loaded")
        
        st.markdown("---")
        
        # File Upload Section
        st.subheader("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=[ext[1:] for ext in SUPPORTED_FORMATS],  # Remove dots
            accept_multiple_files=True,
            help=f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )
        
        if uploaded_files:
            # Validate file sizes
            valid_files = []
            for file in uploaded_files:
                file_size_mb = file.size / (1024 * 1024)
                if file_size_mb > MAX_FILE_SIZE:
                    st.error(f"❌ {file.name}: Too large ({file_size_mb:.1f}MB > {MAX_FILE_SIZE}MB)")
                else:
                    valid_files.append(file)
            
            if valid_files:
                if st.button("🚀 Process Documents", type="primary"):
                    process_documents(valid_files)
        
        # Clear Documents
        if st.session_state.vectorstore:
            st.markdown("---")
            if st.button("🗑️ Clear All Documents", type="secondary"):
                clear_documents()
        
        # Suggested Questions
        if st.session_state.vectorstore:
            st.markdown("---")
            st.subheader("💡 Suggested Questions")
            suggested_questions = [
                "What is the main topic?",
                "Summarize the key points",
                "Who are the main authors?",
                "What are the conclusions?",
                "List important dates or numbers"
            ]
            
            for question in suggested_questions:
                if st.button(f"❓ {question}", key=f"suggest_{hash(question)}"):
                    st.session_state.suggested_query = question

def process_documents(uploaded_files: List) -> None:
    """Process uploaded documents and create vectorstore"""
    with st.spinner("🔄 Processing documents..."):
        try:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process files
            status_text.text("Loading documents...")
            progress_bar.progress(25)
            
            docs, chunk_map, ocr_status, preview_images, vectorstore = process_uploaded_files(uploaded_files)
            
            progress_bar.progress(75)
            status_text.text("Creating vectorstore...")
            
            if vectorstore and docs:
                # Update session state
                st.session_state.vectorstore = vectorstore
                st.session_state.chunk_map = chunk_map
                st.session_state.ocr_status = ocr_status
                st.session_state.preview_images = preview_images
                st.session_state.processed_files = {file.name: True for file in uploaded_files}
                
                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")
                
                st.success(f"🎉 Successfully processed {len(uploaded_files)} files with {len(docs)} chunks!")
                
                # Auto-rerun to update sidebar
                st.rerun()
            else:
                st.error("❌ No documents were processed successfully")
                
        except Exception as e:
            st.error(f"❌ Error processing documents: {str(e)}")
        finally:
            # Clean up progress indicators
            if 'progress_bar' in locals():
                progress_bar.empty()
            if 'status_text' in locals():
                status_text.empty()

def clear_documents() -> None:
    """Clear all processed documents and reset session state"""
    st.session_state.vectorstore = None
    st.session_state.processed_files = {}
    st.session_state.ocr_status = {}
    st.session_state.preview_images = {}
    st.session_state.chunk_map = {}
    st.session_state.chat_history = []
    st.success("🗑️ All documents cleared!")
    st.rerun()

def display_document_status():
    """Display processed document status"""
    if not st.session_state.processed_files:
        st.info("📄 No documents loaded. Please upload documents using the sidebar.")
        return
    
    st.subheader("📚 Processed Documents")
    
    # Create columns for better layout
    cols = st.columns([2, 1, 1, 2])
    cols[0].write("**File Name**")
    cols[1].write("**Status**")
    cols[2].write("**Chunks**")
    cols[3].write("**Preview**")
    
    for filename in st.session_state.processed_files:
        cols = st.columns([2, 1, 1, 2])
        
        # File name
        cols[0].write(filename)
        
        # Status
        status = st.session_state.ocr_status.get(filename, "Unknown")
        if "✅" in status:
            cols[1].success(status)
        elif "⚠️" in status:
            cols[1].warning(status)
        else:
            cols[1].error(status)
        
        # Chunk count
        chunk_count = len(st.session_state.chunk_map.get(filename, []))
        cols[2].write(f"{chunk_count} chunks")
        
        # Preview
        preview = st.session_state.preview_images.get(filename)
        if preview:
            with cols[3]:
                st.image(preview, width=100, caption="Preview")
        else:
            cols[3].write("No preview")

def display_chat_interface():
    """Display the main chat interface for querying documents"""
    st.subheader("💬 Ask Questions")
    
    if not st.session_state.vectorstore:
        st.warning("⚠️ Please upload and process documents first!")
        return
    
    if not check_ollama_status():
        st.error("❌ Ollama is not available. Please check the sidebar for setup instructions.")
        return
    
    # Query input
    query_input = st.text_input(
        "Ask a question about your documents:",
        placeholder="What is the main topic of the document?",
        key="query_input"
    )
    
    # Handle suggested query
    if hasattr(st.session_state, 'suggested_query'):
        query_input = st.session_state.suggested_query
        delattr(st.session_state, 'suggested_query')
    
    # Query processing
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("🔍 Ask", type="primary", disabled=not query_input.strip()):
            process_query(query_input)
    
    with col2:
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("💭 Chat History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q{len(st.session_state.chat_history)-i}: {chat['query'][:50]}..."):
                st.markdown(f"**Question:** {chat['query']}")
                st.markdown(f"**Answer:** {chat['answer']}")
                
                if chat['sources']:
                    st.markdown("**Sources:**")
                    for j, source in enumerate(chat['sources'], 1):
                        source_name = source.metadata.get('source', 'Unknown')
                        st.markdown(f"*Source {j} - {source_name}:*")
                        st.text(source.page_content[:200] + "..." if len(source.page_content) > 200 else source.page_content)
                
                # Download button for this specific answer
                if st.button(f"📥 Download Answer {len(st.session_state.chat_history)-i}", key=f"download_{i}"):
                    download_answer(chat['query'], chat['answer'], chat['sources'])

def process_query(query: str) -> None:
    """Process a user query and display results"""
    if not query.strip():
        st.warning("⚠️ Please enter a question")
        return
    
    with st.spinner("🤔 Thinking..."):
        try:
            answer, sources = query_documents(query, st.session_state.vectorstore)
            
            # Add to chat history
            chat_entry = {
                'timestamp': datetime.now(),
                'query': query,
                'answer': answer,
                'sources': sources
            }
            st.session_state.chat_history.append(chat_entry)
            
            # Display results
            st.markdown("---")
            st.subheader("🧠 Answer")
            st.markdown(answer)
            
            if sources:
                st.subheader("📚 Sources")
                for i, source in enumerate(sources, 1):
                    source_name = source.metadata.get('source', 'Unknown')
                    with st.expander(f"📄 Source {i} - {source_name}"):
                        st.text(source.page_content)
                
                # Download button
                if st.button("📥 Download Answer as PDF"):
                    download_answer(query, answer, sources)
            else:
                st.info("ℹ️ No specific sources found for this query")
            
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")

def download_answer(query: str, answer: str, sources: List) -> None:
    """Generate and provide download for the answer"""
    try:
        # Generate PDF
        pdf_content = save_answer_as_file(answer, sources, query)
        
        if pdf_content:
            # Create download button
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_answer_{timestamp}.pdf"
            
            st.download_button(
                label="📥 Download PDF",
                data=pdf_content,
                file_name=filename,
                mime="application/pdf",
                key=f"download_{timestamp}"
            )
            st.success("✅ PDF generated successfully!")
        else:
            st.error("❌ Failed to generate PDF")
            
    except Exception as e:
        st.error(f"❌ Error generating download: {str(e)}")

def display_document_explorer():
    """Display document chunks explorer"""
    if not st.session_state.chunk_map:
        return
    
    st.subheader("🔍 Document Explorer")
    
    # File selector
    selected_file = st.selectbox(
        "Select a file to explore:",
        options=list(st.session_state.chunk_map.keys()),
        key="file_explorer"
    )
    
    if selected_file:
        chunks = st.session_state.chunk_map[selected_file]
        st.write(f"📊 **{selected_file}** contains {len(chunks)} chunks")
        
        # Chunk viewer
        chunk_index = st.slider(
            "Chunk number:",
            min_value=1,
            max_value=len(chunks),
            value=1,
            key="chunk_slider"
        ) - 1
        
        with st.expander(f"📄 Chunk {chunk_index + 1} of {len(chunks)}", expanded=True):
            st.text_area(
                "Content:",
                value=chunks[chunk_index],
                height=200,
                disabled=True,
                key=f"chunk_content_{chunk_index}"
            )

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Display sidebar
    display_sidebar()
    
    # Main content area
    st.title("🤖 Local RAG System")
    st.markdown("Upload documents, ask questions, get answers with sources - all running locally!")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📚 Documents", "🔍 Explorer"])
    
    with tab1:
        display_chat_interface()
    
    with tab2:
        display_document_status()
    
    with tab3:
        display_document_explorer()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8em;'>
        🔒 Privacy-First • 🏠 Runs Locally • 🚀 Powered by LangChain + Ollama
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

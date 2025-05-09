import PyPDF2
import re
import streamlit as st
from transformers import pipeline, BartTokenizer
from typing import List, Optional
import time
from io import BytesIO
import base64
import pandas as pd

# Constants
MAX_CHUNK_SIZE = 1024
DEFAULT_MODEL = "facebook/bart-large-cnn"

# Initialize tokenizer (with caching)
@st.cache_resource
def load_tokenizer():
    return BartTokenizer.from_pretrained(DEFAULT_MODEL)

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model=DEFAULT_MODEL)

# 1. Enhanced PDF Text Extraction
def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from PDF with error handling and validation."""
    try:
        text = ""
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        
        if len(pdf_reader.pages) == 0:
            raise ValueError("The uploaded PDF appears to be empty.")
            
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # Only add if text was extracted
                text += page_text
                
        if not text.strip():
            raise ValueError("No readable text could be extracted from the PDF.")
            
        return text
    except PyPDF2.PdfReadError:
        raise ValueError("The uploaded file is not a valid PDF or is corrupted.")

# 2. Improved Text Cleaning
def clean_text(text: str) -> str:
    """Clean extracted text with comprehensive preprocessing."""
    # Remove page numbers and headers/footers
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'\bPage \d+\b', '', text)
    
    # Normalize whitespace and clean special characters
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII chars
    
    # Remove common document artifacts
    text = re.sub(r'\b\d{6,}\b', '', text)  # Remove long digit sequences
    text = re.sub(r'\b[A-Z]{6,}\b', '', text)  # Remove long uppercase sequences
    
    return text

# 3. Enhanced Text Chunking
def split_text(text: str, max_chunk: int = MAX_CHUNK_SIZE) -> List[str]:
    """Split text into coherent chunks respecting token limits."""
    tokenizer = load_tokenizer()
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_tokens = len(tokenizer.tokenize(sentence))
        
        if current_length + sentence_tokens > max_chunk and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_length += sentence_tokens
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# 4. Advanced Summarization with Progress Tracking
def summarize_text(text: str, summarizer, max_length: int = 150, min_length: int = 30) -> str:
    """Generate summary with progress feedback and chunk processing."""
    chunks = split_text(text)
    summaries = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, chunk in enumerate(chunks):
        status_text.text(f"Processing chunk {i+1} of {len(chunks)}...")
        progress_bar.progress((i + 1) / len(chunks))
        
        try:
            summary = summarizer(
                chunk,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            summaries.append(summary[0]['summary_text'])
        except Exception as e:
            st.warning(f"Error processing chunk {i+1}: {str(e)}")
            summaries.append("")  # Continue with other chunks
    
    progress_bar.empty()
    status_text.empty()
    return ' '.join(summaries)

# 5. Helper function for file download
def create_download_link(content: str, filename: str, title: str) -> str:
    """Generate a download link for the summary."""
    b64 = base64.b64encode(content.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">{title}</a>'

# 6. Main Application with Enhanced UI
def main():
    # Configure page
    st.set_page_config(
        page_title="Advanced Insurance Document Summarizer",
        page_icon="📄",
        layout="wide"
    )
    
    # Custom CSS for better appearance
    st.markdown("""
    <style>
        .main {background-color: #f8f9fa;}
        .stButton>button {border-radius: 5px; padding: 10px 20px;}
        .stTextArea textarea {border-radius: 5px;}
        .stDownloadButton>button {width: 100%;}
        .summary-box {border-left: 4px solid #4e79a7; padding: 10px 15px; background-color: #f0f7ff; border-radius: 5px;}
        .header {color: #2c3e50;}
        .sidebar .sidebar-content {background-color: #e9ecef;}
    </style>
    """, unsafe_allow_html=True)
    
    # App header
    st.markdown("<h1 class='header'>📑 Advanced Insurance Document Summarizer</h1>", unsafe_allow_html=True)
    st.markdown("""
    Upload your insurance policy PDF to get a clear, concise summary. 
    This tool helps you quickly understand complex documents without reading every page.
    """)
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        difficulty = st.radio(
            "Summary Detail Level:",
            ("Simple", "Balanced", "Detailed"),
            index=1,
            help="Simple = very concise, Detailed = more comprehensive"
        )
        
        show_stats = st.checkbox("Show document statistics", True)
        show_original = st.checkbox("Show original text preview", False)
        
        st.markdown("---")
        st.markdown("**About**")
        st.markdown("""
        This tool uses advanced NLP to summarize insurance documents.
        - Powered by BART-large-CNN model
        - Processes documents in chunks for better accuracy
        - Maintains key information while reducing length
        """)
    
    # Main content area
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type="pdf",
        help="Upload your insurance policy or other document to summarize"
    )
    
    if uploaded_file is not None:
        with st.spinner("Processing your document..."):
            try:
                # Extract and clean text with error handling
                start_time = time.time()
                raw_text = extract_text_from_pdf(uploaded_file)
                cleaned_text = clean_text(raw_text)
                
                # Show document statistics
                if show_stats:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Pages", len(PyPDF2.PdfReader(uploaded_file).pages))
                    with col2:
                        st.metric("Original Characters", f"{len(raw_text):,}")
                    with col3:
                        st.metric("Cleaned Characters", f"{len(cleaned_text):,}")
                
                # Show original text preview if requested
                if show_original:
                    with st.expander("📋 Original Text Preview", expanded=False):
                        st.text_area(
                            "Original Text", 
                            value=cleaned_text[:3000] + ("..." if len(cleaned_text) > 3000 else ""), 
                            height=300,
                            label_visibility="collapsed"
                        )
                
                # Initialize summarizer
                summarizer = load_summarizer()
                
                # Generate summary based on difficulty
                st.subheader("📝 Generated Summary")
                with st.spinner(f"Generating {difficulty.lower()} summary..."):
                    if difficulty == "Simple":
                        summary = summarize_text(cleaned_text, summarizer, max_length=80, min_length=20)
                    elif difficulty == "Balanced":
                        summary = summarize_text(cleaned_text, summarizer, max_length=150, min_length=50)
                    else:  # Detailed
                        summary = summarize_text(cleaned_text, summarizer, max_length=300, min_length=100)
                
                # Display summary in a nice box
                st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
                
                # Show performance metrics
                processing_time = time.time() - start_time
                st.caption(f"Processed in {processing_time:.1f} seconds | Summary length: {len(summary):,} characters")
                
                # Add download option
                st.markdown("---")
                st.markdown("### Download Summary")
                st.markdown(create_download_link(summary, "insurance_summary.txt", "⬇️ Download Summary as Text File"), unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.stop()

if __name__ == "__main__":
    main()
# Import required libraries
import PyPDF2
import re
from transformers import pipeline
import streamlit as st
from transformers import BartTokenizer

# Initialize tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# 1. PDF Text Extraction Function (same as before)
def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# 2. Text Cleaning Function (same as before)
def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'Page \d+ of \d+', '', text)
    return text

# 3. NEW: Function to split text into chunks
def split_text(text, max_chunk=1024):
    words = text.split(' ')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(tokenizer.tokenize(word)) + 1 <= max_chunk:
            current_chunk.append(word)
            current_length += len(tokenizer.tokenize(word)) + 1
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(tokenizer.tokenize(word)) + 1
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# 4. Summarization Function with chunking
def summarize_text(text, summarizer, max_length=150):
    chunks = split_text(text)
    summaries = []
    
    for chunk in chunks:
        summary = summarizer(
            chunk,
            max_length=max_length,
            min_length=30,
            do_sample=False
        )
        summaries.append(summary[0]['summary_text'])
    
    return ' '.join(summaries)

# 5. Main Application
def main():
    st.title("Insurance Document Summarizer")
    st.write("Upload your insurance policy PDF to get a simple summary")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    difficulty = st.radio(
        "Select summary type:",
        ("Simple", "Detailed"),
        horizontal=True
    )
    
    if uploaded_file is not None:
        # Extract and clean text
        raw_text = extract_text_from_pdf(uploaded_file)
        cleaned_text = clean_text(raw_text)
        
        # Initialize summarizer
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Show original text (optional)
        with st.expander("View Original Text"):
            st.write(cleaned_text[:1000] + "...")  # Show first 1000 chars
            
        # Generate summary based on difficulty
        if difficulty == "Simple":
            summary = summarize_text(cleaned_text, summarizer, max_length=100)
        else:
            summary = summarize_text(cleaned_text, summarizer, max_length=300)
        
        st.subheader(f"{difficulty} Summary")
        st.write(summary)

if __name__ == "__main__":
    main()
import streamlit as st
import fitz  # PyMuPDF - modern, fast PDF parser
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
import os

# --- Load environment variables ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- Streamlit App UI ---
st.set_page_config(page_title="PDF Topic Evaluator", layout="wide")
st.title("📄 PDF Topic Evaluator")
st.markdown("Upload one or more PDFs and check if a specific topic is addressed.")

# --- User Inputs ---
topic = st.text_input("🔍 Topic to evaluate (e.g. 'test-driven development')")

considerations_input = st.text_area(
    "🧠 Considerations to Analyze (one per line)",
    value="\n".join([
        "Values",
        "Included groups",
        "Excluded groups",
        "National myths or symbols",
        "Civic assumptions"
    ]),
    height=150
)

# Convert to a formatted list
considerations_list = considerations_input.strip().splitlines()
formatted_considerations = "\n- " + "\n- ".join(considerations_list)

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# --- Validation ---
if not openai_api_key:
    st.error("API key not found. Please create a .env file with OPENAI_API_KEY=your_key_here.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# --- Helper: Extract text from PDF using PyMuPDF ---
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF using PyMuPDF (faster and more accurate than PyPDF2)"""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# --- Helper: Chunk text intelligently ---
def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks

# --- Helper: Get embeddings from OpenAI ---
def get_embeddings(texts):
    """Get embeddings using OpenAI's latest embedding model"""
    response = client.embeddings.create(
        model="text-embedding-3-large",  # Latest and most capable embedding model
        input=texts
    )
    return [item.embedding for item in response.data]

# --- Helper: Find relevant chunks using semantic search ---
def find_relevant_chunks(chunks, topic, top_k=5):
    """Find most relevant chunks using cosine similarity"""
    if not chunks:
        return []
    
    # Get embeddings for all chunks and the topic
    all_texts = chunks + [topic]
    embeddings = get_embeddings(all_texts)
    
    chunk_embeddings = np.array(embeddings[:-1])
    topic_embedding = np.array(embeddings[-1]).reshape(1, -1)
    
    # Calculate cosine similarity
    similarities = cosine_similarity(topic_embedding, chunk_embeddings)[0]
    
    # Get top k most similar chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    relevant_chunks = [chunks[i] for i in top_indices]
    
    return relevant_chunks

# --- Topic Evaluation Logic ---
def evaluate_topic(text, topic, considerations_text, filename):
    """Evaluate if PDF addresses the topic using OpenAI's best model"""
    
    # Chunk the document
    chunks = chunk_text(text)
    
    if not chunks:
        return f"Unable to extract text from {filename}"
    
    # Find most relevant chunks using semantic search
    relevant_chunks = find_relevant_chunks(chunks, topic, top_k=5)
    
    # Combine relevant chunks for analysis
    context = "\n\n---\n\n".join(relevant_chunks)
    
    # Create the evaluation prompt
    prompt = f"""Analyze the following document excerpts to determine if they address the topic: "{topic}"

Document excerpts:
{context}

Task:
1. Determine if this document addresses the topic "{topic}"
2. If YES:
   - Analyze the relevant passages using these considerations:
   {considerations_text}
   - Provide 1-2 direct quotes (short phrases) that support your findings
   
3. If NO:
   - Explain why the topic appears to be absent from this document

Be concise and analytical in your response."""

    # Call OpenAI API with the best model
    response = client.chat.completions.create(
        model="gpt-4o",  # Latest and most capable model
        messages=[
            {"role": "system", "content": "You are an expert document analyst specializing in topic evaluation and thematic analysis."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=800
    )
    
    return response.choices[0].message.content

# --- Main Button Logic ---
if st.button("Evaluate PDFs", key="evaluate_button"):
    # Validation
    if not topic:
        st.warning("Please enter a topic to evaluate.")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF file.")
    else:
        # Process PDFs
        results = []
        
        for file in uploaded_files:
            with st.spinner(f"Evaluating: {file.name}..."):
                try:
                    text = extract_text_from_pdf(file)
                    
                    if not text.strip():
                        evaluation = f"No text could be extracted from {file.name}"
                    else:
                        evaluation = evaluate_topic(text, topic, formatted_considerations, file.name)
                    
                    results.append({"PDF": file.name, "Evaluation": evaluation})
                except Exception as e:
                    results.append({"PDF": file.name, "Evaluation": f"Error processing file: {str(e)}"})
        
        st.subheader("📊 Evaluation Results")
        
        # Display results in expandable sections for better readability
        for result in results:
            with st.expander(f"📄 {result['PDF']}", expanded=True):
                st.markdown(result['Evaluation'])
        
        # Also provide a downloadable summary
        st.divider()
        summary = "\n\n" + "="*80 + "\n\n".join([
            f"PDF: {r['PDF']}\n\nEvaluation:\n{r['Evaluation']}" 
            for r in results
        ])
        
        st.download_button(
            label="Download Results as Text",
            data=summary,
            file_name="pdf_evaluation_results.txt",
            mime="text/plain"
        )
"""
src/rag_pipeline.py

Task 3: RAG Pipeline with Retriever + Generator + Evaluation
"""

import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
import json

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent
VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store" / "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

class ComplaintRAG:
    def __init__(self, vector_store_path=None, embedding_model=None):
        if vector_store_path is None:
            vector_store_path = VECTOR_STORE_PATH
        if embedding_model is None:
            embedding_model = EMBEDDING_MODEL
        
        # Load embedding model
        self.model = SentenceTransformer(embedding_model)
        
        # Load vector store
        self.client = chromadb.PersistentClient(path=str(vector_store_path))
        self.collection = self.client.get_collection("complaints")
        
        print(f"✅ Loaded vector store with {self.collection.count()} chunks")
    
    def retrieve(self, query, top_k=None):
        """Retrieve top-k relevant chunks for a query"""
        if top_k is None:
            top_k = TOP_K
        
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Search in vector store
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Extract documents and metadata
        documents = results['documents'][0] if results['documents'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        distances = results['distances'][0] if results['distances'] else []
        
        retrieved = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            retrieved.append({
                'text': doc,
                'complaint_id': meta.get('complaint_id', 'Unknown'),
                'product': meta.get('product', 'Unknown'),
                'score': 1 - dist  # convert distance to similarity score
            })
        
        return retrieved
    
    def generate_answer(self, query, retrieved_chunks, llm_function):
        """Generate answer using retrieved context and LLM"""
        # Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            context_parts.append(f"[Source {i+1}] Product: {chunk['product']}\n{chunk['text']}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are a financial analyst assistant for CreditTrust. Your task is to answer questions about customer complaints. Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate response using the provided LLM function
        response = llm_function(prompt)
        return response
    
    def query(self, query, llm_function, top_k=None):
        """Full RAG pipeline: retrieve + generate"""
        # 1. Retrieve
        retrieved = self.retrieve(query, top_k)
        
        # 2. Generate
        answer = self.generate_answer(query, retrieved, llm_function)
        
        return {
            'question': query,
            'answer': answer,
            'sources': retrieved
        }


# --- LLM Functions ---

def use_huggingface_local(prompt, model_name="HuggingFaceH4/zephyr-7b-beta"):
    """Use a local Hugging Face model"""
    from transformers import pipeline
    generator = pipeline("text-generation", model=model_name, device="cpu")
    result = generator(prompt, max_new_tokens=200, do_sample=False)
    return result[0]['generated_text']

def use_huggingface_api(prompt, model="mistralai/Mistral-7B-Instruct-v0.3"):
    """Use Hugging Face Inference API (free tier)"""
    import requests
    import os
    
    API_TOKEN = os.getenv("HF_TOKEN")
    if not API_TOKEN:
        print("⚠️ HF_TOKEN not set. Using mock LLM.")
        return mock_llm(prompt)
    
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 200, "temperature": 0.3}
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()[0]['generated_text']
        else:
            print(f"⚠️ API Error: {response.status_code}")
            return mock_llm(prompt)
    except Exception as e:
        print(f"⚠️ API Exception: {e}")
        return mock_llm(prompt)

def mock_llm(prompt):
    """Placeholder for testing"""
    return f"[Mock Answer] The system retrieved relevant complaint excerpts. Please set up a real LLM for actual answers."


# --- Evaluation ---

def evaluate_rag(rag_system, test_questions, llm_function):
    """Run evaluation on test questions"""
    results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"Test Question {i}/{len(test_questions)}")
        print(f"Q: {question}")
        print('-'*60)
        
        result = rag_system.query(question, llm_function)
        
        print(f"A: {result['answer']}")
        print(f"\nSources: {len(result['sources'])} chunks retrieved")
        print(f"Top source score: {result['sources'][0]['score']:.3f}")
        
        results.append(result)
    
    return results


# --- Main Execution ---

def main():
    # Test questions (5-10 questions for evaluation)
    test_questions = [
        "Why are people unhappy with Credit Cards?",
        "What are the most common complaints about Money Transfers?",
        "Are there any complaints about unauthorized transactions?",
        "What billing issues do customers report?",
        "Do customers complain about customer service?",
        "Are there any complaints about fees?",
        "What fraud-related complaints do customers have?",
    ]
    
    # Initialize RAG system
    print("="*60)
    print("Task 3: RAG Pipeline Evaluation")
    print("="*60)
    
    print("\nInitializing RAG system...")
    rag = ComplaintRAG()
    
    # Choose LLM function
    # Option 1: Use Hugging Face API (requires HF_TOKEN)
    # llm_function = use_huggingface_api
    
    # Option 2: Use local model (requires ~7GB RAM)
    # llm_function = use_huggingface_local
    
    # Option 3: Use mock LLM for testing (fastest)
    llm_function = mock_llm
    print("Using mock LLM for testing. Set up real LLM for actual answers.")
    
    # Run evaluation
    print("\n" + "="*60)
    print("Running Evaluation on Test Questions")
    print("="*60)
    
    results = evaluate_rag(rag, test_questions, llm_function)
    
    # Print summary
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    
    for i, r in enumerate(results, 1):
        print(f"\n{i}. Q: {r['question']}")
        print(f"   Sources: {len(r['sources'])} chunks retrieved")
        # Get top product and score
        top_product = r['sources'][0]['product'] if r['sources'] else 'N/A'
        top_score = r['sources'][0]['score'] if r['sources'] else 0
        print(f"   Top source product: {top_product} (score: {top_score:.3f})")
        print(f"   Answer preview: {r['answer'][:100]}...")


if __name__ == "__main__":
    main()
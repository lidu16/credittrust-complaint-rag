"""
src/vector_store.py

Task 2: Chunking, Embedding, and Vector Store Indexing
- Load filtered dataset
- Stratified sample (~12,000 complaints)
- Chunk with LangChain's RecursiveCharacterTextSplitter
- Embed with sentence-transformers/all-MiniLM-L6-v2
- Store in ChromaDB (persistent)
"""

import pandas as pd
import numpy as np
from pathlib import Path
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=0, length_function=len, separators=None):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.length_function = length_function
            self.separators = separators or ["\n\n", "\n", " ", ""]

        def split_text(self, text):
            if not isinstance(text, str) or self.length_function(text) == 0:
                return []
            if self.length_function(text) <= self.chunk_size:
                return [text]

            for sep in self.separators:
                if sep and sep in text:
                    parts = [p for p in text.split(sep) if p]
                    if len(parts) > 1:
                        chunks = []
                        for part in parts:
                            chunks.extend(self.split_text(part))
                        merged = []
                        current = ""
                        for chunk in chunks:
                            if not current:
                                current = chunk
                            elif self.length_function(current + sep + chunk) <= self.chunk_size:
                                current = current + sep + chunk
                            else:
                                merged.append(current)
                                current = chunk
                        if current:
                            merged.append(current)
                        return merged

            chunks = []
            text_len = self.length_function(text)
            start = 0
            while start < text_len:
                end = min(start + self.chunk_size, text_len)
                chunks.append(text[start:end])
                if end == text_len:
                    break
                start = max(end - self.chunk_overlap, start + 1)
            return chunks

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import uuid
import shutil
import time

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "filtered_complaints.csv"
VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store" / "chroma_db"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SAMPLE_SIZE = 12000          # total number of complaints to sample
PRODUCT_COL = "Product"      # column name in your CSV

def stratified_sample(df, sample_size, stratum_col):
    """
    Perform proportional stratified sampling.
    """
    strata = df[stratum_col].value_counts(normalize=True)
    sampled_dfs = []
    for stratum, proportion in strata.items():
        n = int(round(proportion * sample_size))
        if n <= 0:
            continue
        stratum_df = df[df[stratum_col] == stratum]
        # sample with replacement if necessary
        if len(stratum_df) < n:
            sampled = stratum_df.sample(n=n, replace=True, random_state=42)
        else:
            sampled = stratum_df.sample(n=n, random_state=42)
        sampled_dfs.append(sampled)
    result = pd.concat(sampled_dfs).reset_index(drop=True)
    # if we have more than sample_size, trim
    if len(result) > sample_size:
        result = result.sample(n=sample_size, random_state=42)
    return result

def main():
    print("=" * 60)
    print("Task 2: Building Vector Store")
    print("=" * 60)

    # 1. Load data
    print(f"\n[1/5] Loading data from {PROCESSED_DATA_PATH}")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"   Total records: {len(df)}")

    # 2. Stratified sampling
    print(f"\n[2/5] Stratified sampling (target: {SAMPLE_SIZE} complaints)")
    df_sample = stratified_sample(df, SAMPLE_SIZE, PRODUCT_COL)
    print(f"   Actual sample size: {len(df_sample)}")
    print(f"   Product distribution:\n{df_sample[PRODUCT_COL].value_counts()}")

    # 3. Chunking
    print(f"\n[3/5] Chunking narratives (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []
    for idx, row in df_sample.iterrows():
        narrative = row.get("cleaned_narrative", "")
        if not isinstance(narrative, str) or len(narrative.strip()) == 0:
            continue
        # split
        text_chunks = text_splitter.split_text(narrative)
        for i, chunk in enumerate(text_chunks):
            chunks.append({
                "complaint_id": row.get("Complaint ID", idx),
                "product": row[PRODUCT_COL],
                "chunk_text": chunk,
                "chunk_index": i,
                "total_chunks": len(text_chunks)
            })

    print(f"   Total chunks generated: {len(chunks)}")
    if not chunks:
        print("ERROR: No chunks generated. Check your data.")
        return

    # 4. Embeddings
    print(f"\n[4/5] Generating embeddings with {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    chunk_texts = [c["chunk_text"] for c in chunks]
    # encode in batches
    batch_size = 64
    all_embeddings = []
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i:i+batch_size]
        batch_emb = model.encode(batch, show_progress_bar=False)
        all_embeddings.extend(batch_emb)
    print(f"   Embedding dimension: {len(all_embeddings[0])}")

    # 5. Store in ChromaDB
    print(f"\n[5/5] Building ChromaDB at {VECTOR_STORE_PATH}")
    # Remove existing if any
    if VECTOR_STORE_PATH.exists():
        shutil.rmtree(VECTOR_STORE_PATH)
        print("   Removed existing vector store")

    # Create persistent client
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_PATH))

    # Create collection with embedding function
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_or_create_collection(
        name="complaints",
        embedding_function=embedding_fn
    )

    # Prepare data for insertion
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
    documents = [c["chunk_text"] for c in chunks]
    metadatas = [
        {
            "complaint_id": str(c["complaint_id"]),
            "product": c["product"],
            "chunk_index": c["chunk_index"],
            "total_chunks": c["total_chunks"]
        }
        for c in chunks
    ]

    # Insert in batches
    batch_size_insert = 1000
    total_inserted = 0
    for i in range(0, len(chunks), batch_size_insert):
        batch_end = min(i + batch_size_insert, len(chunks))
        collection.add(
            ids=ids[i:batch_end],
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end]
        )
        total_inserted += (batch_end - i)
        print(f"   Inserted {total_inserted} / {len(chunks)} chunks")

    print(f"\n✅ Vector store built successfully!")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Location: {VECTOR_STORE_PATH}")

if __name__ == "__main__":
    main()
"""
src/data_loader.py

Load, clean, and filter the CFPB complaint dataset.
"""

import pandas as pd
import re
from pathlib import Path

# Project paths (relative to this file)
ROOT_DIR = Path(__file__).parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FILTERED_CSV_PATH = DATA_PROCESSED_DIR / "filtered_complaints.csv"

# List of target products (use exact spelling from the dataset)
TARGET_PRODUCTS = ['Credit card', 'Personal loan', 'Savings account', 'Money transfer', 'Money transfers']

def clean_text(text: str) -> str:
    """
    Clean a complaint narrative:
    - lowercase
    - remove special chars (keep letters, numbers, spaces, ., , ? !)
    - remove common boilerplate phrases
    - collapse multiple spaces
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\?\!]', '', text)

    boilerplates = [
        "i am writing to file a complaint",
        "i am writing to complain about",
        "i am writing to notify you",
        "this is a complaint regarding",
        "i am submitting this complaint because"
    ]
    for phrase in boilerplates:
        text = text.replace(phrase, "")

    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess(raw_path=None, save=True):
    """
    Load the raw CSV, filter for target products, clean narratives,
    and optionally save to processed directory.
    """
    if raw_path is None:
        raw_path = DATA_RAW_DIR / "complaints.csv"

    print(f"Loading raw data from {raw_path}...")
    df = pd.read_csv(raw_path, low_memory=False)

    # Use the exact column names from your dataset
    product_col = 'Product'   # capital P
    narrative_col = 'Consumer complaint narrative'  # exact with spaces

    print(f"Using product column: {product_col}")
    print(f"Using narrative column: {narrative_col}")

    # Filter by product
    df_filtered = df[df[product_col].isin(TARGET_PRODUCTS)].copy()
    print(f"Records after product filter: {len(df_filtered)}")

    # Drop rows with missing narrative
    before = len(df_filtered)
    df_filtered = df_filtered.dropna(subset=[narrative_col])
    print(f"Dropped {before - len(df_filtered)} rows with empty narratives.")

    # Clean the narrative
    df_filtered['cleaned_narrative'] = df_filtered[narrative_col].apply(clean_text)

    # Optionally drop rows where cleaned narrative is empty
    df_filtered = df_filtered[df_filtered['cleaned_narrative'].str.len() > 0].copy()
    print(f"Final dataset size: {len(df_filtered)}")

    # Save if requested
    if save:
        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df_filtered.to_csv(FILTERED_CSV_PATH, index=False)
        print(f"Saved to {FILTERED_CSV_PATH}")

    return df_filtered

if __name__ == "__main__":
    # Run from command line: python src/data_loader.py
    df = load_and_preprocess()
    print("\nProduct distribution after preprocessing:")
    product_col = 'Product'
    print(df[product_col].value_counts())

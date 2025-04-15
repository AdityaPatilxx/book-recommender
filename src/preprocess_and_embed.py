# src/preprocess_and_embed.py

import pandas as pd
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Changed
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
import os
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "../data/books.csv"
CLEANED_DATA_PATH = "../data/df_cleaned.csv"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"  # Changed to all-mpnet-base-v2
PERSIST_DIRECTORY = "chroma_db"
CHUNK_SIZE = 500  # Reduced chunk size
CHUNK_OVERLAP = 50

def clean_data(df):
    # Create 'missing_description' and 'book_age' columns
    df["missing_description"] = np.where(df['description'].isna(), 1, 0)
    df['book_age'] = 2025 - df['published_year']

    # Filter out rows with missing essential values
    df = df[~(df["num_pages"].isna()) &
              (~df["average_rating"].isna()) &
              (~df["description"].isna()) &
              (~df["published_year"].isna())]

    # Create 'words_in_description' and filter based on its length
    df['words_in_description'] = df['description'].str.split().str.len()
    df = df[df['words_in_description'] >= 25]

    # Create 'title and subtitle'
    df['title and subtitle'] = np.where(
        df['subtitle'].isna(),
        df['title'],
        df[['title', 'subtitle']].astype(str).agg(": ".join, axis=1))

    # Create combined text field
    df['combined_text'] = df['title'] + " " + df['authors'] + " " + df['description'] #added authors

    # Create 'tagged_desc'
    df['tagged_desc'] = df[["isbn13", "combined_text"]].astype(str).agg(" ".join, axis=1) #changed to combined_text

    # Drop the intermediate columns
    return df.drop(['subtitle', 'missing_description', 'book_age', 'words_in_description', 'combined_text'], axis=1) #removed combined_text

def create_embeddings_and_store(df, persist_directory):
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    loader = DataFrameLoader(df, page_content_column="tagged_desc")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n", " ", ""]) # Changed
    texts = text_splitter.split_documents(documents)
    db = Chroma.from_documents(texts, embedding, persist_directory=persist_directory)
    db.persist()
    print(f"Embeddings stored in: {persist_directory}")

if __name__ == "__main__":
    if not os.path.exists(CLEANED_DATA_PATH):
        print("Cleaning data...")
        df = pd.read_csv(DATA_PATH)
        df_cleaned = clean_data(df)
        df_cleaned.to_csv(CLEANED_DATA_PATH, index=False)
        print(f"Cleaned data saved to: {CLEANED_DATA_PATH}")
    else:
        print(f"Loading cleaned data from: {CLEANED_DATA_PATH}")
        df_cleaned = pd.read_csv(CLEANED_DATA_PATH)

    if not os.path.exists(PERSIST_DIRECTORY):
        print("Creating embeddings and storing in ChromaDB...")
        create_embeddings_and_store(df_cleaned, PERSIST_DIRECTORY)
    else:
        print(f"ChromaDB already exists at: {PERSIST_DIRECTORY}. Skipping embedding creation.")
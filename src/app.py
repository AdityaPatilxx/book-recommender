# src/app.py

import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import pipeline
import gradio as gr
import os
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "../data/df_cleaned.csv"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PERSIST_DIRECTORY = "chroma_db"
CLASSIFICATION_MODEL_NAME = "facebook/bart-large-mnli"

def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: {DATA_PATH} not found. Make sure you have run the preprocessing script.")
        return None

def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def load_vector_store(embedding, persist_directory):
    try:
        db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        return db
    except Exception as e:
        print(f"Error loading ChromaDB from {persist_directory}: {e}")
        return None

def get_semantic_recommendations(query, df, db, top_k=10):
    if db is None or df is None:
        return pd.DataFrame(columns=['title', 'authors', 'categories'])
    docs = db.similarity_search(query, k=50)
    books_list = []
    for doc in docs:
        try:
            isbn_candidate = int(doc.page_content.strip('"').split()[0])
            books_list.append(isbn_candidate)
        except ValueError:
            continue
    recommendations_df = df[df["isbn13"].isin(books_list)][['title', 'authors', 'categories']].head(top_k)
    return recommendations_df

def load_classifier(model_name):
    try:
        classifier = pipeline("zero-shot-classification", model=model_name)
        return classifier
    except Exception as e:
        print(f"Error loading classifier {model_name}: {e}")
        return None

def classify_genre(description, classifier, candidate_labels=None):
    if classifier is None:
        return None
    if candidate_labels is None:
        candidate_labels = ["fiction", "non-fiction", "mystery", "science fiction", "fantasy", "biography", "history", "young adult", "children's"]
    try:
        result = classifier(description, candidate_labels, multi_label=False)
        return result['labels'][0]
    except Exception as e:
        print(f"Error during classification: {e}")
        return None

def get_semantic_recommendations_with_genre_filter(query, genre, df, db, classifier, top_k=10):
    if db is None or df is None or classifier is None:
        return pd.DataFrame(columns=['title', 'authors', 'predicted_genre'])
    docs = db.similarity_search(query, k=100)
    books_list = []
    for doc in docs:
        try:
            isbn_candidate = int(doc.page_content.strip('"').split()[0])
            books_list.append(isbn_candidate)
        except ValueError:
            continue
    recommended_df = df[df["isbn13"].isin(books_list)].copy()
    recommended_df['predicted_genre'] = recommended_df['description'].apply(lambda x: classify_genre(x, classifier))
    filtered_df = recommended_df[recommended_df['predicted_genre'] == genre][['title', 'authors', 'predicted_genre']].head(top_k)
    return filtered_df

def recommend_interface(query, use_genre_filter=False, genre_choice="fiction"):
    df = load_data()
    embedding = load_embeddings()
    db = load_vector_store(embedding, PERSIST_DIRECTORY)
    classifier = load_classifier(CLASSIFICATION_MODEL_NAME)

    if df is None or db is None:
        return pd.DataFrame(columns=['title', 'authors', 'categories'] if classifier is None else ['title', 'authors', 'predicted_genre'])

    if use_genre_filter and classifier:
        return get_semantic_recommendations_with_genre_filter(query, genre_choice, df, db, classifier)
    else:
        return get_semantic_recommendations(query, df, db)

def main():
    classifier_loaded = load_classifier(CLASSIFICATION_MODEL_NAME)
    output_headers = ['title', 'authors', 'categories'] if classifier_loaded is None else ['title', 'authors', 'predicted_genre']

    iface = gr.Interface(
        fn=recommend_interface,
        inputs=[
            gr.Textbox(label="Enter a book title or description"),
            gr.Checkbox(label="Filter by Genre?"),
            gr.Dropdown(choices=["fiction", "non-fiction", "mystery", "science fiction", "fantasy", "biography", "history", "young adult", "children's"], label="Select Genre")
        ],
        outputs=gr.DataFrame(headers=output_headers),
        title="Semantic Book Recommendation System",
        description="Get book recommendations based on semantic understanding of text.",
    )
    iface.launch()

if __name__ == "__main__":
    main()
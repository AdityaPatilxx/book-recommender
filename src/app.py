# src/app.py

import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer  # Added
import gradio as gr
import os
import torch
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "../data/df_cleaned.csv"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"  # Changed
PERSIST_DIRECTORY = "chroma_db"
CLASSIFICATION_MODEL_NAME = "facebook/bart-large-mnli"
RERANKING_MODEL_NAME = "cross-encoder/ms-marco-bert-base-reranker" # Added


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

def load_reranker(model_name):  # Added function to load reranking model
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading reranking model {model_name}: {e}")
        return None, None, None

def rerank_results(query, results, model, tokenizer, device): # Added
    if model is None or tokenizer is None:
        return results
    try:
        inputs = tokenizer(
            [query] * len(results),
            results,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            scores = torch.sigmoid(model(**inputs).logits)[:, 0].cpu().tolist()
        reranked_results = [(result, score) for result, score in zip(results, scores)]
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        return [result for result, _ in reranked_results]
    except Exception as e:
        print(f"Error during reranking: {e}")
        return results
def get_semantic_recommendations(query, df, db, reranker_model, reranker_tokenizer, reranker_device, top_k=10): #added reranker
    if db is None or df is None:
        return pd.DataFrame(columns=['title', 'authors', 'categories'])
    docs = db.similarity_search(query, k=50)  # Increased k
    results = [doc.page_content for doc in docs]
    reranked_results = rerank_results(query, results, reranker_model, reranker_tokenizer, reranker_device) #added
    books_list = []
    for doc in reranked_results: #changed
        try:
            isbn_candidate = int(doc.strip('"').split()[0])
            books_list.append(isbn_candidate)
        except ValueError:
            continue
    recommendations_df = df[df["isbn13"].isin(books_list)][['title', 'authors', 'categories']].head(top_k)
    return recommendations_df

def get_semantic_recommendations_with_genre_filter(query, genre, df, db, classifier, reranker_model, reranker_tokenizer, reranker_device, top_k=10): #added reranker
    if db is None or df is None or classifier is None:
        return pd.DataFrame(columns=['title', 'authors', 'predicted_genre'])
    docs = db.similarity_search(query, k=100)  # Increased k
    results = [doc.page_content for doc in docs]
    reranked_results = rerank_results(query, results, reranker_model, reranker_tokenizer, reranker_device) #added
    books_list = []
    for doc in reranked_results: #changed
        try:
            isbn_candidate = int(doc.strip('"').split()[0])
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
    reranker_model, reranker_tokenizer, reranker_device = load_reranker(RERANKING_MODEL_NAME) #added

    if df is None or db is None:
        return pd.DataFrame(columns=['title', 'authors', 'categories'] if classifier is None else ['title', 'authors', 'predicted_genre'])

    if use_genre_filter and classifier:
        return get_semantic_recommendations_with_genre_filter(query, genre_choice, df, db, classifier, reranker_model, reranker_tokenizer, reranker_device) #added reranker
    else:
        return get_semantic_recommendations(query, df, db, reranker_model, reranker_tokenizer, reranker_device) #added reranker

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
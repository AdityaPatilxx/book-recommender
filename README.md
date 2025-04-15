# Semantic Book Recommendation System

## Overview

This project implements a semantic book recommendation system using Python. It leverages natural language processing (NLP) techniques to understand the meaning of book descriptions and user queries, providing more relevant recommendations than traditional keyword-based approaches. The system uses:

- **Hugging Face Transformers:** For generating text embeddings.
- **Langchain and Langchain-Community:** For managing the NLP pipeline and interacting with the vector database.
- **ChromaDB:** A vector database for storing and efficiently searching book embeddings.
- **Gradio:** A Python library for creating an interactive web interface.

## Features

- **Semantic Search:** Recommends books based on the semantic similarity between book descriptions and user queries (title or description).
- **Genre Filtering (Optional):** Allows users to filter recommendations by genre using zero-shot classification.
- **Web Interface:** Provides a user-friendly web interface via Gradio.

## Setup

### Prerequisites

- Python 3.8 or higher
- pip
- A Kaggle account (for downloading the dataset)

### Installation

1.  **Clone the repository (if applicable):**

    ```bash
    git clone <your_repository_url>  # If you have it on a Git hosting service
    cd semantic_book_recommender
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    .venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the dataset:**

    - You'll need the `books.csv` dataset. The script uses `kagglehub` to download it.
    - Ensure you have your Kaggle API credentials configured if running this outside of a Kaggle environment. You can find instructions on how to do this on the Kaggle website.

    ```bash
    python -c "import kagglehub; kagglehub.dataset_download('dylanjcastillo/7k-books-with-metadata', path='data')"
    ```

    - This will download the dataset to the `data/` directory.

### Usage

1.  **Preprocess data and create embeddings:**

    ```bash
    cd src
    python preprocess_and_embed.py
    cd ..
    ```

    - This script cleans the data, generates text embeddings using the "all-MiniLM-L6-v2" model, and stores them in a ChromaDB database in the `chroma_db/` directory. This step may take some time on the first run.
    - Subsequent runs will be faster as it will use the existing database if available.

2.  **Run the application:**

    ```bash
    python src/app.py
    ```

    - This will start the Gradio web interface.
    - Open the displayed URL in your web browser (usually `http://localhost:7860/`).

3.  **Interact with the application:**

    - Enter a book title or description in the text box.
    - Optionally, check the "Filter by Genre?" box and select a genre from the dropdown.
    - View the recommended books in the output table.

## Code Structure

- `data/`: Contains the `books.csv` dataset and the cleaned `df_cleaned.csv`.
- `src/`:
  - `preprocess_and_embed.py`: Handles data cleaning, embedding generation, and storage in ChromaDB.
  - `app.py`: Loads the data and embeddings, and runs the Gradio interface.
- `chroma_db/`: (Created after running `preprocess_and_embed.py`) Stores the ChromaDB database.
- `requirements.txt`: Lists the Python dependencies.
- `.gitignore`: Specifies files and directories that Git should ignore.

## Dependencies

- pandas
- numpy
- transformers
- langchain
- langchain-community
- gradio
- scikit-learn
- sentence-transformers
- chromadb
- kagglehub

## Future Improvements

- Display similarity scores for recommendations.
- Include book thumbnails and descriptions in the output.
- Allow users to provide multiple input queries.
- Implement user feedback mechanisms (thumbs up/down).
- Explore hybrid recommendation approaches (collaborative filtering, content-based filtering).
- Implement more robust error handling.
- Consider asynchronous operations for improved performance.
- Experiment with different embedding models.

## Important Notes

- Ensure you have a stable internet connection during the initial setup and when running the `preprocess_and_embed.py` script, as it downloads the embedding model.
- The `chroma_db` directory can become quite large depending on the size of your dataset.
- The application assumes that the `books.csv` file is located in the `data/` directory. If you place it elsewhere, you'll need to update the `DATA_PATH` variable in `src/preprocess_and_embed.py` and `src/app.py`.

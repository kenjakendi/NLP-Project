import os
import sys
import numpy as np
import pandas as pd
import joblib
from gensim.models import Word2Vec, FastText
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Classifier_Predictor")

# Define base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BEST_MODELS_DIR = os.path.join(BASE_DIR, 'SVM', 'best_models')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'emotion.csv')  # Path to the dataset

# Define label mapping
label_mapping = {0: 'angry', 1: 'happy', 2: 'sad', 3: 'relaxed'}

# Function to load embedding models and classifiers
def load_models():
    try:
        logger.info("Loading best models...")

        # Load Word2Vec SVM model and embedding
        word2vec_svm_path = os.path.join(BEST_MODELS_DIR, 'best_word2vec_svm.joblib')
        word2vec_embedding_path = os.path.join(BEST_MODELS_DIR, 'best_word2vec_embedding.model')

        word2vec_svm = joblib.load(word2vec_svm_path)
        word2vec_embedding = Word2Vec.load(word2vec_embedding_path)

        # Load FastText SVM model and embedding
        fasttext_svm_path = os.path.join(BEST_MODELS_DIR, 'best_fasttext_svm.joblib')
        fasttext_embedding_path = os.path.join(BEST_MODELS_DIR, 'best_fasttext_embedding.model')

        fasttext_svm = joblib.load(fasttext_svm_path)
        fasttext_embedding = FastText.load(fasttext_embedding_path)

        logger.info("Models loaded successfully.")
        return {
            "word2vec": {"svm": word2vec_svm, "embedding": word2vec_embedding},
            "fasttext": {"svm": fasttext_svm, "embedding": fasttext_embedding},
        }

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

# Function to process text into embeddings
def text_to_vector(text, embedding_model, embedding_dim):
    vectors = [embedding_model.wv[word] for word in text if word in embedding_model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(embedding_dim)

# Function to classify input text
def classify_text(input_text, models):
    logger.info(f"Classifying text: {input_text}")

    # Tokenize input (simple whitespace split for demonstration)
    tokenized_text = input_text.split()

    results = {}
    for model_type, model_data in models.items():
        embedding_model = model_data["embedding"]
        svm_model = model_data["svm"]
        embedding_dim = embedding_model.vector_size

        # Transform text into embedding
        text_vector = text_to_vector(tokenized_text, embedding_model, embedding_dim)

        # Predict using the SVM model
        numeric_prediction = svm_model.predict([text_vector])[0]
        text_prediction = label_mapping[numeric_prediction]  # Convert numeric prediction to label
        results[model_type] = text_prediction

    return results

# Main function to load models, read dataset, and classify a selected row
def main():
    try:
        # Load models
        models = load_models()

        # Load the dataset
        logger.info(f"Loading dataset from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)

        # Display a sample of the dataset and prompt the user to select a row
        logger.info("Dataset sample:")
        print(df.head())
        row_index = int(input("Enter the row index you want to classify: "))

        # Validate row index
        if row_index < 0 or row_index >= len(df):
            raise ValueError(f"Invalid row index. Must be between 0 and {len(df) - 1}.")

        # Extract the text from the selected row
        text_column = 'text'  # Replace with the actual text column name if different
        input_text = df.iloc[row_index][text_column]

        # Classify the input text
        predictions = classify_text(input_text, models)
        logger.info("Predictions:")
        for model_type, prediction in predictions.items():
            logger.info(f"{model_type}: {prediction}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()

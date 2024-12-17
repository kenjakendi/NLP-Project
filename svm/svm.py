import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from gensim.models import Word2Vec, FastText
import logging
import json
from itertools import product
from analyze_results import analyze_results  # Import funkcji analizy


# Dodanie katalogu głównego do ścieżki Pythona
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
from preprocess import Preprocess

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SVM_Classifier")

# Ścieżki plików
DATA_PATH = os.path.join(BASE_DIR, 'data', 'emotion.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'SVM', 'results.json')

# Parametry programu
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

# Rozszerzone parametry embeddingów do przetestowania
EMBEDDING_PARAMS_RANGES = {
    'model_type': ['Word2Vec', 'FastText'],  # Typ modelu embeddingu
    'vector_size': [50, 100, 200, 300],  # Rozmiar embeddingów
    'window': [2, 3, 5, 7, 10],  # Rozmiar okna
    'min_count': [1, 3, 5, 10]  # Minimalna liczba wystąpień słowa
}

# Siatka hiperparametrów SVM
SVM_PARAMS_RANGES = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'degree': [2, 3, 4]  # Tylko dla kernela poly
}

# Funkcja do tworzenia embeddingów
def text_to_vector(text, model, embedding_dim):
    vectors = [model.wv[word] for word in text if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(embedding_dim)

# Główna funkcja
def main():
    try:
        # Preprocessing danych
        logger.info("Initializing preprocessing...")
        preprocessor = Preprocess(logger, path=DATA_PATH, text_column=TEXT_COLUMN, label_column=LABEL_COLUMN)
        preprocessor.run()
        data = preprocessor.data
        logger.info("Preprocessing complete.")

        # Zamiana etykiet na wartości liczbowe
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(data["label"])

        # Przechowywanie wyników
        results = []

        # Pętla po kombinacjach parametrów embeddingów
        for params in product(*EMBEDDING_PARAMS_RANGES.values()):
            embedding_params = dict(zip(EMBEDDING_PARAMS_RANGES.keys(), params))
            logger.info(f"Training {embedding_params['model_type']} model with params: {embedding_params}")

            # Trening odpowiedniego modelu
            if embedding_params['model_type'] == 'Word2Vec':
                model = Word2Vec(
                    sentences=data["text"],
                    vector_size=embedding_params['vector_size'],
                    window=embedding_params['window'],
                    min_count=embedding_params['min_count'],
                    workers=4
                )
            elif embedding_params['model_type'] == 'FastText':
                model = FastText(
                    sentences=data["text"],
                    vector_size=embedding_params['vector_size'],
                    window=embedding_params['window'],
                    min_count=embedding_params['min_count'],
                    workers=4
                )

            # Przekształcanie tekstu na embeddingi
            logger.info("Transforming texts to embeddings...")
            X = np.array([text_to_vector(text, model, embedding_params['vector_size']) for text in data["text"]])

            # Podział danych na zbiór treningowy i testowy
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Przeszukiwanie siatki hiperparametrów dla SVM
            logger.info(f"Starting GridSearchCV for SVM with {embedding_params['model_type']} embeddings...")
            svm_model = SVC(random_state=42)

            # Logowanie informacji o parametrach siatki hiperparametrów
            logger.info(f"GridSearchCV SVM params: {SVM_PARAMS_RANGES}")

            grid_search = GridSearchCV(estimator=svm_model, param_grid=SVM_PARAMS_RANGES, cv=5, scoring='f1_weighted', n_jobs=-1)

            # Logowanie liczby kombinacji hiperparametrów przed rozpoczęciem treningu
            total_combinations = np.prod([len(v) for v in SVM_PARAMS_RANGES.values()])
            logger.info(f"Total hyperparameter combinations to test: {total_combinations}")

            grid_search.fit(X_train, y_train)

            # Logowanie najlepszego zestawu parametrów
            logger.info(f"Best SVM params for current embedding ({embedding_params['model_type']}): {grid_search.best_params_}")

            # # Przeszukiwanie siatki hiperparametrów dla SVM
            # logger.info(f"Starting GridSearchCV for SVM with {embedding_params['model_type']} embeddings...")
            # svm_model = SVC(random_state=42)
            # grid_search = GridSearchCV(estimator=svm_model, param_grid=SVM_PARAMS_RANGES, cv=5, scoring='f1_weighted', n_jobs=-1)
            # grid_search.fit(X_train, y_train)

            # Ocena najlepszego modelu
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)

            # Zapis wyników
            results.append({
                "embedding_params": embedding_params,
                "svm_params": grid_search.best_params_,
                "classification_report": report
            })

            logger.info(f"Results for {embedding_params['model_type']} with params {embedding_params} saved.")

        # Po zakończeniu eksperymentów: zapisz wyniki i wywołaj analizę
        logger.info("Saving results to file...")
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"All results saved to {OUTPUT_PATH}")

        logger.info("Running analysis on results...")
        analyze_results(OUTPUT_PATH, EMBEDDING_PARAMS_RANGES, SVM_PARAMS_RANGES)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()

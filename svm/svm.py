import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from gensim.models import Word2Vec
import logging
import json

# Dodanie katalogu głównego do ścieżki Pythona
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
from preprocess import Preprocess

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SVM_Classifier")

# **Ścieżki plików**
DATA_PATH = os.path.join(BASE_DIR, 'data', 'songs.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'SVM', 'results.json')

# **Parametry programu**
TEXT_COLUMN = "lyrics"
LABEL_COLUMN = "emotion"
EMBEDDING_DIM = 100

# **Rozszerzona siatka hiperparametrów**
PARAM_GRID = {
    'C': [0.1, 1, 10, 100, 1000],  # Regularizacja
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Rodzaje kerneli
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Parametr gamma
    'degree': [2, 3, 4]  # Stopień wielomianu dla kernela poly
}


# **Funkcja do tworzenia embeddingów**
def text_to_vector(text, model, embedding_dim):
    vectors = [model.wv[word] for word in text if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(embedding_dim)


# **Główna funkcja**
def main():
    try:
        # **Preprocessing danych**
        logger.info("Initializing preprocessing...")
        preprocessor = Preprocess(logger, path=DATA_PATH, text_column=TEXT_COLUMN, label_column=LABEL_COLUMN)
        preprocessor.run()
        data = preprocessor.data

        # **Tworzenie embeddingów Word2Vec**
        logger.info("Training Word2Vec embeddings...")
        word2vec_model = Word2Vec(
            sentences=data["text"],
            vector_size=EMBEDDING_DIM,
            window=5,
            min_count=1,
            workers=4
        )

        # Przekształcanie tekstu na embeddingi
        logger.info("Transforming texts to embeddings...")
        X = np.array([text_to_vector(text, word2vec_model, EMBEDDING_DIM) for text in data["text"]])

        # Zamiana etykiet na wartości liczbowe
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(data["label"])

        # **Podział danych na zbiór treningowy i testowy**
        logger.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # **Przeszukiwanie siatki hiperparametrów**
        logger.info("Starting hyperparameter tuning...")
        svm_model = SVC(random_state=42)
        grid_search = GridSearchCV(estimator=svm_model, param_grid=PARAM_GRID, cv=5, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Wyniki najlepszego modelu
        best_params = grid_search.best_params_
        logger.info(f"Best parameters found: {best_params}")

        # **Ocena najlepszego modelu na zbiorze testowym**
        logger.info("Evaluating the best SVM model...")
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # **Raport klasyfikacji**
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
        logger.info("Classification report generated.")

        # **Zapis wyników do pliku**
        results = {
            "best_params": best_params,
            "classification_report": report
        }
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Results saved to {OUTPUT_PATH}")

        # Wyświetlenie raportu
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()

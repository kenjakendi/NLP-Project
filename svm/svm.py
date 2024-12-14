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


# **Funkcja główna**
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
        logger.info("Word2Vec training complete.")

        # **Tworzenie embeddingów FastText**
        logger.info("Training FastText embeddings...")
        fasttext_model = FastText(
            sentences=data["text"],
            vector_size=EMBEDDING_DIM,
            window=5,
            min_count=1,
            workers=4
        )
        logger.info("FastText training complete.")

        # **Przekształcanie tekstu na embeddingi dla obu modeli**
        logger.info("Transforming texts to embeddings for Word2Vec...")
        X_word2vec = np.array([text_to_vector(text, word2vec_model, EMBEDDING_DIM) for text in data["text"]])

        logger.info("Transforming texts to embeddings for FastText...")
        X_fasttext = np.array([text_to_vector(text, fasttext_model, EMBEDDING_DIM) for text in data["text"]])

        # Zamiana etykiet na wartości liczbowe
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(data["label"])

        # **Podział danych na zbiór treningowy i testowy**
        logger.info("Splitting data into train and test sets...")
        X_w2v_train, X_w2v_test, y_train, y_test = train_test_split(X_word2vec, y, test_size=0.2, random_state=42)
        X_ft_train, X_ft_test, _, _ = train_test_split(X_fasttext, y, test_size=0.2, random_state=42)

        # **Przeszukiwanie siatki hiperparametrów dla Word2Vec**
        logger.info("Starting hyperparameter tuning for Word2Vec embeddings...")
        svm_model_w2v = SVC(random_state=42)
        grid_search_w2v = GridSearchCV(estimator=svm_model_w2v, param_grid=PARAM_GRID, cv=5, scoring='f1_weighted', n_jobs=-1)
        grid_search_w2v.fit(X_w2v_train, y_train)

        # **Przeszukiwanie siatki hiperparametrów dla FastText**
        logger.info("Starting hyperparameter tuning for FastText embeddings...")
        svm_model_ft = SVC(random_state=42)
        grid_search_ft = GridSearchCV(estimator=svm_model_ft, param_grid=PARAM_GRID, cv=5, scoring='f1_weighted', n_jobs=-1)
        grid_search_ft.fit(X_ft_train, y_train)

        # Wyniki najlepszego modelu
        best_params_w2v = grid_search_w2v.best_params_
        best_params_ft = grid_search_ft.best_params_

        logger.info(f"Best parameters for Word2Vec: {best_params_w2v}")
        logger.info(f"Best parameters for FastText: {best_params_ft}")

        # **Ocena najlepszego modelu na zbiorze testowym**
        logger.info("Evaluating the best Word2Vec SVM model...")
        best_model_w2v = grid_search_w2v.best_estimator_
        y_pred_w2v = best_model_w2v.predict(X_w2v_test)

        logger.info("Evaluating the best FastText SVM model...")
        best_model_ft = grid_search_ft.best_estimator_
        y_pred_ft = best_model_ft.predict(X_ft_test)

        # **Raporty klasyfikacji**
        report_w2v = classification_report(y_test, y_pred_w2v, target_names=label_encoder.classes_, output_dict=True)
        report_ft = classification_report(y_test, y_pred_ft, target_names=label_encoder.classes_, output_dict=True)

        logger.info("Classification reports generated.")

        # **Zapis wyników do pliku**
        results = {
            "Word2Vec": {
                "best_params": best_params_w2v,
                "classification_report": report_w2v
            },
            "FastText": {
                "best_params": best_params_ft,
                "classification_report": report_ft
            }
        }
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Results saved to {OUTPUT_PATH}")

        # Wyświetlenie raportów
        print("Classification Report for Word2Vec:")
        print(classification_report(y_test, y_pred_w2v, target_names=label_encoder.classes_))

        print("\nClassification Report for FastText:")
        print(classification_report(y_test, y_pred_ft, target_names=label_encoder.classes_))

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()

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
import joblib
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
# DATA_PATH = os.path.join(BASE_DIR, 'data', 'emotion.csv')
# OUTPUT_PATH = os.path.join(BASE_DIR, 'SVM', 'results.json')
CONFIG_PATH = os.path.join(BASE_DIR, 'SVM', 'config.json')  # Ścieżka do pliku konfiguracyjnego

# Funkcja do wczytywania konfiguracji
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

# Funkcja do tworzenia embeddingów
def text_to_vector(text, embedding_model, embedding_dim):
    vectors = [embedding_model.wv[word] for word in text if word in embedding_model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(embedding_dim)

## Główna funkcja
def main():
    try:
        # Wczytywanie konfiguracji
        config = load_config(CONFIG_PATH)
        DATA_PATH = os.path.join(BASE_DIR, config["data_path"])
        OUTPUT_PATH = os.path.join(BASE_DIR, config["output_path"])
        TEXT_COLUMN = config["text_column"]
        LABEL_COLUMN = config["label_column"]
        EMBEDDING_PARAMS_RANGES = config["embedding_params_ranges"]
        SVM_PARAMS_RANGES = config["svm_params_ranges"]

        # Preprocessing danych
        logger.info("Initializing preprocessing...")
        preprocessor = Preprocess(logger, path=DATA_PATH, text_column=TEXT_COLUMN, label_column=LABEL_COLUMN)
        preprocessor.run()
        data = preprocessor.data
        logger.info("Preprocessing complete.")

        # Zamiana etykiet na wartości liczbowe
        label_mapping = {'angry': 0, 'happy': 1, 'sad': 2, 'relaxed': 3}
        data[LABEL_COLUMN] = data[LABEL_COLUMN].map(label_mapping)
        y =data[LABEL_COLUMN].to_numpy()
         # Zamiana etykiet na wartości liczbowe
        # label_encoder = LabelEncoder()
        # y = label_encoder.fit_transform(data["label"])
        # Przechowywanie wyników
        results = []
        
        # Inicjalizacja najlepszych wyników
        best_fasttext_result = {"score": -np.inf, "svm_model": None, "embedding_model": None, "embedding_params": None}
        best_word2vec_result = {"score": -np.inf, "svm_model": None, "embedding_model": None, "embedding_params": None}

        # Pętla po kombinacjach parametrów embeddingów
        for params in product(*EMBEDDING_PARAMS_RANGES.values()):
            embedding_params = dict(zip(EMBEDDING_PARAMS_RANGES.keys(), params))
            logger.info(f"Training {embedding_params['model_type']} embedding model with params: {embedding_params}")

            # Trening odpowiedniego modelu
            if embedding_params['model_type'] == 'Word2Vec':
                embedding_model = Word2Vec(
                    sentences=data[TEXT_COLUMN],
                    vector_size=embedding_params['vector_size'],
                    window=embedding_params['window'],
                    min_count=embedding_params['min_count'],
                    workers=2
                )
            elif embedding_params['model_type'] == 'FastText':
                embedding_model = FastText(
                    sentences=data[TEXT_COLUMN],
                    vector_size=embedding_params['vector_size'],
                    window=embedding_params['window'],
                    min_count=embedding_params['min_count'],
                    workers=2
                )

            # Przekształcanie tekstu na embeddingi
            logger.info("Transforming texts to embeddings...")
            X = np.array([text_to_vector(text, embedding_model, embedding_params['vector_size']) for text in data[TEXT_COLUMN]])

            # Podział danych na zbiór treningowy i testowy
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Przeszukiwanie siatki hiperparametrów dla SVM
            logger.info(f"Starting GridSearchCV for SVM with {embedding_params['model_type']} embeddings...")
            svm_model = SVC(random_state=42)

            grid_search = GridSearchCV(estimator=svm_model, param_grid=SVM_PARAMS_RANGES, cv=5, scoring='f1_weighted', n_jobs=2)
            grid_search.fit(X_train, y_train)

            # Ocena najlepszego modelu
            best_svm_model = grid_search.best_estimator_
            
             # Generowanie predykcji dla zbioru uczącego
            y_train_pred = best_svm_model.predict(X_train)
            train_report = classification_report(y_train, y_train_pred, target_names=label_mapping, output_dict=True)

            # Generowanie predykcji dla zbioru testowego
            y_test_pred = best_svm_model.predict(X_test)
            test_report = classification_report(y_test, y_test_pred, target_names=label_mapping, output_dict=True)
            test_f1 = test_report["weighted avg"]["f1-score"]

            # Zapis wyników
            results.append({
                "embedding_params": embedding_params,
                "svm_params": grid_search.best_params_,
                "train_classification_report": train_report,
                "test_classification_report": test_report

            })
            logger.info(f"Results for {embedding_params['model_type']} with params {embedding_params} saved.")


            # Aktualizacja najlepszego modelu dla FastText
            if embedding_params['model_type'] == 'FastText' and test_f1 > best_fasttext_result["score"]:
                best_fasttext_result = {
                    "score": test_f1,
                    "svm_model": best_svm_model,
                    "embedding_model": embedding_model,
                    "embedding_params": embedding_params
                }

            # Aktualizacja najlepszego modelu dla Word2Vec
            if embedding_params['model_type'] == 'Word2Vec' and test_f1 > best_word2vec_result["score"]:
                best_word2vec_result = {
                    "score": test_f1,
                    "svm_model": best_svm_model,
                    "embedding_model": embedding_model,
                    "embedding_params": embedding_params
                }



        # Zapis wyników do pliku
        logger.info("Saving results to file...")
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"All results saved to {OUTPUT_PATH}")


        # Zapis najlepszych modeli na koniec
        # Tworzenie folderu 'svm/best_models', jeśli nie istnieje
        best_models_dir = os.path.join(BASE_DIR, 'SVM', 'best_models')
        os.makedirs(best_models_dir, exist_ok=True)

        if best_fasttext_result["svm_model"]:
            logger.info("Saving best FastText SVM model and embedding model...")
            joblib.dump(best_fasttext_result["svm_model"], os.path.join(best_models_dir, 'best_fasttext_svm.joblib'))
            best_fasttext_result["embedding_model"].save(os.path.join(best_models_dir, 'best_fasttext_embedding.model'))

        if best_word2vec_result["svm_model"]:
            logger.info("Saving best Word2Vec SVM model and embedding model...")
            joblib.dump(best_word2vec_result["svm_model"], os.path.join(best_models_dir, 'best_word2vec_svm.joblib'))
            best_word2vec_result["embedding_model"].save(os.path.join(best_models_dir, 'best_word2vec_embedding.model'))

        logger.info("Best models saved successfully.")


        # Analiza wyników
        logger.info("Running analysis on results...")
        analyze_results(OUTPUT_PATH, EMBEDDING_PARAMS_RANGES, SVM_PARAMS_RANGES)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()

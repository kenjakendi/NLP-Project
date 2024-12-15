import os
import json
import pandas as pd

# Ścieżka do pliku z wynikami
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
RESULTS_PATH = os.path.join(BASE_DIR, 'results.json')

def analyze_results(results_path):
    try:
        # Wczytaj dane z pliku JSON
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Inicjalizacja najlepszych wyników
        best_word2vec = None
        best_fasttext = None
        best_word2vec_score = -1
        best_fasttext_score = -1

        # Przeglądanie wyników
        for result in results:
            embedding_params = result['embedding_params']
            classification_report = result['classification_report']
            f1_score = classification_report['weighted avg']['f1-score']
            
            svm_params = result['svm_params']

            if embedding_params['model_type'] == 'Word2Vec':
                if f1_score > best_word2vec_score:
                    best_word2vec_score = f1_score
                    best_word2vec = {
                        "embedding_params": embedding_params,
                        "svm_params": svm_params,
                        "classification_report": classification_report
                    }
            
            elif embedding_params['model_type'] == 'FastText':
                if f1_score > best_fasttext_score:
                    best_fasttext_score = f1_score
                    best_fasttext = {
                        "embedding_params": embedding_params,
                        "svm_params": svm_params,
                        "classification_report": classification_report
                    }
        
        # Wyświetlenie najlepszych wyników
        print("\n" + "-"*50 )
        print("Best Word2Vec classifier:")
        display_classification_report(best_word2vec['classification_report'], best_word2vec['svm_params'], best_word2vec['embedding_params'])
        
        print("\n" + "-"*50 )
        print("Best FastText classifier:")
        display_classification_report(best_fasttext['classification_report'], best_fasttext['svm_params'], best_fasttext['embedding_params'])
    
    except Exception as e:
        print(f"An error occurred: {e}")

def display_classification_report(report, svm_params, embedding_params):
    """
    Funkcja do wyświetlania klasyfikacji w formie tabeli
    oraz parametrów SVM i embeddingu
    """
    # Konwersja klasyfikacji na DataFrame
    report_df = pd.DataFrame(report).transpose()

    # Tylko interesujące nas miary
    report_df = report_df[['precision', 'recall', 'f1-score', 'support']]

    # Wyświetlenie parametrów embeddingu w jednej linii
    embedding_str = (f"Vector size: {embedding_params.get('vector_size', 'N/A')}, "
                     f"Window size: {embedding_params.get('window', 'N/A')}, "
                     f"Min count: {embedding_params.get('min_count', 'N/A')}")
    
    print(f"\nEmbedding Parameters: ")
    print(embedding_str)
    # Wyświetlenie parametrów SVM
    print("\nSVM Parameters:")
    print(f"C: {svm_params['C']}, Kernel: {svm_params['kernel']}, Gamma: {svm_params['gamma']}")

    # Wyświetlenie miar
    print("Classification Report:")
    print(report_df.to_string(index=True))

if __name__ == "__main__":
    analyze_results(RESULTS_PATH)

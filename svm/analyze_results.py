import os
import json
import pandas as pd

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_RESULTS_PATH = os.path.join(BASE_DIR, "results.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "best_parameters.json")


def analyze_results(results_path, embedding_param_ranges=None, svm_param_ranges=None):
    try:
        # Jeśli zakresy parametrów nie zostały podane, spróbuj je wywnioskować z pliku wyników
        if embedding_param_ranges is None or svm_param_ranges is None:
            with open(results_path, 'r') as f:
                results = json.load(f)

            # Inicjalizacja zbiorów dla zakresów parametrów
            embedding_param_ranges = embedding_param_ranges or {}
            svm_param_ranges = svm_param_ranges or {}

            for result in results:
                embedding_params = result['embedding_params']
                svm_params = result['svm_params']

                # Aktualizacja zakresów embeddingów
                for param, value in embedding_params.items():
                    embedding_param_ranges.setdefault(param, set()).add(value)

                # Aktualizacja zakresów SVM
                for param, value in svm_params.items():
                    svm_param_ranges.setdefault(param, set()).add(value)

            # Konwersja zbiorów na listy (dla czytelniejszego wyświetlania)
            embedding_param_ranges = {param: sorted(values) for param, values in embedding_param_ranges.items()}
            svm_param_ranges = {param: sorted(values) for param, values in svm_param_ranges.items()}

        # Wyświetlenie zakresu badanych parametrów
        if embedding_param_ranges and svm_param_ranges:
            print("\n" + "="*50)
            print("Experiment Parameter Ranges:")
            print("\nEmbedding Parameters:")
            for param, values in embedding_param_ranges.items():
                print(f"  {param}: {values}")
            
            print("\nSVM Parameters:")
            for param, values in svm_param_ranges.items():
                print(f"  {param}: {values}")
            print("="*50 + "\n")
        
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
    
         # Zapisz najlepsze parametry oraz zakresy do pliku JSON
        best_params = {
            "embedding_param_ranges": embedding_param_ranges,
            "svm_param_ranges": svm_param_ranges,
            "best_word2vec": best_word2vec,
            "best_fasttext": best_fasttext
        }

        with open(OUTPUT_PATH, 'w') as output_file:
            json.dump(best_params, output_file, indent=4)
        print(f"\nBest parameters and ranges saved to: {OUTPUT_PATH}")

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
    # Domyślne wywołanie analizy z domyślną ścieżką
    print("Running analysis with default results path...\n")
    analyze_results(DEFAULT_RESULTS_PATH)
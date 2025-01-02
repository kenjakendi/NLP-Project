import os
import json
import pandas as pd

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_RESULTS_PATH = os.path.join(BASE_DIR, "results.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "best_parameters.json")
TXT_OUTPUT_PATH = os.path.join(BASE_DIR, "best_parameters_report.txt")  # Ścieżka do pliku tekstowego


def analyze_results(results_path, embedding_param_ranges=None, svm_param_ranges=None):
    if embedding_param_ranges is None or svm_param_ranges is None:
        with open(results_path, 'r') as f:
            results = json.load(f)

        embedding_param_ranges = embedding_param_ranges or {}
        svm_param_ranges = svm_param_ranges or {}

        for result in results:
            embedding_params = result['embedding_params']
            svm_params = result['svm_params']

            for param, value in embedding_params.items():
                embedding_param_ranges.setdefault(param, set()).add(value)
            for param, value in svm_params.items():
                svm_param_ranges.setdefault(param, set()).add(value)

        embedding_param_ranges = {param: values for param, values in embedding_param_ranges.items()}
        svm_param_ranges = {param: values for param, values in svm_param_ranges.items()}

    print("\n" + "=" * 50)
    print("Experiment Parameter Ranges:")
    print("\nEmbedding Parameters:")
    for param, values in embedding_param_ranges.items():
        print(f"  {param}: {values}")

    print("\nSVM Parameters:")
    for param, values in svm_param_ranges.items():
        print(f"  {param}: {values}")
    print("=" * 50 + "\n")

    with open(results_path, 'r') as f:
        results = json.load(f)

    best_word2vec = None
    best_fasttext = None
    best_word2vec_score = -1
    best_fasttext_score = -1

    for result in results:
        embedding_params = result['embedding_params']
        train_report = result.get('train_classification_report', {})
        test_report = result['test_classification_report']
        test_f1_score = test_report['weighted avg']['f1-score']

        svm_params = result['svm_params']

        if embedding_params['model_type'] == 'Word2Vec':
            if test_f1_score > best_word2vec_score:
                best_word2vec_score = test_f1_score
                best_word2vec = {
                    "embedding_params": embedding_params,
                    "svm_params": svm_params,
                    "train_classification_report": train_report,
                    "test_classification_report": test_report
                }

        elif embedding_params['model_type'] == 'FastText':
            if test_f1_score > best_fasttext_score:
                best_fasttext_score = test_f1_score
                best_fasttext = {
                    "embedding_params": embedding_params,
                    "svm_params": svm_params,
                    "train_classification_report": train_report,
                    "test_classification_report": test_report
                }

    print("\n" + "-" * 50)
    print("Best Word2Vec classifier:")
    display_classification_reports(best_word2vec)

    print("\n" + "-" * 50)
    print("Best FastText classifier:")
    display_classification_reports(best_fasttext)

    best_params = {
        "embedding_param_ranges": {param: list(values) for param, values in embedding_param_ranges.items()},
        "svm_param_ranges": {param: list(values) for param, values in svm_param_ranges.items()},
        "best_word2vec": best_word2vec,
        "best_fasttext": best_fasttext
    }

    with open(OUTPUT_PATH, 'w') as output_file:
        json.dump(best_params, output_file, indent=4)
    print(f"\nBest parameters and ranges saved to: {OUTPUT_PATH}")


def display_classification_reports(result):
    """
    Wyświetla raporty klasyfikacji dla zbiorów uczącego i testowego
    """
    train_report = pd.DataFrame(result['train_classification_report']).transpose()
    train_report = train_report[['precision', 'recall', 'f1-score', 'support']]

    test_report = pd.DataFrame(result['test_classification_report']).transpose()
    test_report = test_report[['precision', 'recall', 'f1-score', 'support']]

    embedding_params = result['embedding_params']
    svm_params = result['svm_params']

    print("\nEmbedding Parameters:")
    print(f"  Model: {embedding_params.get('model_type', 'N/A')}")
    print(f"  Vector size: {embedding_params.get('vector_size', 'N/A')}")
    print(f"  Window size: {embedding_params.get('window', 'N/A')}")
    print(f"  Min count: {embedding_params.get('min_count', 'N/A')}")

    print("\nSVM Parameters:")
    print(f"  C: {svm_params['C']}")
    print(f"  Kernel: {svm_params['kernel']}")
    print(f"  Gamma: {svm_params['gamma']}")

    print("\nTraining Set Classification Report:")
    print(train_report.to_string(index=True))

    print("\nTest Set Classification Report:")
    print(test_report.to_string(index=True))

    with open(TXT_OUTPUT_PATH, 'a') as txt_file:
        txt_file.write("\nEmbedding Parameters:\n")
        txt_file.write(f"  Model: {embedding_params.get('model_type', 'N/A')}\n")
        txt_file.write(f"  Vector size: {embedding_params.get('vector_size', 'N/A')}\n")
        txt_file.write(f"  Window size: {embedding_params.get('window', 'N/A')}\n")
        txt_file.write(f"  Min count: {embedding_params.get('min_count', 'N/A')}\n")

        txt_file.write("\nSVM Parameters:\n")
        txt_file.write(f"  C: {svm_params['C']}\n")
        txt_file.write(f"  Kernel: {svm_params['kernel']}\n")
        txt_file.write(f"  Gamma: {svm_params['gamma']}\n")

        txt_file.write("\nTraining Set Classification Report:\n")
        txt_file.write(train_report.to_string(index=True) + "\n")

        txt_file.write("\nTest Set Classification Report:\n")
        txt_file.write(test_report.to_string(index=True) + "\n")
        txt_file.write("=" * 50 + "\n")

    

if __name__ == "__main__":
    # Domyślne wywołanie analizy z domyślną ścieżką
    print("Running analysis with default results path...\n")
    analyze_results(DEFAULT_RESULTS_PATH)
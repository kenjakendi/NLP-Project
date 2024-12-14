import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from gensim.models import Word2Vec
import logging

# # Dodanie katalogu głównego do ścieżki Pythona
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
from preprocess import Preprocess


# Konfiguracja loggera
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SVM_Classifier")

# Ścieżki plików
DATA_PATH = os.path.join(BASE_DIR, 'data', 'songs.csv')

# Parametry kolumn w pliku CSV
TEXT_COLUMN = "lyrics"
LABEL_COLUMN = "emotion"

# **Preprocessing danych**
logger.info("Initializing preprocessing...")
preprocessor = Preprocess(logger, path=DATA_PATH, text_column=TEXT_COLUMN, label_column=LABEL_COLUMN)
preprocessor.run()
data = preprocessor.data

# **Tworzenie embeddingów Word2Vec**
logger.info("Training Word2Vec embeddings...")
embedding_dim = 100
word2vec_model = Word2Vec(
    sentences=data["text"],  # Lista tokenizowanych tekstów
    vector_size=embedding_dim,
    window=5,
    min_count=1,
    workers=4
)

# Funkcja zamieniająca tekst na średnią embeddingów słów
def text_to_vector(text, model, embedding_dim):
    vectors = [model.wv[word] for word in text if word in model.wv]
    if vectors:
        a = np.mean(vectors, axis=0)
        return np.mean(vectors, axis=0)  # Średnia z embeddingów słów
    else:
        return np.zeros(embedding_dim)  # Pusty wektor, gdy brak znanych słów

# Przekształcanie danych tekstowych na embeddingi
logger.info("Transforming texts to embeddings...")
X = np.array([text_to_vector(text, word2vec_model, embedding_dim) for text in data["text"]])

# Zamiana etykiet na wartości liczbowe
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["label"])

# **Podział danych na zbiór treningowy i testowy**
logger.info("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **Trenowanie klasyfikatora SVM**
logger.info("Training SVM classifier...")
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# **Ocena klasyfikatora**
logger.info("Evaluating SVM classifier...")
y_pred = svm_model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

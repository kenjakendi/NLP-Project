import pandas as pd
from collections import Counter
from torch.utils.data import DataLoader
from lyric_dataset import LyricDataset
from bert_classifier import BertClassifier
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import yaml

if __name__ == "__main__":
    # Załaduj paramy
    with open("bert/bert_params.yaml", "r") as file:
        config = yaml.safe_load(file)
    label_column = config["label_column"]
    text_column = config["text_column"]

    # Przygotuj dane
    data = pd.read_csv("data/emotion.csv")
    emotion_labels = {label: idx for idx, label in enumerate(data[label_column].unique())}
    data['label'] = data[label_column].map(emotion_labels)

    # Podział danych
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        data[text_column], data['label'], test_size=config["split_size"], random_state=config["split_seed"]
    )
    # Zliczanie liczności klas
    train_class_counts = Counter(train_labels)
    test_class_counts = Counter(test_labels)
    # Wypisywanie liczności klas
    print("Liczność klas w zbiorze treningowym:")
    for class_label, count in train_class_counts.items():
        print(
            f"Klasa {class_label} ({list(emotion_labels.keys())[list(emotion_labels.values()).index(class_label)]}): {count} próbek")
    print("\nLiczność klas w zbiorze testowym:")
    for class_label, count in test_class_counts.items():
        print(
            f"Klasa {class_label} ({list(emotion_labels.keys())[list(emotion_labels.values()).index(class_label)]}): {count} próbek")

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(config["name"])
    # Zbiory danych w loadery
    train_loader = DataLoader(
        LyricDataset(train_texts.values, train_labels.values, tokenizer, config["max_len"]),
        batch_size=config["batch_size"],
        shuffle=True
    )
    test_loader = DataLoader(
        LyricDataset(test_texts.values, test_labels.values, tokenizer, config["max_len"]),
        batch_size=config["batch_size"]
    )

    # Inicjalizacja Berta
    classifier = BertClassifier(
        config_path="bert/bert_params.yaml",
        train_loader=train_loader,
        test_loader=test_loader,
        emotion_labels=emotion_labels
    )
    # Uczenie i wyświetlenie wyników
    classifier.train()
    acc, report = classifier.evaluate()
    print(f"Test accuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)

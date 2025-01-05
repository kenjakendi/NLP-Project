import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import pandas as pd
from bert_classifier import BertClassifier, BertForSequenceClassification
from lyric_dataset import LyricDataset
import yaml
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Załaduj paramy
    with open("bert/bert_params.yaml", "r") as file:
        config = yaml.safe_load(file)
    label_column = config["label_column"]
    text_column = config["text_column"]
    model_path = config["model_path"]

    # Przygotuj dane
    data = pd.read_csv("data/emotion.csv")
    emotion_labels = {label: idx for idx, label in enumerate(data[label_column].unique())}
    data['label'] = data[label_column].map(emotion_labels)

    # Podział danych
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        data[text_column], data['label'], test_size=config["split_size"], random_state=config["split_seed"]
    )

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # Zbiory danych w loadery
    test_loader = DataLoader(
        LyricDataset(test_texts.values, test_labels.values, tokenizer, config["max_len"]),
        batch_size=config["batch_size"]
    )
    train_loader = DataLoader(
        LyricDataset(train_texts.values, train_labels.values, tokenizer, config["max_len"]),
        batch_size=config["batch_size"],
        shuffle=True
    )
    # Inicjalizacja klasyfikatora
    classifier = BertClassifier(
        config_path=None,
        train_loader=train_loader,
        test_loader=test_loader,
        emotion_labels=emotion_labels
    )
    my_model = BertForSequenceClassification.from_pretrained(model_path)
    my_model.to(torch.device("cuda"))
    classifier.model = my_model

    # Ewaluacja
    accuracy, report = classifier.evaluate()

    # Wyświetlenie wyników
    print(f"Test Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

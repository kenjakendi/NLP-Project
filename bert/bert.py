import pandas as pd
from torch.utils.data import DataLoader
from lyric_dataset import LyricDataset
from bert_classifier import BertClassifier
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import yaml

if __name__ == "__main__":
    # Load data
    with open("bert/bert_params.yaml", "r") as file:
        config = yaml.safe_load(file)
    # Prepare Data
    data = pd.read_csv("data/songs.csv")
    emotion_labels = {label: idx for idx, label in enumerate(data["emotion"].unique())}
    data['label'] = data["emotion"].map(emotion_labels)
    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        data["lyrics"], data['label'], test_size=config["split_size"], random_state=config["split_seed"]
    )
    # Get tokenizer
    tokenizer = BertTokenizer.from_pretrained(config["name"])
    train_loader = DataLoader(
        LyricDataset(train_texts.values, train_labels.values, tokenizer, config["max_len"]),
        batch_size=config["batch_size"],
        shuffle=True
    )
    test_loader = DataLoader(
        LyricDataset(test_texts.values, test_labels.values, tokenizer, config["max_len"]),
        batch_size=config["batch_size"]
    )
    # Get BertClassifier
    classifier = BertClassifier(
        config_path="bert/bert_params.yaml",
        train_loader=train_loader,
        test_loader=test_loader,
        emotion_labels=emotion_labels
    )
    # Train and report after training
    classifier.train()
    acc, report = classifier.evaluate()
    print(f"Test accuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)
    # Save to file
    classifier.save_model(f"bert/output/emotion_classification_model_{classifier.max_len}_{classifier.batch_size}_{classifier.epochs}_{classifier.dropout_rate}_{classifier.learning_rate}")

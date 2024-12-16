import pandas as pd
import re
from collections import Counter
from torch.utils.data import DataLoader
from lyric_dataset import LyricDataset
from bert_classifier import BertClassifier
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import yaml

if __name__ == "__main__":
    # Load params
    with open("bert/bert_params.yaml", "r") as file:
        config = yaml.safe_load(file)
    label_column = config["label_column"]
    text_column = config["text_column"]

    # Prepare Data
    data = pd.read_csv("data/emotion.csv")
    emotion_labels = {label: idx for idx, label in enumerate(data[label_column].unique())}
    data['label'] = data[label_column].map(emotion_labels)

    # Split data
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
    #self.save_model(f"bert/models/emotion_classification_model_Epoch{epoch+1}_len{self.max_len}_bs{self.batch_size}_ep{self.epochs}_dr{self.dropout_rate}_lr{str(self.learning_rate).replace('.', 'p')}")

"""
Epoch 1/5
Training: 100%|██████████| 84/84 [38:03<00:00, 27.18s/it]
Training loss: 1.2304
Accuracy: 0.4246
F1 score: 0.4218
Epoch time: 2283.13 seconds
Epoch 2/5
Training: 100%|██████████| 84/84 [38:36<00:00, 27.58s/it]
Training loss: 0.8748
Accuracy: 0.6527
F1 score: 0.6523
Epoch time: 2316.73 seconds
Epoch 3/5
Training: 100%|██████████| 84/84 [39:47<00:00, 28.42s/it]
Training:   0%|          | 0/84 [00:00<?, ?it/s]Training loss: 0.5896
Accuracy: 0.8005
F1 score: 0.8002
Epoch time: 2387.25 seconds
Epoch 4/5
Training: 100%|██████████| 84/84 [40:04<00:00, 28.63s/it]
Training:   0%|          | 0/84 [00:00<?, ?it/s]Training loss: 0.3514
Accuracy: 0.8912
F1 score: 0.8914
Epoch time: 2404.63 seconds
Epoch 5/5
Training: 100%|██████████| 84/84 [39:54<00:00, 28.50s/it]
Training loss: 0.1794
Accuracy: 0.9490
F1 score: 0.9490
Epoch time: 2394.28 seconds
Evaluating: 100%|██████████| 21/21 [00:33<00:00,  1.60s/it]
Test accuracy: 0.6347
Classification Report:
              precision    recall  f1-score   support
       happy       0.71      0.76      0.74        79
         sad       0.63      0.44      0.52        96
       angry       0.82      0.69      0.75        84
     relaxed       0.46      0.69      0.56        75
    accuracy                           0.63       334
   macro avg       0.66      0.65      0.64       334
weighted avg       0.66      0.63      0.64       334   

import pandas as pd
from transformers import BertTokenizer

def filter_long_texts(input_csv, output_csv, text_column, label_column, max_tokens=512):
    # Wczytaj dane z pliku CSV
    data = pd.read_csv(input_csv)

    # Inicjalizacja tokenizatora BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for _, row in data.iterrows():
        # Pobieranie tekstu i etykiety
        text = row[text_column]
        label = row[label_column]
        # Usuwanie fragmentów w nawiasach kwadratowych
        text = re.sub(r'\[.*?\]', '', text)
    # Filtruj teksty, które mają mniej niż `max_tokens` tokenów
    filtered_data = []
    for _, row in data.iterrows():
        # Pobieranie tekstu i etykiety
        text = row[text_column]
        label = row[label_column]

        # Usuwanie fragmentów w nawiasach kwadratowych
        text = re.sub(r'\[.*?\]', '', text)

        # Tokenizowanie tekstu
        tokenized = tokenizer(text, truncation=False, return_tensors='pt')
        num_tokens = tokenized['input_ids'].shape[1]

        # Zachowaj tylko te, które mają <= `max_tokens`
        if num_tokens <= max_tokens:
            filtered_data.append({'label': label, 'text': text.strip()})

    # Tworzenie DataFrame z przefiltrowanymi danymi
    filtered_df = pd.DataFrame(filtered_data)

    # Zapisz do nowego pliku CSV
    filtered_df.to_csv(output_csv, index=False)
    print(f"Przefiltrowane dane zapisano w pliku: {output_csv}")
# Wywołanie funkcji
filter_long_texts(
    input_csv='data/filtered_dataset_with_lyrics.csv',  # Plik wejściowy
    output_csv='data/filtered_dataset_with_lyrics_512.csv',  # Plik wyjściowy
    text_column='lyrics',  # Kolumna z tekstami
    label_column='emotion_2Q'  # Kolumna z labelami
)
"""



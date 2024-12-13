import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertForSequenceClassification
#from transformers import AdamW
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import yaml

# max size to 746, średnia liczba słów to 139?
# Wczytaj konfigurację z pliku YAML
with open("bert/bert_params.yaml", "r") as file:
    config = yaml.safe_load(file)

# 1. Załadowanie danych
class SongDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 2. Przygotowanie danych
# Wczytaj dane z pliku CSV
data = pd.read_csv("data/songs.csv")

# Kodowanie emocji na liczby, dodanie kolumny z nimi
emotion_labels = {label: idx for idx, label in enumerate(data["emotion"].unique())}
data['label'] = data["emotion"].map(emotion_labels)

# Podział na zbiory treningowe i testowe
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data["lyrics"], data['label'], test_size=0.2, random_state=42
)

# Tokenizer i konfiguracja
model_name = config["name"]
max_len = config["max_len"]
batch_size = config["batch_size"]
epochs = config["epochs"]
tokenizer = BertTokenizer.from_pretrained(model_name)
learning_rate = float(config["learning_rate"])

# Dataset i DataLoader
dataset_train = SongDataset(train_texts.values, train_labels.values, tokenizer, max_len)
dataset_test = SongDataset(test_texts.values, test_labels.values, tokenizer, max_len)

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=batch_size)

# 3. Model i optymalizacja
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(emotion_labels))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate)

# 4. Funkcja trenowania i ewaluacji
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=emotion_labels.keys())
    return acc, report

# 5. Trenowanie
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_loss = train_epoch(model, train_loader, optimizer)
    print(f"Training loss: {train_loss:.4f}")

# 6. Ewaluacja
acc, report = evaluate(model, test_loader)
print(f"Test accuracy: {acc:.4f}")
print("Classification Report:")
print(report)

# 7. Zapis modelu
model.save_pretrained("bert/output/emotion_classification_model")
tokenizer.save_pretrained("bert/output/emotion_classification_model")

import yaml
import time
import torch
import os
from tqdm import tqdm
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, f1_score


class BertClassifier:
    def __init__(self, config_path, train_loader, test_loader, emotion_labels):
        # Load configuration
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        self.model_name = config["name"]
        self.max_len = config["max_len"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.learning_rate = float(config["learning_rate"])
        self.dropout_rate = config["dropout_rate"]

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.emotion_labels = emotion_labels

        # Initialize device
        self.device = torch.device("cuda")
        # Initialize model
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(emotion_labels),
            hidden_dropout_prob=self.dropout_rate,
            attention_probs_dropout_prob=self.dropout_rate,
        ).to(self.device)
        # Initialize optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

    def save_model(self, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.model.save_pretrained(output_path)
        BertTokenizer.from_pretrained(self.model_name).save_pretrained(output_path)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        for batch in tqdm(self.train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            # Collect predictions and true labels for metrics
            logits = outputs.logits
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Calculate accuracy and F1 score
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        # Generate classification report
        report = classification_report(all_labels, all_preds, target_names=self.emotion_labels.keys())
        return total_loss / len(self.train_loader), acc, f1, report

    def evaluate(self):
        self.model.eval()
        predictions, true_labels = [], []
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions, target_names=self.emotion_labels.keys())
        return acc, report

    def train(self):
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            """
            # =======================================================================
            # Zamrażanie niższych warstw BERT-a przez pierwsze dwie epoki
            if epoch < 2:
                print("Freezing BERT layers.")
                for name, param in self.model.named_parameters():
                    if "classifier" not in name:  # "classifier" to nazwa warstwy liniowej w BERT
                        param.requires_grad = False
            else:
                print("Unfreezing BERT layers.")
                for name, param in self.model.named_parameters():
                    param.requires_grad = True
            # ===========================================================================
            """
            start_time = time.time()

            train_loss, acc, f1, train_report = self.train_epoch()

            epoch_duration = time.time() - start_time
            print(f"Training loss: {train_loss:.4f}")
            print(f"Training accuracy: {acc:.4f}")
            print(f"Training F1 score: {f1:.4f}")
            print(f"Training Classification Report:\n{train_report}")
            print(f"Training Epoch time: {epoch_duration:.2f} seconds")
            acc, report = self.evaluate()
            print(f"Test accuracy: {acc:.4f}")
            print("Test classification Report:")
            print(f"Test Classification Report:\n{report}")
            self.save_model(f"bert/models/emotion_classification_model_Epoch{epoch + 1}_len{self.max_len}_bs{self.batch_size}_ep{self.epochs}_dr{self.dropout_rate}_lr{str(self.learning_rate).replace('.', 'p')}")
            print("=====next epoch=====")


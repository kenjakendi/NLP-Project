import os
import re
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# import data from csv -> z tego mamy kolumne z tekstem i labelami
def load_data(data_file):
    df = pd.read_csv(data_file)
    texts = df['lyrics'].tolist()
    labels = df['emotion_4Q'].tolist()
    return texts, labels
def remove_annotations(text):
    return re.sub(r'\[.*?\]', '', text)
data_file = "data/filtered_dataset_with_lyrics.csv"
texts, labels = load_data(data_file)
texts = [remove_annotations(text) for text in texts]

# Create a custom dataset class for text classification
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}


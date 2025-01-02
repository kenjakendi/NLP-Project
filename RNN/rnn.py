import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import json

from gensim.models import Word2Vec, FastText
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
from preprocess import Preprocess


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        out = torch.softmax(out, dim=1)
        return out
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        out = torch.softmax(out, dim=1)
        return out


def create_embedding_matrix(tokenized_texts, model):
    embedding_dim = model.vector_size
    embeddings = []
    for tokens in tokenized_texts:
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        if vectors:
            embeddings.append(np.mean(vectors, axis=0))
        else:
            embeddings.append(np.zeros(embedding_dim))
    return np.array(embeddings)


if __name__ == "__main__":
    level = logging.DEBUG
    logger = logging.getLogger()
    console_handler = logging.StreamHandler()
    logger.setLevel(level)
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    with open('params.json', 'r') as file:
        params = json.load(file)

    pp = Preprocess(logger=logger ,path=params["data_path"], text_column=params["text_column"], label_column=params["label_column"])
    pp.run()

    logger.info("Loading RNN settings")
    with open('RNN/rnn_params.json', 'r') as file:
        params = json.load(file)

    run_mode = params["run_mode"]
    hidden_size = params["hidden_size"]
    epochs = params["epochs"]
    test_size = params["test_size"]
    model_file_path = params["model_file_path"]
    embedding_type = params["embedding_type"]
    learning_rate = params["learning_rate"]
    vector_size = params["vector_size"]
    window = params["window"]
    min_count = params["min_count"]
    num_layers = params["num_layers"]
    unsqueeze = params["unsqueeze"]
    dropout = params["dropout"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    tokenized_lyrics = pp.data["text"]
    emotion_labels = pp.data["label"]

    logger.info(f"Converting tokenized lyrics to embeddings with {embedding_type}")
    if embedding_type == "word2vec":
        embedding_model = Word2Vec(sentences=tokenized_lyrics, vector_size=vector_size, window=window, min_count=min_count, workers=4)
    else:
        embedding_model = FastText(sentences=tokenized_lyrics, vector_size=vector_size, window=window, min_count=min_count, workers=4)

    lyrics_embeddings = create_embedding_matrix(tokenized_lyrics, embedding_model)

    label_encoder = LabelEncoder()
    emotion_labels = label_encoder.fit_transform(emotion_labels)

    logger.info("Preparing data for training")
    lyrics_train, lyrics_test, emotion_train, emotion_test = train_test_split(lyrics_embeddings, emotion_labels, test_size=test_size)

    lyrics_train = torch.tensor(lyrics_train, dtype=torch.float32).to(device)
    lyrics_test = torch.tensor(lyrics_test, dtype=torch.float32).to(device)
    emotion_train = torch.tensor(emotion_train, dtype=torch.long).to(device)
    emotion_test = torch.tensor(emotion_test, dtype=torch.long).to(device)

    lyrics_train = lyrics_train.unsqueeze(unsqueeze)
    lyrics_test = lyrics_test.unsqueeze(unsqueeze)

    logger.info(f"Chosen run mode: {run_mode}")

    input_size = lyrics_train.shape[2]
    output_size = len(label_encoder.classes_)

    logger.info("Creating RNN model.")

    # model = RNN(
    #     input_size=input_size,
    #     hidden_size=hidden_size,
    #     output_size=output_size,
    #     num_layers=num_layers,
    # ).to(device=device)

    model = LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device=device)
    
    if "train" in run_mode:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        logger.info("Training has been STARTED...")
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(lyrics_train)
            loss = criterion(outputs, emotion_train)
            loss.backward()
            optimizer.step()

            if (epoch+1) % 100 == 0:
                logger.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

        logger.info("Training has been FINISHED.")

        logger.info("Saving trained model")
        model_file_name = f'{model_file_path}rnn_model_{embedding_type}_{epochs}_{hidden_size}_{num_layers}.pth'
        torch.save(model.state_dict(), model_file_name)
        logger.info(f"Model saved to {model_file_name}")


    if "test" in run_mode:
        logger.info("Evaluating the model on test data")
        model_file_name = f'{model_file_path}rnn_model_{embedding_type}_{epochs}_{hidden_size}_{num_layers}.pth'
        model.load_state_dict(torch.load(model_file_name, weights_only=True))
        logger.info(f"Model loaded from {model_file_name}")

        model.eval()
        with torch.no_grad():
            test_outputs = model(lyrics_test)
            _, predicted = torch.max(test_outputs, 1)
            correct = (predicted == emotion_test).sum().item()
            accuracy = correct / len(emotion_test) * 100

        logger.info(f"Test Accuracy: {accuracy:.2f}%")

        print("Classification report")
        print(classification_report(emotion_test.cpu(), predicted.cpu(), target_names=label_encoder.classes_))

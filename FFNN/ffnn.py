import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import json

from gensim.models import Word2Vec, FastText
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from preprocess import Preprocess


class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out - torch.softmax(out, dim=1)
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

    logger.info("Loading FFNN settings")
    with open('FFNN/ffnn_params.json', 'r') as file:
        params = json.load(file)

    run_mode = params["run_mode"]
    hidden_size = params["hidden_size"]
    epochs = params["epochs"]
    test_size = params["test_size"]
    model_file_path = params["model_file_path"]
    embedding_type = params["embedding_type"]
    learning_rate = params["learning_rate"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    tokenized_lyrics = pp.data["text"]
    emotion_labels = pp.data["label"]

    logger.info(f"Converting tokenized lyrics to embeddings with {embedding_type}")
    if embedding_type == "word2vec":
        embedding_model = Word2Vec(sentences=tokenized_lyrics, vector_size=100, window=5, min_count=1, workers=4)
    else:
        embedding_model = FastText(sentences=tokenized_lyrics, vector_size=100, window=5, min_count=1, workers=4)

    lyrics_embeddings = create_embedding_matrix(tokenized_lyrics, embedding_model)

    label_encoder = LabelEncoder()
    emotion_labels = label_encoder.fit_transform(emotion_labels)

    logger.info("Preparing data for training")
    lyrics_train, lyrics_test, emotion_train, emotion_test = train_test_split(lyrics_embeddings, emotion_labels, test_size=test_size)

    lyrics_train = torch.tensor(lyrics_train, dtype=torch.float32).to(device)
    lyrics_test = torch.tensor(lyrics_test, dtype=torch.float32).to(device)
    emotion_train = torch.tensor(emotion_train, dtype=torch.long).to(device)
    emotion_test = torch.tensor(emotion_test, dtype=torch.long).to(device)
    
    logger.info(f"Chosen run mode: {run_mode}")

    input_size = lyrics_train.shape[1]
    output_size = len(label_encoder.classes_)

    logger.info("Creating FFNN model.")
    model = FFNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
    ).to(
        device=device
    )

    if "train" in run_mode:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)


        logger.info("Training has been STARTED...")
        for epoch in range(epochs):
            logger.info(f"Epoch: {epoch}")

            model.train()
            optimizer.zero_grad()
            outputs = model(lyrics_train)
            loss = criterion(outputs, emotion_train)
            loss.backward()
            optimizer.step()

            logger.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

        logger.info("Training has been FINISHED.")

        logger.info("Saving trained model")
        torch.save(model.state_dict(), model_file_path)
        logger.info(f"Model saved to {model_file_path}")


    if "test" in run_mode:
        logger.info("Evaluating the model on test data")

        model.load_state_dict(torch.load(model_file_path, weights_only=True))
        logger.info(f"Model loaded from {model_file_path}")

        model.eval()
        with torch.no_grad():
            test_outputs = model(lyrics_test)
            _, predicted = torch.max(test_outputs, 1)
            correct = (predicted == emotion_test).sum().item()
            accuracy = correct / len(emotion_test) * 100

        logger.info(f"Test Accuracy: {accuracy:.2f}%")

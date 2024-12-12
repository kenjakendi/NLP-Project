import pandas
import nltk
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from logging import Logger

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

class Preprocess:
    def __init__(self, logger: Logger,  path: str, text_column: str, label_column: str):
        self.logger = logger
        self.logger.info("Loading data STARTED ...")
        self.data = self.load_data(path, text_column, label_column)
        self.logger.info("Loading data FINISHED.")

    def run(self):
        self.logger.info("Preprocessing STARTED ...")

        self.logger.debug(self.data.head(3))
        self.convert_to_lower()
        self.logger.debug(self.data.head(3))
        self.remove_non_words()
        self.logger.debug(self.data.head(3))
        self.tokenize()
        self.logger.debug(self.data.head(3))
        self.remove_stopwords()
        self.logger.debug(self.data.head(3))
        self.lemmatization()
        self.logger.debug(self.data.head(3))

        self.logger.info("Preprocessing FINISHED.")
    
    def load_data(self, path: str, text_column: str, label_column: str):
        df = pandas.read_csv(path)
        df.rename(columns={text_column: "text", label_column: "label"}, inplace=True)
        return df[["text", "label"]]
    
    def convert_to_lower(self):
        self.data["text"] = self.data["text"].map(lambda text: text.lower() if isinstance(text, str) else text)
    
    def remove_non_words(self):
        self.data["text"] = self.data["text"].replace(to_replace=r'\[.*?\]', value='', regex=True)
        self.data["text"] = self.data["text"].replace(to_replace=r'[^\w\s]', value='', regex=True)
        self.data["text"] = self.data["text"].replace(to_replace=r'\d', value='', regex=True)

    def tokenize(self):
        self.data["text"] = self.data["text"].apply(word_tokenize)

    def remove_stopwords(self):
        swords = set(stopwords.words('english'))
        self.data["text"] = self.data["text"].apply(lambda text: [word for word in text if word not in swords])

    def lemmatization(self):
        lemmatizer = WordNetLemmatizer()
        self.data["text"] = self.data["text"].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])
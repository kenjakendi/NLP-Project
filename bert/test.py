from transformers import BertTokenizer
import pandas as pd
import yaml
from transformers import DataCollatorWithPadding
import numpy as np
import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# Wczytaj konfigurację z pliku YAML
with open("bert/bert_params.yaml", "r") as file:
    config = yaml.safe_load(file)

# Tokenizer i konfiguracja
model_name = config["name"]
max_len = config["max_len"]
batch_size = config["batch_size"]
epochs = config["epochs"]
tokenizer = BertTokenizer.from_pretrained(model_name)
learning_rate = float(config["learning_rate"])

#dane
data = pd.read_csv("data/songs.csv")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_data = data.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


id2label = {0: "sad", 1: "happy", 2: "angry", 3: "relaxed"}
label2id = {"sad": 0, "happy": 1, "angry": 2, "relaxed": 3}


model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=4, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,

)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,

)

trainer.train()
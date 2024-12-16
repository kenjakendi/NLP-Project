import pandas as pd
"""
# Wczytaj dane z pliku CSV
data = pd.read_csv("data/songs.csv")
# Sprawdź, czy wymagane kolumny istnieją
if "emotion" not in data.columns or "lyrics" not in data.columns:
    raise ValueError("Plik CSV musi zawierać kolumny 'emotion' i 'lyrics'.")
# Zlicz liczbę wystąpień każdej emocji
emotion_counts = data["emotion"].value_counts()

# Wyświetl wyniki
print("Liczba wystąpień każdej emocji:")
print(emotion_counts)
# Wczytaj dane z pliku
data = pd.read_csv("data/songs.csv")
# Inicjalizacja tokenizera
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # Zamień na config["name"] w razie potrzeby
# Lista przechowująca długości tokenów
token_lengths = []
# Tokenizuj każdy tekst i oblicz długości
for text in data["lyrics"]:
    # Tokenizacja
    input_ids = tokenizer.encode(text, max_length=512, truncation=True)
    # Dodaj długość tokenów do listy
    token_lengths.append(len(input_ids))
# Oblicz średnią długość
average_length = sum(token_lengths) / len(token_lengths)
print(f"Średnia długość ztokenizowanych tekstów: {average_length:.2f}")


przed zmianami
Średnia długość ztokenizowanych tekstów: 330.64
Maksymalna długość ztokenizowanych tekstów: 1844.00
Dłuższe od 512: 289.00
happy      563
angry      483
sad        461
relaxed    449

po zmianach (ucięcie ztokenizowanych >512, niecałe 300)
Średnia długość ztokenizowanych tekstów: 257.24
Maksymalna długość ztokenizowanych tekstów: 512.00
Dłuższe od 512: 0.00
sad        444
relaxed    421
happy      406
angry      396

import pandas as pd
from transformers import BertTokenizer

def filter_long_texts(input_csv, output_csv, text_column, label_column, max_tokens=512):
    # Wczytaj dane z pliku CSV
    data = pd.read_csv(input_csv)

    # Inicjalizacja tokenizatora BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Filtruj teksty, które mają mniej niż `max_tokens` tokenów
    filtered_data = []
    for _, row in data.iterrows():
        text = row[text_column]
        label = row[label_column]

        # Tokenizowanie tekstu
        tokenized = tokenizer(text, truncation=False, return_tensors='pt')
        num_tokens = tokenized['input_ids'].shape[1]

        # Zachowaj tylko te, które mają <= `max_tokens`
        if num_tokens <= max_tokens:
            filtered_data.append({'text': text, 'label': label})

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
def convert_txt_to_csv(input_txt, output_csv):
    """Konwertuje dane z pliku tekstowego '<label> <tekst>' do pliku CSV."""
    labels = []
    texts = []

    with open(input_txt, "r", encoding="utf-8") as file:
        for line in file:
            # Rozdziel etykietę od tekstu (pierwszy element to etykieta, reszta to tekst)
            label, text = line.strip().split(' ', 1)
            labels.append(int(label))  # Konwertuj etykietę na liczbę całkowitą
            texts.append(text)

    # Tworzenie DataFrame
    data = pd.DataFrame({"label": labels, "text": texts})

    # Zapis do pliku CSV
    data.to_csv(output_csv, index=False)
    print(f"Dane zostały zapisane w pliku: {output_csv}")

# Wywołanie funkcji
convert_txt_to_csv(
    input_txt="data/emotion.txt",  # Plik wejściowy (txt)
    output_csv="data/emotion.csv"  # Plik wyjściowy (csv)
)
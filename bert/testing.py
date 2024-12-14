import pandas as pd

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

"""
przed zmianami
Średnia długość ztokenizowanych tekstów: 330.64
Maksymalna długość ztokenizowanych tekstów: 1844.00
Dłuższe od 512: 289.00
happy      563
angry      483
sad        461
relaxed    449

po zmianach (ucięcie ztokenizowanych >512)
Średnia długość ztokenizowanych tekstów: 257.24
Maksymalna długość ztokenizowanych tekstów: 512.00
Dłuższe od 512: 0.00
sad        444
relaxed    421
happy      406
angry      396
"""
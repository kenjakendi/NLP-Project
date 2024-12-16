import pandas as pd

# Wczytaj plik Excel - podaj nazwę swojego pliku Excel
excel_file = "ml_balanced.xlsx"  # Zamień na nazwę swojego pliku

# Odczytanie danych od wiersza 17 (pomijamy nagłówki do wiersza 16)
df = pd.read_excel(excel_file, header=None, skiprows=16, usecols="A:C", engine="openpyxl")

# Przypisz kolumnom odpowiednie nazwy
df.columns = ["Index", "Artist", "Title"]

# Eksportuj dane do pliku CSV
output_csv = "Moodylyrics4Q.csv"  # Nazwa wyjściowego pliku CSV
df.to_csv(output_csv, index=False)

print(f"Dane zostały zapisane do pliku: {output_csv}")
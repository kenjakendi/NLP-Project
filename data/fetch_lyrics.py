import pandas as pd
import lyricsgenius

# Dane API Genius
client_id = "YOUR_CLIENT_ID"  # Podaj swój Client ID
access_token = "7KM-P_evPmMbJzUghKG8ESr7DYItd3TbnUAS7R-tjUv1uIxYNax08Se7fRFrNyyT"  # Token dostępu (zdobądź przez stronę Genius)

# Inicjalizacja klienta Genius
genius = lyricsgenius.Genius(access_token)

# Wczytanie pliku CSV
input_csv = "Moodylyrics4Q.csv"
df = pd.read_csv(input_csv)

# Funkcja do pobierania tekstów piosenek
def fetch_lyrics(artist, title):
    try:
        song = genius.search_song(title, artist)
        return song.lyrics if song else "Lyrics not found"
    except Exception as e:
        return f"Error fetching lyrics: {e}"

# Dodanie kolumny z tekstami piosenek
df["Lyrics"] = df.apply(lambda row: fetch_lyrics(row["Artist"], row["Title"]), axis=1)

# Zapisanie wyników do nowego pliku CSV
output_csv = "Moodylyrics_with_lyrics.csv"
df.to_csv(output_csv, index=False)

print(f"Dane zostały zapisane do pliku: {output_csv}")

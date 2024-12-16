import pandas as pd
import lyricsgenius
import os


# Wczytanie pliku CSV
input_csv = os.path.join('data2', 'Moodylyrics4Q.csv')
data = pd.read_csv(input_csv)

# Inicjalizacja API Lyrics Genius
api_key = "adamW"  # Zamień na swój klucz API
genius = lyricsgenius.Genius(api_key)
# Dostosowanie nagłówków użytkownika
# genius.headers.update({"User-Agent": "Mozilla/5.0 (compatible; LyricsFetcher/1.0)"})

def fetch_lyrics(artist, title):
    """Funkcja do pobierania tekstów piosenek."""
    try:
        song = genius.search_song(title, artist)
        if song:
            return song.lyrics
        else:
            return "Lyrics not found"
    except Exception as e:
        print(f"Error fetching lyrics for {artist} - {title}: {e}")
        return "Error fetching lyrics"

# Dodawanie kolumny z tekstami piosenek
lyrics = []
for index, row in data.iterrows():
    artist = row["Artist"]
    title = row["Title"]
    print(f"Fetching lyrics for: {artist} - {title}")
    lyrics.append(fetch_lyrics(artist, title))

data["Lyrics"] = lyrics

# Zapisanie wyników do nowego pliku CSV
output_file = "Moodylyrics_with_lyrics.csv"
data.to_csv(output_file, index=False)

print(f"Zaktualizowany plik zapisano jako {output_file}")

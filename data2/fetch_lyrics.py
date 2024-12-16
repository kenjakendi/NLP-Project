import os
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup


# Funkcja do przetwarzania nazw artysty i piosenki na format URL AZLyrics
def format_artist_and_song(artist, song):
    artist = artist.replace(" ", "").lower()
    song = song.replace(" ", "").lower()
    artist = "".join(letter for letter in artist if letter.isalnum())
    song = "".join(letter for letter in song if letter.isalnum())
    return artist, song


# Funkcja do pobierania tekstu piosenki z AZLyrics
def fetch_lyrics(artist, song):
    url = f"https://www.azlyrics.com/lyrics/{artist}/{song}.html"
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            return ""
        
        soup = BeautifulSoup(response.content, 'html.parser')
        lyrics_div = soup.select_one(".ringtone ~ div")
        if not lyrics_div:
            return ""
        
        lyrics = lyrics_div.get_text(strip=True, separator=" ")
        return lyrics
    except Exception as e:
        print(f"Błąd podczas pobierania tekstu: {e}")
        return ""


if __name__ == "__main__":
    # Ścieżka do pliku CSV
    input_csv = os.path.join('data2', 'Moodylyrics4Q.csv')
    output_csv = os.path.join('data2', 'Moodylyrics4Q_with_lyrics.csv')

    # Wczytaj dane z pliku wejściowego
    data = pd.read_csv(input_csv)

    # Jeśli plik wynikowy istnieje, wczytaj istniejące dane
    if os.path.exists(output_csv):
        existing_data = pd.read_csv(output_csv)
        # Zestaw kluczy (Artist, Title) dla już zapisanych piosenek
        processed_songs = set(zip(existing_data['Artist'], existing_data['Title']))
    else:
        # Jeśli plik wynikowy nie istnieje, utwórz nagłówki
        pd.DataFrame(columns=['Index', 'Artist', 'Title', 'Tekst']).to_csv(output_csv, index=False, encoding='utf-8')
        processed_songs = set()

    # Pobieranie tekstów piosenek
    for index, row in data.iterrows():
        song_key = (row['Artist'], row['Title'])
        if song_key in processed_songs:
            # Jeśli tekst dla piosenki już istnieje, pomiń ją
            print(f"Pomiń: {row['Artist']} - {row['Title']} ({index + 1}/{len(data)})")
            continue

        # Pobierz tekst piosenki
        artist, song = format_artist_and_song(row['Artist'], row['Title'])
        print(f"Pobieram tekst dla: {row['Artist']} - {row['Title']} ({index + 1}/{len(data)})")
        lyrics = fetch_lyrics(artist, song)
        lyrics = lyrics if lyrics else "Lyrics not found"

        # Zapisz wiersz na bieżąco do pliku CSV
        pd.DataFrame([{
            'Index': row['Index'],
            'Artist': row['Artist'],
            'Title': row['Title'],
            'Tekst': lyrics
        }]).to_csv(output_csv, mode='a', header=False, index=False, encoding='utf-8')

        # Dodaj piosenkę do zestawu przetworzonych
        processed_songs.add(song_key)

        # Opóźnienie między zapytaniami
        time.sleep(5)

    print(f"Plik wynikowy zapisano jako: {output_csv}")

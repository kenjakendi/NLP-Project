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

    # Wczytaj dane z pliku
    data = pd.read_csv(input_csv)

    # Jeśli plik wynikowy nie istnieje, zapisz nagłówki
    if not os.path.exists(output_csv):
        pd.DataFrame(columns=['Index', 'Artist', 'Title', 'Tekst']).to_csv(output_csv, index=False, encoding='utf-8')

    # Pobieranie tekstów piosenek
    for index, row in data.iterrows():
        artist, song = format_artist_and_song(row['Artist'], row['Title'])
        print(f"Pobieram tekst dla: {row['Artist']} - {row['Title']} ({index + 1}/{len(data)})")
        lyrics = fetch_lyrics(artist, song)
        lyrics = lyrics if lyrics else "Lyrics not found"

        # Zapisz wiersz na bieżąco do pliku CSV, zachowując kolumnę Index
        pd.DataFrame([{
            'Index': row['Index'],
            'Artist': row['Artist'],
            'Title': row['Title'],
            'Tekst': lyrics
        }]).to_csv(output_csv, mode='a', header=False, index=False, encoding='utf-8')

        time.sleep(5)  # Opóźnienie między zapytaniami

    print(f"Plik wynikowy zapisano jako: {output_csv}")

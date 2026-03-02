🚢 Titanic Survival Predictor (FastAPI)
Interaktywna aplikacja webowa wykorzystująca Machine Learning do przewidywania szans na przeżycie pasażerów statku Titanic. Projekt łączy nowoczesny backend w FastAPI z przejrzystym interfejsem użytkownika.

✨ Funkcje
Predykcja w czasie rzeczywistym: Wykorzystuje model Scikit-learn do analizy danych pasażera.

Asynchroniczny Interface: Dzięki technologii AJAX (JavaScript Fetch API) wyniki pojawiają się bez przeładowania strony.

Responsywny Design: Nowoczesny wygląd z animacjami, dostosowany do urządzeń mobilnych i komputerów.

Gotowość do wdrożenia: Skonfigurowany pod kątem hostingu na platformach takich jak Render.

🛠️ Technologie
Backend: Python 3.x, FastAPI, Uvicorn.

Machine Learning: Pandas, Scikit-learn, Pickle.

Frontend: HTML5, CSS3 (Custom Styles), JavaScript (Vanilla JS).

Deployment: Gunicorn.

📂 Struktura Projektu
Plaintext
titanic-fastapi/
├── data/               # Zbiory danych (train/test)
├── inference/          # Główny kod aplikacji
│   ├── templates/      # Pliki HTML (index.html)
│   └── app.py          # Serwer FastAPI i logika predykcji
├── models/             # Trenowane modele (pliki .pkl)
├── requirements.txt    # Lista bibliotek do zainstalowania
└── README.md           # Dokumentacja projektu
🚀 Jak uruchomić projekt lokalnie?
Sklonuj repozytorium:

Bash
git clone https://github.com/volodiafedishin-code/titanic-fastapi.git
cd titanic-fastapi
Stwórz i aktywuj wirtualne środowisko:

Bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
Zainstaluj wymagane biblioteki:

Bash
pip install -r requirements.txt
Uruchom serwer:

Bash
uvicorn inference.app:app --reload
Aplikacja będzie dostępna pod adresem: http://127.0.0.1:8000

📊 Jak działa predykcja?
Aplikacja pobiera od użytkownika dane takie jak:

Klasa biletowa (Pclass)

Wiek (Age)

Opłata za bilet (Fare)

Płeć (Sex) — w trakcie wdrażania

Dane te są przesyłane metodą POST do endpointu /predict, gdzie model Machine Learning dokonuje klasyfikacji i zwraca wynik ("Survived" lub "Not Survived").

Projekt stworzony w celach edukacyjnych jako demonstracja integracji ML z aplikacją webową.

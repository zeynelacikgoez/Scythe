# src/scythe/data/ingestion.py

from pathlib import Path
import pandas as pd
import requests

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Lädt Rohdaten aus einer CSV-Datei und gibt sie als Pandas DataFrame zurück.
    
    Parameter:
    ----------
    file_path : str
        Der vollständige Pfad zur CSV-Datei (z. B. "./data/raw_data.csv").
    
    Rückgabe:
    ---------
    pd.DataFrame : Der geladene DataFrame.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"[ingestion] CSV-Datei nicht gefunden: {path}")

    print(f"[ingestion] Lade Daten aus {path} ...")
    df = pd.read_csv(path, sep=",", header=0, encoding="utf-8")
    print(f"[ingestion] Gelesen: {len(df)} Zeilen, {len(df.columns)} Spalten.")
    return df


def load_multiple_csv(data_dir: str, file_list: list = None) -> pd.DataFrame:
    """
    Lädt mehrere CSV-Dateien aus einem Verzeichnis und fügt sie zu einem DataFrame zusammen.
    
    Parameter:
    ----------
    data_dir : str
        Der Pfad zum Verzeichnis, in dem die CSV-Dateien liegen (z. B. "./data").
    file_list : list of str, optional
        Liste der Dateinamen. Falls None, werden alle *.csv-Dateien im Verzeichnis geladen.
    
    Rückgabe:
    ---------
    pd.DataFrame : Zusammengeführter DataFrame aller geladenen CSV-Dateien.
    """
    base_path = Path(data_dir)
    if not base_path.is_dir():
        raise NotADirectoryError(f"[ingestion] Verzeichnis nicht gefunden: {base_path}")

    if file_list is None:
        file_list = [f.name for f in base_path.glob("*.csv")]

    if not file_list:
        raise ValueError(f"[ingestion] Keine CSV-Dateien gefunden in {base_path}")

    df_list = []
    for fname in file_list:
        csv_path = base_path / fname
        if csv_path.is_file():
            print(f"[ingestion] Lade {csv_path} ...")
            df_part = pd.read_csv(csv_path, sep=",", header=0, encoding="utf-8")
            df_list.append(df_part)
        else:
            print(f"[ingestion] Datei nicht gefunden: {csv_path}")

    if not df_list:
        raise ValueError("[ingestion] Keine gültigen CSV-Dateien geladen.")

    df_merged = pd.concat(df_list, ignore_index=True)
    print(f"[ingestion] Zusammengeführt: {len(df_merged)} Zeilen, {len(df_merged.columns)} Spalten.")
    return df_merged


def load_data_from_api(api_url: str, params: dict = None) -> pd.DataFrame:
    """
    Lädt Daten über eine API-Anfrage und gibt sie als DataFrame zurück.
    
    Parameter:
    ----------
    api_url : str
        Der URL/Endpoint der API.
    params : dict, optional
        Parameter, die an die API übergeben werden.
    
    Rückgabe:
    ---------
    pd.DataFrame : DataFrame mit den geladenen Daten.
    """
    if params is None:
        params = {}

    print(f"[ingestion] API-Anfrage an {api_url} mit params={params}")
    response = requests.get(api_url, params=params)
    if response.status_code != 200:
        raise ConnectionError(f"Fehler beim API-Aufruf, Status={response.status_code}")

    data_json = response.json()
    df = pd.DataFrame(data_json)
    print(f"[ingestion] Geladene API-Datensätze: {len(df)}")
    return df


if __name__ == "__main__":
    # Demo: Verwende einen relativen Pfad zum Datenordner
    demo_data_dir = "./data"
    
    # Test 1: load_raw_data
    try:
        # Beispiel: "./data/raw_data.csv"
        df_demo = load_raw_data(str(Path(demo_data_dir) / "raw_data.csv"))
        print("[ingestion] Erster Einblick in raw_data.csv:")
        print(df_demo.head())
    except Exception as e:
        print(e)
    
    # Test 2: load_multiple_csv
    try:
        df_multi = load_multiple_csv(demo_data_dir)
        print("[ingestion] Head der zusammengeführten CSV-Dateien:")
        print(df_multi.head())
    except Exception as e:
        print(e)
    
    # Test 3: load_data_from_api (Beispiel, API muss existieren)
    try:
        df_api = load_data_from_api("https://api.example.com/data")
        print("[ingestion] Head der API-Daten:")
        print(df_api.head())
    except Exception as e:
        print(e)

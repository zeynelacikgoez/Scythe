"""
Das data-Paket:
------------------------
Dieses Paket stellt Funktionen zum Laden, Bereinigen und Vorverarbeiten von Daten bereit.
Die Implementierungen in den Untermodulen (z. B. in ingestion.py, preprocessing.py und utils.py)
nutzen dabei robuste Pfadmethoden (etwa mit pathlib) für ein plattformunabhängiges Dateipfadhandling.

Typische Module:
  - ingestion.py    : Einlesen von Daten (z. B. CSV-Dateien, APIs) mit robustem Pfadhandling
  - preprocessing.py: Bereinigung, Feature-Engineering und Umwandlung in Tensors
  - utils.py        : Diverse Hilfsfunktionen (z. B. Datumsparsing, kategoriale Umwandlungen)

Beispielhafter Gebrauch:
    from scythe.data import load_raw_data, preprocess_data

    df = load_raw_data("data/raw_data.csv")
    df_clean = preprocess_data(df)
"""

from .ingestion import load_raw_data
from .preprocessing import preprocess_data, create_embedding_tensor
from .utils import parse_date_column, convert_categorical

__all__ = [
    "load_raw_data",
    "preprocess_data",
    "create_embedding_tensor",
    "parse_date_column",
    "convert_categorical"
]

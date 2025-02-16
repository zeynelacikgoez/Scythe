```markdown
# data_pipeline/

Das Verzeichnis **data_pipeline/** enthält alle Module und Skripte, die sich um das **Einlesen** und **Vorverarbeiten** (Preprocessing) von Rohdaten kümmern. Hier werden Datenquellen (z. B. CSV-Dateien, APIs) in ein standardisiertes Format konvertiert (z. B. `pandas.DataFrame`), fehlende Werte bereinigt, Features erstellt, und bei Bedarf in embeddings oder Tensors für Modellarchitekturen umgewandelt.

---

## Ordnerstruktur

```
data_pipeline/
├── __init__.py
├── data_ingestion.py
├── data_preprocessing.py
├── data_utils.py
└── readme.md
```

### 1. `__init__.py`
- Kennzeichnet `data_pipeline/` als Python-Paket.  
- (Optional) **Re-Exports** zentraler Funktionen, sodass sie via `from data_pipeline import ...` importiert werden können.  
- Typische Import-Statements:
  ```python
  from .data_ingestion import load_raw_data, load_multiple_csv
  from .data_preprocessing import preprocess_data, create_embedding_tensor
  # ...
  __all__ = [...]
  ```

### 2. `data_ingestion.py`
- Enthält **Funktionen zum Laden** von Rohdaten:
  - `load_raw_data(data_dir, file_name)`: CSV-Datei einlesen und als `pandas.DataFrame` zurückgeben.  
  - `load_multiple_csv(data_dir, file_list)`: Mehrere CSV-Dateien zusammenführen.  
  - `load_data_from_api(api_url, ...)`: Beispielhafte API-Anbindung (JSON → DataFrame).  
- Hier kannst du **jeden** Datenquellentyp (z. B. SQL-Datenbanken, Excel, Parquet) integrieren.

### 3. `data_preprocessing.py`
- Sammelt **Preprocessing-Schritte**, die mit dem eingelesenen DataFrame durchgeführt werden:
  - `preprocess_data(df)`: z. B. NA-Handling, Feature-Engineering (Log-Spalten, Categorical Encoding) etc.  
  - `create_embedding_tensor(df, feature_cols)`: konvertiert ausgewählte Spalten in einen **PyTorch Tensor**.  
- Eignet sich, um die Daten in den Zustand zu bringen, den **spätere ML- oder KI-Modelle** benötigen.

### 4. `data_utils.py`
- **Hilfsfunktionen** für Parsing, Konvertierungen, Outlier-Erkennung usw.:
  - `parse_date_column(...)`: Datumsstrings zu `datetime64` umwandeln.  
  - `convert_categorical(...)`: Spalten in `pd.Categorical` + ID-Säule.  
  - `fill_missing_values(...)`: Fehldaten z. B. per Mean/Median/Mode füllen.  
  - `detect_outliers_iqr(...)`: Ausreißer per IQR-Regel identifizieren.  
- So bleiben wiederkehrende Code-Patterns **zentral** und `data_ingestion.py`/`data_preprocessing.py` werden schlanker.

### 5. `readme.md` (diese Datei)
- Beschreibt den **Zweck** und die Struktur von `data_pipeline/`.
- Erklärt, wie man die Module verwendet.

---

## Beispielhafter Workflow

1. **Daten laden**  
   ```python
   from data_pipeline import load_raw_data

   df_raw = load_raw_data(data_dir="./data", file_name="train.csv")
   print(df_raw.head())
   ```

2. **Preprocessing**  
   ```python
   from data_pipeline import preprocess_data

   df_clean = preprocess_data(df_raw)
   print(df_clean.head())
   ```

3. **Embeddings / Tensor**  
   ```python
   from data_pipeline import create_embedding_tensor

   input_tensor = create_embedding_tensor(df_clean, feature_cols=["production_log", "consumption_log"])
   print("Input Tensor shape:", input_tensor.shape)
   ```

4. **Utils**  
   ```python
   from data_pipeline.data_utils import parse_date_column

   df_with_date = parse_date_column(df_clean, "date_str", date_format="%Y-%m-%d")
   ```

---

## Anpassung & Erweiterung

- **Pfade & Formate**: Wenn du andere Formate (z. B. Parquet, Feather) oder Datenquellen (Datenbankverbindungen, Cloud-APIs) nutzen möchtest, füge entsprechende **Loader** in `data_ingestion.py` hinzu.  
- **Preprocessing-Schritte**: An deine *Use-Cases* anpassen (z. B. pro Branche/Task anderes Feature-Engineering).  
- **Utils**: Häufig lohnt es sich, hier generische, wiederverwendbare Tools zu sammeln (z. B. `compute_statistics(...)`, `split_train_val(df)`, usw.).

---

## Fazit

Das **`data_pipeline/`**-Paket bietet dir eine **klare** Unterteilung zwischen **Daten-Einlesen** (`data_ingestion.py`), **Vorverarbeitung** (`data_preprocessing.py`) und möglichen **Helper-Funktionen** (`data_utils.py`). So wird deine Projektstruktur **übersichtlich**, und sämtliche datenbezogenen Schritte sind **zentral** abgelegt und leicht erweiterbar.
```
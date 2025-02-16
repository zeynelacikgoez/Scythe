# preprocessing.py

import pandas as pd
import numpy as np
import torch
from typing import Optional, List


def preprocess_data(
    df: pd.DataFrame,
    drop_columns_with_too_many_nans: Optional[List[str]] = None,
    required_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Führt eine erweiterte Datenbereinigung und Feature-Engineering durch.
    
    Beispiele:
      - Entfernen von Zeilen mit fehlenden Werten in Kerndaten-Spalten (z. B. production, consumption)
      - Entfernen von Zeilen, bei denen bestimmte (zusätzliche) Spalten zu viele NaNs haben
      - Umwandlung der Spalte 'category' in numerische Codes (Erzeugung von 'category_id')
      - Log-Transformationen für numerische Spalten (z. B. production, consumption, demand_target)
      - Datumsparsing und Ableitung eines Wochentags (weekday)

    Parameter
    ----------
    df : pd.DataFrame
        Der originale DataFrame (z.B. aus data_ingestion.py).
    drop_columns_with_too_many_nans : Optional[List[str]]
        Liste von Spalten, bei denen Zeilen mit NaN-Werten entfernt werden.
    required_columns : Optional[List[str]]
        Liste von Spalten, die zwingend vorhanden sein müssen. Fehlen sie, wird eine Exception geworfen.

    Returns
    -------
    pd.DataFrame
        Der bereinigte und angereicherte DataFrame.
    """
    # 0) Pflichtspalten prüfen
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Fehlende Pflichtspalten im DataFrame: {missing_cols}")

    # 1) Entferne Zeilen mit fehlenden Kerndaten (z. B. production, consumption)
    core_columns = [col for col in ["production", "consumption"] if col in df.columns]
    if core_columns:
        df = df.dropna(subset=core_columns)

    # 2) Entferne Zeilen, in denen Spalten mit zu vielen NaNs fehlen
    if drop_columns_with_too_many_nans:
        df = df.dropna(subset=drop_columns_with_too_many_nans)

    # 3) Konvertiere 'category'-Spalte in numerische Codes, falls vorhanden
    if 'category' in df.columns:
        df['category_id'] = df['category'].astype('category').cat.codes

    # 4) Log-Transformationen
    if 'production' in df.columns:
        df['production_log'] = np.log1p(df['production'].clip(lower=0))
    if 'consumption' in df.columns:
        df['consumption_log'] = np.log1p(df['consumption'].clip(lower=0))
    if 'demand_target' in df.columns:
        # Hier wird nur einmal +1 addiert, um negative Werte zu vermeiden
        df['demand_target_log'] = np.log1p(df['demand_target'].clip(lower=0) + 1)

    # 5) Datumsparsing und Ableitung des Wochentags
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['weekday'] = df['date'].dt.dayofweek

    # 6) Zum Schluss: Entferne alle verbleibenden Zeilen mit NaNs und setze den Index zurück
    df = df.dropna().reset_index(drop=True)

    return df


def create_embedding_tensor(
    df: pd.DataFrame, 
    feature_cols: Optional[List[str]] = None
) -> torch.Tensor:
    """
    Extrahiert ausgewählte Spalten aus dem DataFrame und packt sie in ein Torch Tensor.
    Dieses Tensor kann als Eingabe-Feature (Embedding) für Modelle wie einen Transformer dienen.

    Parameter
    ----------
    df : pd.DataFrame
        Der bereinigte DataFrame.
    feature_cols : Optional[List[str]]
        Liste der Spalten, die als Features verwendet werden sollen.
        Falls None werden Standardspalten genutzt: 'production_log', 'consumption_log',
        'category_id' und 'demand_target_log' (sofern vorhanden). Sollte keines dieser Felder existieren,
        werden alle numerischen Spalten verwendet.

    Returns
    -------
    torch.Tensor
        Ein Tensor der Form (N, D), wobei N die Anzahl der Zeilen und D die Anzahl der Feature-Spalten ist.
    """
    if feature_cols is None:
        possible_cols = []
        for col in ['production_log', 'consumption_log', 'category_id', 'demand_target_log']:
            if col in df.columns:
                possible_cols.append(col)
        if not possible_cols:
            possible_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = possible_cols

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Die folgenden Spalten fehlen im DataFrame: {missing}")

    features = df[feature_cols].values.astype(np.float32)
    tensor = torch.tensor(features, dtype=torch.float32)

    print(f"[data_preprocessing] Created embedding tensor with shape={tensor.shape} (features: {feature_cols})")
    return tensor


if __name__ == "__main__":
    # Demo: Erstelle einen Beispiel-DataFrame und führe das Preprocessing durch
    demo_data = {
        "production": [10, 20, 30, None, 50],
        "consumption": [5, None, 25, 30, 45],
        "category": ["A", "B", "B", "A", "C"],
        "demand_target": [100, 150, 120, None, 200],
        "date": ["2023-01-01", "2023-01-02", "2023-01-03", None, "2023-01-05"]
    }
    df_demo = pd.DataFrame(demo_data)
    print("[data_preprocessing] Original DataFrame:")
    print(df_demo)

    df_clean = preprocess_data(
        df_demo, 
        required_columns=["production", "consumption"]
    )
    print("\n[data_preprocessing] Cleaned DataFrame:")
    print(df_clean)

    emb_tensor = create_embedding_tensor(df_clean)
    print("\n[data_preprocessing] Embedding Tensor:")
    print(emb_tensor)

# tests/test_preprocess_data.py

import pytest
import pandas as pd
import numpy as np
import torch

from data_pipeline.data_preprocessing import preprocess_data, create_embedding_tensor

def test_preprocess_data_basic():
    """
    Testet einen Basis-Flow mit vorhandenen 'production', 'consumption', 'category' Spalten.
    Erwartet:
      - NaN wird bei production/consumption gedroppt
      - 'production_log' und 'consumption_log' Spalten entstehen
      - 'category_id' Spalte entsteht
      - Resultierende Zeilenanzahl (nach dropna) korrekt
    """
    df = pd.DataFrame({
        "production": [10, 20, None, 40],
        "consumption": [5, None, 15, 20],
        "category": ["A", "B", "A", "C"],
        "other_col": [100, 200, 300, 400]
    })

    # Preprocessing
    df_clean = preprocess_data(df)

    # Wir erwarten, dass Zeilen gedroppt werden,
    # wo production oder consumption == NaN
    # => Zeile 1 (index=1 => consumption=NaN) und Zeile 2 (production=NaN) droppen
    assert len(df_clean) == 2, f"Erwartet 2 Zeilen nach Drop, bekam {len(df_clean)}"

    # Check Spalten
    expected_cols = {"production", "consumption", "production_log", "consumption_log", "category", "category_id", "other_col"}
    assert expected_cols.issubset(df_clean.columns), "Nicht alle erwarteten Spalten vorhanden."

    # production_log + consumption_log check:
    for idx, row in df_clean.iterrows():
        prod = row["production"]
        cons = row["consumption"]
        assert row["production_log"] == np.log1p(prod + 1), f"production_log inkorrekt bei idx={idx}"
        assert row["consumption_log"] == np.log1p(cons + 1), f"consumption_log inkorrekt bei idx={idx}"

    # category_id check
    assert "category_id" in df_clean.columns, "category_id Spalte nicht generiert."
    # Werte
    cat_ids = df_clean["category_id"].unique()
    # Da wir 'category' = {A,B,C} haben, sollte mind. eine ID pro Wert existieren
    # Die Zuordnung ist von pandas.cat.codes abhängig, wir checken nur Existenz
    assert len(cat_ids) <= 3, "Zu viele category_ids erzeugt."


def test_preprocess_data_missing_cols():
    """
    Testet, ob die Funktion preprocess_data() robust mit fehlenden Spalten umgeht
    (production oder consumption fehlt).
    Erwartetes Verhalten:
      - Wenn 'production' oder 'consumption' fehlen, kein Drop => df bleibt unverändert
      - 'production_log' bzw. 'consumption_log' wird nicht erzeugt
      - Kein Fehler
    """
    df = pd.DataFrame({
        "some_col": [1, 2, 3],
        "category": ["X", "Y", "Z"]
    })
    df_clean = preprocess_data(df)

    # Es sollte kein Fehler passieren, df wird nicht gedroppt, da production/consumption nicht existieren
    assert len(df_clean) == 3, "Sollte keine Zeile droppen, wenn production/consumption fehlen."
    assert "production_log" not in df_clean.columns, "production_log sollte nicht existieren."
    assert "consumption_log" not in df_clean.columns, "consumption_log sollte nicht existieren."


def test_create_embedding_tensor_default():
    """
    Testet create_embedding_tensor() mit default-Auswahl
    (nimmt z. B. production_log, consumption_log, category_id, falls vorhanden).
    """
    df = pd.DataFrame({
        "production": [10, 20],
        "consumption": [5, 15],
        "production_log": [2.3, 3.0],
        "consumption_log": [1.7, 2.7],
        "category_id": [0, 1]
    })

    tensor = create_embedding_tensor(df)
    # Default: probable feature_cols = ["production_log", "consumption_log", "category_id"]
    assert tensor.shape == (2, 3), f"Erwartet shape (2,3), bekam {tensor.shape}"
    # Check Datentyp float32
    assert tensor.dtype == torch.float32, "Datentyp sollte float32 sein."


def test_create_embedding_tensor_custom_cols():
    """
    Testet create_embedding_tensor() mit benutzerdefinierten feature_cols.
    """
    df = pd.DataFrame({
        "prod_log": [0.0, 1.0, 2.0],
        "cons_log": [0.5, 1.5, 2.5],
        "extra_col": [10, 20, 30]
    })

    feature_cols = ["prod_log", "cons_log"]
    tensor = create_embedding_tensor(df, feature_cols=feature_cols)
    assert tensor.shape == (3, 2), f"Erwartet (3,2), bekam {tensor.shape}"
    # Prüfen, ob Werte korrekt übernommen werden
    np.testing.assert_array_almost_equal(
        tensor.numpy()[:, 0],
        df["prod_log"].values.astype(np.float32)
    )


def test_preprocess_data_dropna_after_log():
    """
    Prüfen, ob nach der Log-Transformation evtl. weitere NaNs entstehen
    und ob die dropna (Step 4) greift.
    Hier: production= -5 => log1p( -5 + 1 ) => log( -4 ) => NaN in production_log
    => Zeile soll gedroppt werden.
    """
    df = pd.DataFrame({
        "production": [10, -5, 20],
        "consumption": [5, 10, 15]
    })
    df_clean = preprocess_data(df)
    # 2 Zeilen sollten übrig bleiben, da die Zeile mit production=-5 => NaN in production_log => drop
    assert len(df_clean) == 2, f"Erwartet 2 Zeilen übrig, bekam {len(df_clean)}"


def test_preprocess_data_reset_index():
    """
    Testet, ob der DataFrame am Ende ein reset_index(drop=True) erfährt
    (aktuell in code_preprocessing vorhanden).
    """
    df = pd.DataFrame({
        "production": [1, None, 3],
        "consumption": [2, 4, None]
    })
    # => Zeile 1 und 2 haben Missing => droppen => nur Index=0,1,2 => am Ende 1 Zeile?
    df_clean = preprocess_data(df)
    # => 1 Zeile bleibt => index=0
    assert len(df_clean) == 1
    assert df_clean.index[0] == 0, "Erwartet reset_index=0 bei erster Zeile."


if __name__ == "__main__":
    # Falls man dieses Skript direkt ausführen will:
    import sys
    import pytest
    sys.exit(pytest.main([__file__]))

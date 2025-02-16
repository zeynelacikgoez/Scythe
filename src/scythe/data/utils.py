# src/scythe/data/utils.py

import pandas as pd
import numpy as np
from typing import List, Optional


def parse_date_column(df: pd.DataFrame, column_name: str, date_format: Optional[str] = None) -> pd.DataFrame:
    """
    Parst die angegebene Spalte des DataFrames als Datum/Zeit.
    
    Falls `date_format` angegeben ist, wird dieses explizit verwendet; ansonsten erfolgt eine automatische
    Format-Erkennung. Die Funktion wandelt die Spalte in den Datentyp datetime64 um.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame, der die zu konvertierende Spalte enthält.
    column_name : str
        Name der Spalte, die Datums-/Zeitstrings enthält.
    date_format : Optional[str]
        Format-String (z. B. "%Y-%m-%d"). Standardmäßig None.

    Returns
    -------
    pd.DataFrame
        DataFrame, in dem die angegebene Spalte in datetime64 konvertiert wurde.

    Raises
    ------
    ValueError
        Falls die angegebene Spalte nicht im DataFrame existiert.
    """
    if column_name not in df.columns:
        raise ValueError(f"[parse_date_column] Spalte '{column_name}' existiert nicht im DataFrame.")

    if date_format:
        df[column_name] = pd.to_datetime(df[column_name], format=date_format, errors='coerce')
    else:
        df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
    return df


def convert_categorical(df: pd.DataFrame, column_name: str, categories: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Konvertiert eine Spalte in den Datentyp 'category' und erstellt zusätzlich eine numerische
    Repräsentation (mit dem Suffix "_cat_id").

    Parameters
    ----------
    df : pd.DataFrame
        Eingabe-DataFrame.
    column_name : str
        Name der Spalte, die konvertiert werden soll.
    categories : Optional[List[str]]
        Falls angegeben, wird die Reihenfolge der Kategorien entsprechend festgelegt.

    Returns
    -------
    pd.DataFrame
        DataFrame mit der konvertierten Spalte und einer zusätzlichen Spalte 
        "<column_name>_cat_id" mit den numerischen Codes.

    Raises
    ------
    ValueError
        Falls die angegebene Spalte nicht im DataFrame vorhanden ist.
    """
    if column_name not in df.columns:
        raise ValueError(f"[convert_categorical] Spalte '{column_name}' existiert nicht im DataFrame.")

    if categories:
        cat_type = pd.CategoricalDtype(categories=categories, ordered=False)
        df[column_name] = df[column_name].astype(cat_type)
    else:
        df[column_name] = df[column_name].astype('category')

    new_col = column_name + "_cat_id"
    df[new_col] = df[column_name].cat.codes
    return df


def fill_missing_values(df: pd.DataFrame, column_name: str, method: str = "mean") -> pd.DataFrame:
    """
    Füllt fehlende Werte in der angegebenen Spalte mit dem Mittelwert, Median oder Modus.

    Parameters
    ----------
    df : pd.DataFrame
        Eingabe-DataFrame.
    column_name : str
        Spalte, in der fehlende Werte ersetzt werden sollen.
    method : str
        Füllmethode: "mean", "median" oder "mode".

    Returns
    -------
    pd.DataFrame
        DataFrame mit den gefüllten Werten in der angegebenen Spalte.

    Raises
    ------
    ValueError
        Falls die Spalte nicht existiert oder eine unbekannte Methode angegeben wird.
    """
    if column_name not in df.columns:
        raise ValueError(f"[fill_missing_values] Spalte '{column_name}' existiert nicht im DataFrame.")

    if method == "mean":
        fill_val = df[column_name].mean()
    elif method == "median":
        fill_val = df[column_name].median()
    elif method == "mode":
        mode_vals = df[column_name].mode()
        fill_val = mode_vals.iloc[0] if not mode_vals.empty else np.nan
    else:
        raise ValueError(f"[fill_missing_values] Unbekannte Füllmethode: {method}")

    df[column_name] = df[column_name].fillna(fill_val)
    return df


def detect_outliers_iqr(df: pd.DataFrame, column_name: str) -> pd.Series:
    """
    Identifiziert Ausreißer in einer numerischen Spalte anhand der IQR-Methode.
    
    Die Funktion berechnet den Interquartilsabstand (IQR) und markiert Zeilen als Ausreißer,
    wenn der Wert unter (Q1 - 1.5 * IQR) oder über (Q3 + 1.5 * IQR) liegt.

    Parameters
    ----------
    df : pd.DataFrame
        Eingabe-DataFrame.
    column_name : str
        Spalte, in der Ausreißer ermittelt werden sollen.

    Returns
    -------
    pd.Series
        Boolean-Serien, wobei True bedeutet, dass der jeweilige Zeilenwert ein Ausreißer ist.

    Raises
    ------
    ValueError
        Falls die angegebene Spalte nicht vorhanden ist.
    """
    if column_name not in df.columns:
        raise ValueError(f"[detect_outliers_iqr] Spalte '{column_name}' existiert nicht im DataFrame.")

    col_data = df[column_name].dropna()
    q1 = col_data.quantile(0.25)
    q3 = col_data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return (df[column_name] < lower_bound) | (df[column_name] > upper_bound)


if __name__ == "__main__":
    # Demo: Erzeugen eines kleinen DataFrames und Anwenden der Hilfsfunktionen
    demo_data = {
        "date_str": ["2022-01-01", "2022/02/10", "2022-03-15", None],
        "cat_col": ["A", "B", "C", "D"],
        "num_col": [10, 20, 30, 40]
    }
    df_demo = pd.DataFrame(demo_data)
    print("[data/utils] Original DataFrame:")
    print(df_demo)

    # Parst die Datumsspalte
    df_demo = parse_date_column(df_demo, "date_str")
    print("\n[data/utils] Nach parse_date_column:")
    print(df_demo)

    # Konvertiert die kategorische Spalte
    df_demo = convert_categorical(df_demo, "cat_col", categories=["A", "B", "C", "D"])
    print("\n[data/utils] Nach convert_categorical:")
    print(df_demo)

    # Füllt fehlende Werte in 'num_col' (hier: Methode 'median')
    df_demo = fill_missing_values(df_demo, "num_col", method="median")
    print("\n[data/utils] Nach fill_missing_values:")
    print(df_demo)

    # Identifiziert Ausreißer in 'num_col'
    outliers = detect_outliers_iqr(df_demo, "num_col")
    print("\n[data/utils] Outlier in 'num_col':")
    print(outliers)

from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset

class HyperbolicDataset(Dataset):
    """
    HyperbolicDataset:
    ------------------
    Ein Dataset zum Laden von Daten aus einer CSV-Datei.
    
    Die Pfadangabe zur CSV-Datei wird relativ zu einem Basisverzeichnis interpretiert,
    was das Projekt portabler und flexibler in Bezug auf unterschiedliche Verzeichnisstrukturen macht.
    
    Parameter:
    -----------
    csv_path : str
        Relativer oder absoluter Pfad zur CSV-Datei.
    base_dir : str, optional
        Basisverzeichnis, relativ zu dem `csv_path` interpretiert wird.
        Standardmäßig wird das aktuelle Arbeitsverzeichnis verwendet.
    feature_cols : list of str, optional
        Liste der Spalten, die als Input-Features genutzt werden.
        Standard: ["feat1", "feat2", "feat3"].
    target_cols : list of str, optional
        Liste der Spalten, die als Zielwerte genutzt werden.
        Standard: ["t1", "t2", "t3"].
    transform : callable, optional
        Eine Funktion, die auf das Tupel (features, targets) angewendet wird.
    """
    def __init__(self, csv_path: str, base_dir: str = ".",
                 feature_cols=None, target_cols=None, transform=None):
        # Erstelle einen absoluten Pfad mithilfe von pathlib
        self.csv_path = Path(base_dir) / csv_path

        if feature_cols is None:
            feature_cols = ["feat1", "feat2", "feat3"]
        if target_cols is None:
            target_cols = ["t1", "t2", "t3"]
            
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.transform = transform

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"[HyperbolicDataset] CSV file not found: {self.csv_path}")

        # CSV laden
        self.df = pd.read_csv(self.csv_path, sep=",", header=0, encoding="utf-8")
        print(f"[HyperbolicDataset] Loaded {len(self.df)} rows and {len(self.df.columns)} columns from {self.csv_path}")

        # Überprüfe, ob alle erforderlichen Spalten vorhanden sind
        missing_features = [col for col in self.feature_cols if col not in self.df.columns]
        missing_targets = [col for col in self.target_cols if col not in self.df.columns]
        if missing_features or missing_targets:
            raise ValueError(
                f"[HyperbolicDataset] Missing columns in CSV. "
                f"Missing features: {missing_features}, Missing targets: {missing_targets}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Gibt das Tupel (features, targets) als torch.Tensor zurück.
        
        Returns:
        --------
        tuple(torch.Tensor, torch.Tensor): (features, targets)
        """
        row = self.df.iloc[idx]
        # Extrahiere Features und Targets und wandle sie in numpy arrays mit dtype float32 um
        features = row[self.feature_cols].values.astype("float32")
        targets = row[self.target_cols].values.astype("float32")

        # Optionaler Transform
        if self.transform:
            features, targets = self.transform(features, targets)

        x_tensor = torch.tensor(features, dtype=torch.float32)
        y_tensor = torch.tensor(targets, dtype=torch.float32)
        return x_tensor, y_tensor

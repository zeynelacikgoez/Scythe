#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HyperConfig:
-------------
Diese Klasse speichert alle wesentlichen Hyperparameter und Dateipfade 
für das Hyperbolic-Transformer-Projekt. Die Pfade werden relativ zu einem 
definierten Basisverzeichnis (base_dir) gesetzt, sodass das Projekt portabler ist.
Die Verwendung von pathlib ermöglicht eine plattformunabhängige Pfadmanipulation.
"""

import os
from pathlib import Path

class HyperConfig:
    def __init__(self,
                 base_dir: str = ".",
                 train_data_path: str = "data/train.csv",
                 val_data_path: str = "data/val.csv",
                 checkpoint_dir: str = "checkpoints",
                 input_dim: int = 300,
                 hidden_dim: int = 512,
                 output_dim: int = 300,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 ff_dim: int = 2048,
                 dropout: float = 0.1,
                 use_pos_enc: bool = True,
                 batch_size: int = 8,
                 learning_rate: float = 1e-4,
                 epochs: int = 10,
                 use_gpu: bool = True,
                 resume_checkpoint_path: str = ""):
        """
        Konstruktor für HyperConfig.
        
        Parameter:
          - base_dir: Das Basisverzeichnis des Projekts.
          - train_data_path: Pfad zur Trainingsdaten-CSV (relativ zu base_dir).
          - val_data_path: Pfad zur Validierungsdaten-CSV (relativ zu base_dir).
          - checkpoint_dir: Verzeichnis, in dem Checkpoints gespeichert werden (relativ zu base_dir).
          - input_dim, hidden_dim, output_dim, num_layers, num_heads, ff_dim, dropout, use_pos_enc:
            Hyperparameter für das Modell.
          - batch_size, learning_rate, epochs, use_gpu: Trainingsparameter.
          - resume_checkpoint_path: Optionaler Pfad zu einem Checkpoint, um das Training fortzusetzen.
        """
        self.base_dir = Path(base_dir).resolve()
        self.train_data_path = self.base_dir / Path(train_data_path)
        self.val_data_path = self.base_dir / Path(val_data_path)
        self.checkpoint_dir = self.base_dir / Path(checkpoint_dir)
        self.resume_checkpoint_path = self.base_dir / Path(resume_checkpoint_path) if resume_checkpoint_path else ""

        # Modell-Hyperparameter
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.use_pos_enc = use_pos_enc

        # Trainingsparameter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.use_gpu = use_gpu

        # Erstelle notwendige Verzeichnisse
        self.ensure_dirs()

    def ensure_dirs(self):
        """
        Erstellt alle benötigten Verzeichnisse (z. B. für Checkpoints und Outputs),
        falls diese noch nicht existieren.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        return (
            f"HyperConfig(\n"
            f"  base_dir={self.base_dir},\n"
            f"  train_data_path={self.train_data_path},\n"
            f"  val_data_path={self.val_data_path},\n"
            f"  checkpoint_dir={self.checkpoint_dir},\n"
            f"  resume_checkpoint_path={self.resume_checkpoint_path},\n\n"
            f"  input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, output_dim={self.output_dim},\n"
            f"  num_layers={self.num_layers}, num_heads={self.num_heads}, ff_dim={self.ff_dim}, dropout={self.dropout},\n"
            f"  use_pos_enc={self.use_pos_enc},\n\n"
            f"  batch_size={self.batch_size}, learning_rate={self.learning_rate}, epochs={self.epochs}, use_gpu={self.use_gpu}\n"
            f")"
        )

# Beispiel: Wenn dieses Modul direkt ausgeführt wird
if __name__ == "__main__":
    config = HyperConfig(base_dir=".", resume_checkpoint_path="")  # resume_checkpoint_path leer, falls nicht benötigt
    print(config)

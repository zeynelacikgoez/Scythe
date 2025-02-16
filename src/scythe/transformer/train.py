#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train.py

Dieses Skript führt den Trainingslauf für den Hyperbolic Transformer durch.
Es nutzt die zentrale Konfigurationsklasse (HyperConfig), die über
`pathlib.Path` alle Dateipfade verwaltet, um eine flexible und robuste
Pfadbehandlung zu gewährleisten.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from geoopt.optim import RiemannianAdam
from pathlib import Path

# Eigene Module
from config import HyperConfig
from dataset import HyperbolicDataset
from model import HyperbolicTransformer
from utils import save_checkpoint, load_checkpoint, print_model_parameters


def train_model():
    """
    Führt einen kompletten Trainingslauf durch:
      1) Konfiguration laden
      2) Datensätze (Trainings- und Validierungs-CSV) laden
      3) Modell und Optimizer initialisieren
      4) Optional: Training aus einem Checkpoint wiederaufnehmen
      5) Trainingsschleife mit Validierung und Checkpoint-Speicherung
    """
    # 1) Konfiguration laden (die Pfade sind als Path-Objekte definiert)
    cfg = HyperConfig()

    # 2) Datensätze laden
    # Konvertiere Path-Objekte in Strings für Kompatibilität (oder passe HyperbolicDataset entsprechend an)
    train_ds = HyperbolicDataset(str(cfg.train_data_path))
    val_ds = HyperbolicDataset(str(cfg.val_data_path))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    # 3) Modell und Optimizer initialisieren
    model = HyperbolicTransformer(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.output_dim,
        curvature=cfg.curvature_init
    )
    print_model_parameters(model)

    device = torch.device("cuda" if (cfg.use_gpu and torch.cuda.is_available()) else "cpu")
    model.to(device)
    optimizer = RiemannianAdam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss()

    # 4) Optional: Checkpoint laden (falls resume_checkpoint_path gesetzt)
    start_epoch = 0
    resume_path = Path(cfg.resume_checkpoint_path) if cfg.resume_checkpoint_path else None
    if resume_path and resume_path.is_file():
        print(f"[train] Resuming from checkpoint: {resume_path}")
        start_epoch = load_checkpoint(model, optimizer, str(resume_path), map_location=str(device))

    # 5) Trainingsschleife
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        total_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{cfg.epochs} - Train Loss: {avg_loss:.4f}")

        val_loss = validate(model, val_loader, criterion, device)
        print(f"           Validation Loss: {val_loss:.4f}")

        # Checkpoint-Speicherung: Verwende cfg.checkpoint_dir (als Path) für den Pfad
        checkpoint_path = cfg.checkpoint_dir / f"model_epoch{epoch + 1}.pt"
        save_checkpoint(model, optimizer, epoch + 1, str(checkpoint_path))


def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    """
    Führt eine Validierungsschleife durch und berechnet den durchschnittlichen Loss.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(loader)


if __name__ == "__main__":
    train_model()

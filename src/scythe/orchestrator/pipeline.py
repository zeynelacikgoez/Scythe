#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from pathlib import Path

# Importiere Funktionen aus dem Daten-Pipeline-Paket
from scythe.data.ingestion import load_raw_data
from scythe.data.preprocessing import preprocess_data, create_embedding_tensor

# SDM und VSA
from scythe.sdm import SparseDistributedMemory

# Hyperbolic Transformer als Modell
from scythe.transformer.model import HyperbolicTransformer

# Torch DataLoader und TensorDataset
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

# Plan-Optimizer
from scythe.optimizer.plan_optim import optimize_plan


def run_entire_pipeline(data_dir: str):
    """
    Führt einen kompletten Pipeline-Lauf durch:
      1. Laden und Vorverarbeiten der Rohdaten.
      2. Erstellung eines Regressionsdatensatzes (Features und Ziel).
      3. Demonstration eines SDM (Sparse Distributed Memory).
      4. Training eines Hyperbolic Transformers als Regressor.
      5. Anwendung des Plan-Optimizers auf Basis der Prognose.
      6. Rückgabe des finalen Plans inklusive Report.
    """
    print("[pipeline] 1) Laden und Vorverarbeiten der Daten")
    # Erstelle einen absoluten Pfad zur Rohdaten-CSV
    raw_csv_path = Path(data_dir) / "raw_data.csv"
    df_raw = load_raw_data(str(raw_csv_path))
    df_clean = preprocess_data(df_raw)

    # Prüfe, ob das Ziel (z.B. 'demand_target') vorhanden ist
    if "demand_target" not in df_clean.columns:
        print("[pipeline] WARNUNG: 'demand_target' nicht im DataFrame vorhanden!")
        return None

    # Erstelle das Input-Embedding (z. B. aus log-transformierten Spalten)
    all_embeddings = create_embedding_tensor(df_clean)
    print("[pipeline] Embedding tensor shape:", all_embeddings.shape)

    # Erstelle den Ziel-Tensor aus der Spalte "demand_target"
    y_data = df_clean["demand_target"].values  # Shape: (N,)
    y_tensor = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)

    # Erstelle ein TensorDataset und splitte es in Training und Validierung (80:20)
    dataset = TensorDataset(all_embeddings, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    # -----------------------------------------
    # 2) SDM-Demo
    # -----------------------------------------
    print("[pipeline] 2) SDM-Demo")
    sdm_address_size = all_embeddings.shape[1]
    sdm = SparseDistributedMemory(
        address_size=sdm_address_size,
        memory_size=500,
        activation_percent=0.1,
        init_mode="random"
    )
    # Konvertiere das Embedding in bipolare Vektoren (z.B. +1, -1)
    X_np = all_embeddings.cpu().numpy()
    bipolars = np.where(X_np > 0, 1, -1)
    for i in range(min(10, len(bipolars))):
        sdm.write_memory(bipolars[i])
    # Test: Abruf des ersten Vektors
    retrieved = sdm.read_memory(bipolars[0])
    sim_sdm = np.mean(retrieved == bipolars[0])
    print(f"[pipeline] SDM retrieval similarity: {sim_sdm*100:.2f}%")

    # -----------------------------------------
    # 3) Hyperbolic Transformer als Regressor
    # -----------------------------------------
    print("[pipeline] 3) HyperbolicTransformer als Regressor")
    input_dim = all_embeddings.shape[1]
    output_dim = 1  # Regressionsziel
    hidden_dim = 256
    epochs = 5

    model = HyperbolicTransformer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        curvature=1.0,
        num_layers=2,
        num_heads=4,
        ff_dim=512,
        dropout=0.1,
        use_pos_enc=False,        # Da hier keine Sequenzstruktur erforderlich ist
        use_mobius_residual=False   # Für einfachere Residualverbindungen
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validierung
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                val_preds = model(x_val)
                loss_val = criterion(val_preds, y_val)
                val_loss += loss_val.item()
        val_loss /= len(val_loader)

        print(f"[pipeline] Epoch {epoch+1}/{epochs} - TrainLoss={avg_loss:.4f}, ValLoss={val_loss:.4f}")

    # -----------------------------------------
    # 4) Prognose & Plan-Optimierung
    # -----------------------------------------
    print("[pipeline] 4) Prognose + Plan-Optimierung")
    # Hier simulieren wir die Prognose, indem wir den ersten Validierungssample nutzen.
    if len(val_ds) > 0:
        x_future, _ = val_ds[0]
        x_future = x_future.unsqueeze(0).to(device)
        with torch.no_grad():
            model.eval()
            demand_forecast = model(x_future).item()
    else:
        demand_forecast = 50.0

    print(f"[pipeline] Prognostizierter Demand = {demand_forecast:.2f}")

    # Setze ein Plan-Szenario auf – hier exemplarisch:
    scenario_data = {
        "factories": [
            {"name": "FactoryA", "max_production": 100},
            {"name": "FactoryB", "max_production": 200},
        ],
        "demands": [demand_forecast],
        "max_resource": 150.0,
    }

    plan_solution, plan_obj_val, plan_report = optimize_plan(
        data=scenario_data,
        solver_type="ortools",
        use_heuristics=False,
        multi_objective_mode=False,
        scenario_generation=False,
        verbose=True
    )

    print("[pipeline] Best Plan obj =", plan_obj_val)
    print("[pipeline] Plan report:\n", plan_report)

    return {
        "plan_solution": plan_solution,
        "objective_value": plan_obj_val,
        "report": plan_report
    }


if __name__ == "__main__":
    data_folder = "./data"
    final_plan = run_entire_pipeline(data_folder)
    print("[pipeline] Final Plan:", final_plan)

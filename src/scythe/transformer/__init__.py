# __init__.py

"""
Das hyperbolic_transformer-Paket:
---------------------------------
Bündelt alle relevanten Komponenten für einen Hyperbolic-Transformer:
- config.py        : HyperConfig mit zentralen Hyperparametern
- dataset.py       : HyperbolicDataset, DataLoader-Integration
- model.py         : Hauptmodell HyperbolicTransformer und zugehörige Layer
- train.py         : Trainingsskript, train_model-Funktion
- inference.py     : Inferenzskript, run_inference-Funktion
- utils.py         : Hilfsfunktionen (Checkpoint-Handling, Distanz, CosSim etc.)
- modules/         : Enthält Teilmodule (PreNet, PostNet, Attention, etc.)

In diesem __init__-File können wir wichtige Elemente 
für externe Nutzer re-exportieren.
"""

# Importiere deine Schlüssel-Funktionen/Klassen aus den benachbarten Dateien
from .config import HyperConfig
from .dataset import HyperbolicDataset
from .model import HyperbolicTransformer
from .train import train_model
from .inference import run_inference
from .utils import (
    compute_cosine_similarity,
    hyperbolic_distance,
    print_model_parameters,
    save_checkpoint,
    load_checkpoint
)

# Falls du aus dem modules-Ordner direkt importieren willst:
from .modules import (
    PreNet,
    PostNet,
    HyperbolicCube,
    HyperbolicAttention,
    HyperbolicPositionalEncoding,
    HyperbolicFeedForward
)

__all__ = [
    "HyperConfig",
    "HyperbolicDataset",
    "HyperbolicTransformer",
    "train_model",
    "run_inference",
    "compute_cosine_similarity",
    "hyperbolic_distance",
    "print_model_parameters",
    "save_checkpoint",
    "load_checkpoint",

    # Re-Exports aus modules/
    "PreNet",
    "PostNet",
    "HyperbolicCube",
    "HyperbolicAttention",
    "HyperbolicPositionalEncoding",
    "HyperbolicFeedForward"
]

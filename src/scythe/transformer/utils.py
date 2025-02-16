# utils.py

import torch
import torch.nn.functional as F
import os
from typing import Optional
from geoopt import Manifold

def compute_cosine_similarity(x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Berechnet die Cosine Similarity zwischen zwei Tensoren x und y.
    
    Parameter:
    ----------
    x, y : torch.Tensor
        Tensoren gleicher Shape. Z. B. [batch_size, hidden_dim].
    dim : int
        Die Dimension, entlang der wir die Ähnlichkeit berechnen.

    Rückgabe:
    ---------
    torch.Tensor
        Cosine-Similarity-Werte. Bei [batch_size, hidden_dim] => Ausgabe [batch_size].
    """
    return F.cosine_similarity(x, y, dim=dim)


def hyperbolic_distance(u: torch.Tensor, v: torch.Tensor, manifold: Manifold) -> torch.Tensor:
    """
    Berechnet die Distanz zweier Tensoren u, v im übergebenen Manifold,
    z. B. PoincareBall. Die meisten Manifolds in 'geoopt' bieten .dist(u, v).

    Parameter:
    ----------
    u, v : torch.Tensor
        Punkte im hyperbolischen Raum, z. B. shape [batch_size, hidden_dim].
    manifold : geoopt.Manifold
        Ein geoopt.Manifold-Objekt, z. B. PoincareBall(c=1.0).

    Rückgabe:
    ---------
    torch.Tensor
        Distanzen => [batch_size] (wenn Input shape [batch_size, hidden_dim]).
    """
    return manifold.dist(u, v)


def print_model_parameters(model: torch.nn.Module) -> None:
    """
    Gibt alle Parameter eines Modells mit ihrem Namen, der Shape und
    dem 'requires_grad'-Status aus. Nützlich für Debugging.

    Parameter:
    ----------
    model : torch.nn.Module
        Ein beliebiges PyTorch-Modell, dessen Parameter inspiziert werden sollen.
    """
    for name, param in model.named_parameters():
        print(f"Param: {name}, shape={param.shape}, grad={param.requires_grad}")


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, checkpoint_path: str) -> None:
    """
    Speichert den Modell-Zustand (Weights) und den Optimizer-Zustand (z. B. Adam, RiemannianAdam)
    in eine Datei. Zusätzliche Informationen (z. B. aktueller Epoch-Zähler) werden ebenfalls abgelegt.

    Parameter:
    ----------
    model : torch.nn.Module
        Das trainierte Modell, dessen state_dict gespeichert wird.
    optimizer : torch.optim.Optimizer
        Der Optimizer (z. B. RiemannianAdam), dessen state_dict mitgespeichert wird.
    epoch : int
        Die aktuelle Epoche (oder beliebiger Zähler), den wir für das Wiederaufnehmen notieren.
    checkpoint_path : str
        Dateipfad, an den wir den Checkpoint speichern (z. B. "checkpoints/model_epoch10.pt").
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"[utils] Checkpoint saved at epoch {epoch} to {checkpoint_path}")


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    checkpoint_path: str, map_location: Optional[str] = None) -> int:
    """
    Lädt einen gespeicherten Modell- und Optimizer-Zustand und setzt
    das Modell/Optimizer in den entsprechenden Zustand zurück.

    Parameter:
    ----------
    model : torch.nn.Module
        Das (uninitialisierte) Modell, in das die Weights geladen werden.
    optimizer : torch.optim.Optimizer
        Der Optimizer, dessen Zustand (z. B. momenta) wiederhergestellt wird.
    checkpoint_path : str
        Pfad zur Checkpoint-Datei.
    map_location : Optional[str]
        Falls nicht None, wird die Datei auf ein bestimmtes Gerät gemappt
        (z. B. "cpu", "cuda:0"). Nützlich, wenn der Checkpoint auf GPU gespeichert
        wurde, man aber auf CPU laden möchte.

    Rückgabe:
    ---------
    int
        Die Epoche (bzw. Zähler), den wir aus dem Checkpoint auslesen.
        So kann das Training an entsprechender Stelle fortgesetzt werden.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)

    print(f"[utils] Checkpoint loaded from {checkpoint_path}, epoch {epoch}")
    return epoch

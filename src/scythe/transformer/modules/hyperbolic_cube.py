# modules/hyperbolic_cube.py

import torch
import torch.nn as nn
from typing import List

class HyperbolicCube(nn.Module):
    """
    HyperbolicCube:
    ---------------
    Ein mehrstufiges, pyramiden- oder cube-artiges Feed-Forward-Gebilde in
    hyperbolischem Raum.

    Ablauf (für jede Schicht):
      1) Euklidische Projektion via nn.Linear
      2) (Optional) Aktivierungsfunktion
      3) manifold.expmap0(...) => Mapping zurück in den hyperbolischen Raum

    Parameter:
    ----------
    layers_dims : List[int]
        Liste von Dimensionen, z. B. [512, 256, 128, 256, 512].
        Jede Schicht transformiert von layers_dims[i] -> layers_dims[i+1].
    manifold : geoopt.Manifold
        Ein Objekt, z. B. PoincareBall(c=1.0), um expmap0(...) aufzurufen.
    activation : Optional[Callable]
        Eine Aktivierungsfunktion (z. B. torch.relu). Falls None, keine Aktivierung.
        Du kannst hier z. B. nn.ReLU() oder nn.GELU() übergeben.
    use_norm : bool
        Falls True, wende nach dem Linearschritt eine LayerNorm an.
        (Dies kann bei größeren Netzen helfen, numerische Stabilität zu erhöhen.)
    dropout : float
        Wahrscheinlichkeit für Dropout nach jeder Schicht (0.0 => kein Dropout).
    """

    def __init__(
        self,
        layers_dims: List[int],
        manifold,
        activation=None,
        use_norm: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()

        self.manifold = manifold
        self.activation = activation
        self.use_norm = use_norm
        self.dropout_p = dropout

        # Liste von Schichten erzeugen
        self.layers = nn.ModuleList()
        for i in range(len(layers_dims) - 1):
            in_dim = layers_dims[i]
            out_dim = layers_dims[i + 1]

            layer = nn.Linear(in_dim, out_dim)
            self.layers.append(layer)

        # Optional Norm/Dropout in jeder Schicht
        if self.use_norm:
            # In der Regel eine Liste von Norm-Layern, einer pro Schicht (oder pro Output-Shape).
            self.norms = nn.ModuleList([
                nn.LayerNorm(layers_dims[i + 1]) for i in range(len(layers_dims) - 1)
            ])
        else:
            self.norms = None

        self.dropouts = (
            nn.ModuleList([nn.Dropout(self.dropout_p) for _ in range(len(layers_dims) - 1)])
            if self.dropout_p > 0.0 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward-Pass:
        -------------
        x shape: [batch_size, seq_len, layers_dims[0]]

        Für jede Schicht:
          - lineare Projektion
          - optional Aktivierung
          - optional Norm
          - Mapping in hyperbolischen Raum (expmap0)
          - optional Dropout

        Rückgabe: [batch_size, seq_len, layers_dims[-1]] (im hyperbolischen Raum)
        """
        for i, layer in enumerate(self.layers):
            # 1) lineare Projektion
            x = layer(x)

            # 2) optional Aktivierung
            if self.activation is not None:
                x = self.activation(x)

            # 3) optional Norm
            if self.norms is not None:
                x = self.norms[i](x)

            # 4) Mapping in hyperbolischen Raum
            x = self.manifold.expmap0(x)

            # 5) optional Dropout
            if self.dropouts is not None:
                x = self.dropouts[i](x)

        return x

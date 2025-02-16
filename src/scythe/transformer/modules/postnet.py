# modules/postnet.py

import torch
import torch.nn as nn

class PostNet(nn.Module):
    """
    PostNet:
    --------
    - Erwartet Daten im hyperbolischen Raum (z. B. Poincaré-Ball) mit Dimension 'hidden_dim'.
    - Führt mittels manifold.logmap0(...) die Abbildung zurück nach euklidischen Koordinaten durch.
    - Anschließend wendet es eine lineare Projektion auf 'output_dim' an.
    - (Optional) kann man LayerNorm / Dropout hinzufügen, um die Stabilität zu erhöhen.

    Parameter:
    ----------
    hidden_dim : int
        Die Größe der versteckten Repräsentation, die wir konvertieren wollen.
    output_dim : int
        Ziel-Dimension, z. B. für eine Vorhersage (Klassifikations-Logits, 
        embeddings, Regressionswerte).
    manifold : geoopt.Manifold
        Bietet Zugriff auf logmap0(...), z. B. PoincareBall(c=1.0).
    use_norm : bool
        Falls True, wird ein nn.LayerNorm(output_dim) nach der Projektion angewendet.
    dropout : float
        Dropout-Wahrscheinlichkeit. 0.0 => kein Dropout.
    activation : Optional[Callable]
        Eine Aktivierungsfunktion (z. B. torch.relu). Falls None, keine Aktivierung.

    Forward-Ablauf:
    1) logmap0(x) -> Euklidische Koordinaten
    2) lineare Projektion hidden_dim -> output_dim
    3) optional Aktivierung
    4) optional LayerNorm
    5) optional Dropout
    6) Rückgabe shape: [batch_size, seq_len, output_dim]
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        manifold,
        use_norm: bool = False,
        dropout: float = 0.0,
        activation=None
    ):
        super().__init__()
        self.manifold = manifold
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.use_norm = use_norm
        self.activation = activation

        # LayerNorm (optional)
        if self.use_norm:
            self.norm = nn.LayerNorm(output_dim)

        # Dropout (optional)
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [batch_size, seq_len, hidden_dim] (im hyperbolischen Raum)

        1) logmap0(x) -> Euklidischer Raum
        2) linear(x) -> Projektion auf output_dim
        3) optional Activation
        4) optional LayerNorm
        5) optional Dropout
        6) Rückgabe: [batch_size, seq_len, output_dim]
        """

        # 1) Aus Hyperbolischem Raum ins Euklidische
        euc_x = self.manifold.logmap0(x)  # [batch_size, seq_len, hidden_dim]

        # 2) Lineare Projektion
        out = self.linear(euc_x)  # [batch_size, seq_len, output_dim]

        # 3) Optionale Aktivierung
        if self.activation is not None:
            out = self.activation(out)

        # 4) Optionale LayerNorm
        if self.use_norm:
            out = self.norm(out)

        # 5) Optionales Dropout
        if self.dropout is not None:
            out = self.dropout(out)

        return out

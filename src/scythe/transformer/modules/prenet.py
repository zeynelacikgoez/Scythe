# modules/prenet.py

import torch
import torch.nn as nn

class PreNet(nn.Module):
    """
    PreNet:
    -------
    - Übernimmt rohes Input-Embedding (z. B. GloVe, BERT-Embeddings)
    - Projiziert es via nn.Linear in die 'hidden_dim'
    - (Optional) LayerNorm zur Stabilisierung
    - Wendet expmap0(...) an, um das Ergebnis in den hyperbolischen Raum
      des übergebenen 'manifold' (z. B. PoincareBall) zu mappen

    Parameter:
    ----------
    input_dim : int
        Eingangs-Dimension (z. B. 300 bei GloVe 300d)
    hidden_dim : int
        Ziel-Dimension für die versteckte Repräsentation (z. B. 512)
    manifold : geoopt.Manifold
        Manifold-Objekt, z. B. PoincareBall, um expmap0(...) aufzurufen
    use_norm : bool, optional
        Falls True, wird ein nn.LayerNorm auf das Zwischenresultat angewendet
    """

    def __init__(self, input_dim: int, hidden_dim: int, manifold, use_norm: bool = False):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.manifold = manifold
        self.use_norm = use_norm

        if use_norm:
            self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward-Pass:
        -------------
        x shape: [batch_size, seq_len, input_dim]

        1) Lineare Transformation von input_dim -> hidden_dim
        2) (Optional) LayerNorm
        3) Expmap in den hyperbolischen Raum
        4) Rückgabe: [batch_size, seq_len, hidden_dim] in hyperbolischer Darstellung
        """
        # Schritt 1: lineare Projektion
        x = self.linear(x)

        # Schritt 2: (optional) Normalisierung
        if self.use_norm:
            x = self.norm(x)

        # Schritt 3: in hyperbolischen Raum abbilden
        x = self.manifold.expmap0(x)

        return x

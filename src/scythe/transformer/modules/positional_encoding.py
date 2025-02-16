# modules/positional_encoding.py

import math
import torch
import torch.nn as nn

class HyperbolicPositionalEncoding(nn.Module):
    """
    HyperbolicPositionalEncoding:
    -----------------------------
    Ergänzt "sinusförmige" Positionskodierungen für Sequenzmodelle, 
    mit der Option, das Ergebnis ins hyperbolische Manifold (expmap0) 
    zu überführen.

    Parameter:
    ----------
    hidden_dim : int
        Dimensionalität der versteckten Repräsentation, 
        in die wir Positionen einbetten.
    max_len : int
        Maximale Sequenzlänge, für die wir die Kodierung vorab berechnen.
    manifold : geoopt.Manifold oder None
        Falls nicht None, verwenden wir manifold.expmap0(...), 
        um die Positionsinformation in den hyperbolischen Raum zu mappen.
    dropout : float
        Dropout-Wahrscheinlichkeit auf die Positionskodierung.
    apply_expmap : bool
        Falls True, wenden wir manifold.expmap0(...) auf die 
        sum(Embeddings + PosEncoding) an.
        Falls False, bleibt das Resultat euklidisch.

    Funktionsweise:
    1) Erzeuge sinusförmige Kodierung (klassisch).
    2) Summiere x + positional_encoding[:seq_len].
    3) Dropout.
    4) Optional manifold.expmap0(...).

    Typischer Aufruf:
    x = pos_enc_layer(x)  # x shape: [batch_size, seq_len, hidden_dim]
    """

    def __init__(
        self, 
        hidden_dim: int, 
        max_len: int = 5000,
        manifold=None, 
        dropout: float = 0.0,
        apply_expmap: bool = False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.manifold = manifold
        self.apply_expmap = apply_expmap
        self.dropout = nn.Dropout(dropout)

        # Positionskodierung vorberechnen: [max_len, hidden_dim]
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Frequenzfaktoren
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        # Sinus / Cosinus abwechselnd
        pe[:, 0::2] = torch.sin(position * div_term)  # gerade Dimensionen
        pe[:, 1::2] = torch.cos(position * div_term)  # ungerade Dimensionen

        # shape anpassen => [1, max_len, hidden_dim]
        pe = pe.unsqueeze(0)  
        # Als konstante (nicht trainierbare) Variable registrieren
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [batch_size, seq_len, hidden_dim]

        1) Addiere x + pe (auf seq_len beschränkt)
        2) dropout
        3) optional manifold.expmap0(...)
        4) Rückgabe: [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.size()

        # Positionskodierung addieren (im euklidischen Raum)
        # pe shape: [1, max_len, hidden_dim] => slice auf seq_len
        pos = self.pe[:, :seq_len, :]
        out = x + pos

        # Dropout
        out = self.dropout(out)

        # Optional in hyperbolischen Raum
        if self.manifold is not None and self.apply_expmap:
            out = self.manifold.expmap0(out)

        return out

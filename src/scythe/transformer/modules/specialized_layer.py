# specialized_layer.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# geoopt für hyperbolische Operationen (Möbius-Add etc.)
import geoopt
import geoopt.manifolds.stereographic.math as gmath

class HyperbolicFeedForward(nn.Module):
    """
    HyperbolicFeedForward:
    ----------------------
    Eine zweistufige FFN-artige Schicht, die Euklidische Operationen (Linears, Aktivierung)
    mit hyperbolischen Mappings (expmap0/logmap0) kombiniert. Zusätzlich kann eine
    Möbius-Addition für Residual-Verbindungen genutzt werden, um 'echte' hyperbolische
    Residuals abzubilden.

    Parameter:
    ----------
    d_model : int
        Eingangs- und Ausgangsdimension (z. B. 512).
    d_ff : int
        Breite der versteckten Zwischenschicht (z. B. 2048).
    manifold : geoopt.Manifold
        z. B. PoincareBall(c=1.0) für expmap0/logmap0.
    dropout : float
        Dropout-Wahrscheinlichkeit auf Zwischenschritte.
    activation : Callable, optional
        Aktivierungsfunktion, z. B. F.relu oder nn.GELU().
        Falls None, wird F.relu verwendet.
    use_residual : bool
        Falls True, wird ein Residual-Zweig eingebaut.
    use_norm : bool
        Falls True, wendet man nach dem finalen Schritt eine LayerNorm an.
    final_hyper : bool
        Falls True, kehren wir am Ende in den hyperbolischen Raum (expmap0) zurück.
        Falls False, bleibt das Ergebnis euklidisch.
    use_mobius_add : bool
        Falls True und use_residual=True, wird eine Möbius-Addition statt
        einer vereinfachten euklidischen Addition beim Residual verwendet.

    Forward-Ablauf (vereinfacht):
    1) logmap0(x) -> Euklidische Koordinaten
    2) Linear1 -> Aktivierung -> Dropout
    3) expmap0 -> wieder hyperbolisch
    4) logmap0 -> euklidisch
    5) Linear2 -> Dropout
    6) Residual-Verbindung (entweder euklidisch oder Möbius, je nach use_mobius_add)
    7) (Optional) Norm
    8) (Optional) expmap0 (final_hyper=True)

    Rückgabe: [batch_size, seq_len, d_model]
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        manifold,
        dropout: float = 0.1,
        activation=None,
        use_residual: bool = False,
        use_norm: bool = False,
        final_hyper: bool = True,
        use_mobius_add: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.manifold = manifold
        self.dropout_p = dropout
        self.activation = activation if activation is not None else F.relu
        self.use_residual = use_residual
        self.use_norm = use_norm
        self.final_hyper = final_hyper
        self.use_mobius_add = use_mobius_add

        # Lineare Projektionen
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        # Dropout-Layer
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Optionale LayerNorm
        if self.use_norm:
            self.norm = nn.LayerNorm(d_model)

        # Falls das Manifold z. B. ein PoincareBall(c=1.0) ist, können wir c hier abgreifen
        # (Kann auch None sein, je nach Implementation)
        if hasattr(manifold, 'c'):
            self.curvature = self.manifold.c
        else:
            self.curvature = 1.0  # Default

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [batch_size, seq_len, d_model] (vermutlich hyperbolisches Input)
        """
        # Eventuelles Residual "merken"
        if self.use_residual:
            residual = x  # Aktuelles x als hyperbolischer "Res"-Pfad

        # 1) logmap0 => euklidisch
        x_euc = self.manifold.logmap0(x)

        # 2) Linear1 + Activation + Dropout
        out = self.linear1(x_euc)
        out = self.activation(out)
        out = self.dropout1(out)

        # 3) expmap0 => hyper
        out = self.manifold.expmap0(out)

        # 4) logmap0 => erneut euklidisch für 2. Linear
        out_euc2 = self.manifold.logmap0(out)

        # 5) Linear2 + Dropout
        out_euc2 = self.linear2(out_euc2)
        out_euc2 = self.dropout2(out_euc2)

        # 6) Residual (euklidisch oder Möbius)
        if self.use_residual:
            # Euklidisch + final -> hyper?
            if not self.use_mobius_add:
                # (Vereinfachte Variante: out_euc2 + logmap0(residual))
                residual_euc = self.manifold.logmap0(residual)
                out_euc2 = out_euc2 + residual_euc
            else:
                # (Hyperbolisch: Möbius-Addition)
                # Wir gehen zurück in den Hyper-Raum:
                out_hyp2 = self.manifold.expmap0(out_euc2)
                # Möbius-Add (residual = hyper, out_hyp2 = hyper)
                #   c => self.curvature
                out_hyp2 = gmath.mobius_add(out_hyp2, residual, k=self.curvature, dim=-1)
                # => out_euc2 ist jetzt "out_hyp2" in hyper, wir können es 
                #    für Normierung evtl. logmap0en, je nachdem
                out_euc2 = self.manifold.logmap0(out_hyp2)

        # 7) (Optional) LayerNorm
        # Achtung: LayerNorm ist in euklidischen Koordinaten einfacher
        if self.use_norm:
            out_euc2 = self.norm(out_euc2)

        # 8) finaler Step => ggf. zurück ins Hyper (final_hyper=True)
        if self.final_hyper:
            out_final = self.manifold.expmap0(out_euc2)
        else:
            out_final = out_euc2

        return out_final

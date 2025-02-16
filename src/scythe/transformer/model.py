# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# geoopt für hyperbolisches Manifold
import geoopt
from geoopt import PoincareBall
import geoopt.manifolds.stereographic as sm

# Eigene Module aus dem Ordner "modules/"
from modules.prenet import PreNet
from modules.postnet import PostNet
from modules.hyperbolic_attention import HyperbolicAttention
from modules.specialized_layer import HyperbolicFeedForward
from modules.positional_encoding import HyperbolicPositionalEncoding


class HyperbolicTransformer(nn.Module):
    """
    HyperbolicTransformer:
    ----------------------
    - Ermöglicht standardmäßig eine 'Encoder-only' Struktur
      mit hyperbolischen Layern und Attention.
    - Kann als Autoencoder oder als Regressor/Klassifikator
      genutzt werden, je nach 'output_dim' und Trainingsloop.

    Parameter:
    ----------
    input_dim : int
        Eingabedimension (z. B. 300).
    hidden_dim : int
        Größe der versteckten Repräsentation (z. B. 512).
    output_dim : int
        Enddimension des Outputs. (Bei Regression z. B. 1)
    curvature : float
        Startwert für die lernbare Krümmung 'c' (z. B. 1.0).
    num_layers : int
        Anzahl der HyperbolicTransformerLayer.
    num_heads : int
        Anzahl der Attention-Heads pro Layer.
    ff_dim : int
        Zwischendimension im FeedForward (z. B. 2048).
    dropout : float
        Dropout-Wahrscheinlichkeit.
    use_pos_enc : bool
        Ob eine hyperbolische Positionskodierung genutzt wird.
    use_mobius_residual : bool
        Falls True, verwendet jeder Layer anstatt
        euklidischem residual-add die Möbius-Addition
        (siehe `HyperbolicTransformerLayer`).
    """

    def __init__(
        self,
        input_dim: int = 300,
        hidden_dim: int = 512,
        output_dim: int = 300,
        curvature: float = 1.0,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        use_pos_enc: bool = True,
        use_mobius_residual: bool = False
    ):
        super().__init__()

        # Lernbare Krümmung (sofern geoopt das zulässt)
        self.curvature = nn.Parameter(torch.tensor(curvature, requires_grad=True))
        # Haupt-Manifold: PoincareBall
        self.manifold = PoincareBall(c=self.curvature)

        # Möbius-Operatoren (alternative Implementation)
        self.mobius = sm.Mobius()  # aus geoopt.manifolds.stereographic
        self.use_mobius_residual = use_mobius_residual

        # PreNet: Input -> hidden_dim (Expmap)
        self.prenet = PreNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            manifold=self.manifold,
            use_norm=False
        )

        # (Optionale) Positionskodierung
        self.use_pos_enc = use_pos_enc
        if self.use_pos_enc:
            self.pos_encoding = HyperbolicPositionalEncoding(
                hidden_dim=hidden_dim,
                max_len=5000,
                manifold=self.manifold,
                dropout=dropout,
                apply_expmap=False  # Wir sind nach PreNet schon im Hyp-Raum
            )

        # Stack aus N = num_layers 'HyperbolicTransformerLayer'
        self.layers = nn.ModuleList([
            HyperbolicTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                manifold=self.manifold,
                dropout=dropout,
                use_mobius_residual=self.use_mobius_residual
            )
            for _ in range(num_layers)
        ])

        # PostNet: hidden_dim -> output_dim (Logmap + Linear)
        self.postnet = PostNet(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            manifold=self.manifold,
            use_norm=False,
            dropout=0.0,
            activation=None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [batch_size, seq_len, input_dim] (bei batchweiser Verarbeitung)
                 oder [batch_size, input_dim] (falls seq_len=1)
        Ablauf:
          1) PreNet  (=> hyperbolischer Raum)
          2) Optional: PositionalEncoding
          3) N x HyperbolicTransformerLayer
          4) PostNet => z. B. Euklidische Regressions-Outputs

        Rückgabe:
          [batch_size, seq_len, output_dim]
            oder (batch_size, output_dim), je nachdem ob seq_len=1
        """
        # 1) PreNet
        # Falls x nur 2D ist [batch_size, input_dim], unsqueeze seq_len=1
        if x.dim() == 2:
            x = x.unsqueeze(1)  # => shape [B, 1, input_dim]

        # -> [B, seq_len, hidden_dim (hyp)]
        x = self.prenet(x)

        # 2) (Optional) Positionskodierung
        if self.use_pos_enc:
            x = self.pos_encoding(x)

        # 3) Mehrere Transformer-Layer
        for layer in self.layers:
            x = layer(x)

        # 4) PostNet => euklidischer Output
        out = self.postnet(x)

        return out.squeeze(1)  # [B, seq_len, output_dim] -> i.d.R. [B, output_dim] bei seq_len=1


class HyperbolicTransformerLayer(nn.Module):
    """
    Ein 'Transformer-Block', der hyperbolische Self-Attention
    und ein hyperbolisches Feed-Forward-Netz enthält.

    Parameter:
    ----------
    hidden_dim : int
        Dimension der Eingabe/Output.
    num_heads : int
        Anzahl an Köpfen in der HyperbolicAttention.
    ff_dim : int
        Weite der Zwischenschicht im FeedForward.
    manifold : geoopt.Manifold
        z. B. PoincareBall(c=1.0).
    dropout : float
        Dropout-Wahrscheinlichkeit.
    use_mobius_residual : bool
        Falls True, wird beim Residual-Schritt Möbius-Addition verwendet.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        manifold,
        dropout: float = 0.1,
        use_mobius_residual: bool = False
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.manifold = manifold
        self.use_mobius_residual = use_mobius_residual

        # Möbius-Objekt für Residual
        self.mobius = sm.Mobius()

        # Hyperbolische Attention
        self.attn = HyperbolicAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            manifold=self.manifold,
            dropout=dropout,
            use_hyp_output=True
        )

        # Euklidische LayerNorm (als grobe Annäherung)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Hyperbolisches Feed-Forward
        self.ffn = HyperbolicFeedForward(
            d_model=hidden_dim,
            d_ff=ff_dim,
            manifold=self.manifold,
            dropout=dropout,
            activation=nn.ReLU(),
            use_residual=False,    # Residual wird hier extra gemacht
            use_norm=False,
            final_hyper=True       # wir bleiben in Hyp-Raum
        )

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, hidden_dim] (hyp)

        1) Self-Attention
        2) Residual + Norm
        3) Feed-Forward
        4) Residual + Norm
        """
        # 1) Attention (hyp) => (hyp)
        attn_out = self.attn(x)  # [B, seq_len, hidden_dim] (hyp)

        # 2) Residual + Norm
        out = self._residual_add(x, attn_out)
        out = self.norm1(out)
        out = self.dropout1(out)

        # 3) Feed Forward => [B, seq_len, hidden_dim] (hyp)
        ffn_out = self.ffn(out)

        # 4) Residual + Norm
        out = self._residual_add(out, ffn_out)
        out = self.norm2(out)
        out = self.dropout2(out)

        return out

    def _residual_add(self, hyp_x: torch.Tensor, hyp_y: torch.Tensor) -> torch.Tensor:
        """
        Führt eine Residual-Verknüpfung durch:
        - Euklidischer Fallback (logmap0->Add->expmap0)
        - oder Möbius-Addition (falls use_mobius_residual=True)
        """
        if self.use_mobius_residual:
            # Möbius-Addition
            return self.mobius.add(hyp_x, hyp_y, c=self.manifold.c)
        else:
            # Euklidischer Fallback: logmap0 -> add -> expmap0
            x_euc = self.manifold.logmap0(hyp_x)
            y_euc = self.manifold.logmap0(hyp_y)
            sum_euc = x_euc + y_euc
            return self.manifold.expmap0(sum_euc)

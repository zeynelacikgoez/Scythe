# hyperbolic_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt.manifolds.stereographic as sm
from geoopt import Manifold

class HyperbolicAttention(nn.Module):
    """
    HyperbolicAttention:
    --------------------
    Bietet zwei Modi:
      (1) 'simple' => wie bisher: logmap0 => lineare Q/K/V => dot product => softmax => out => expmap0
      (2) 'mobius' => experimenteller Versuch, Q/K/V in Hyp-Raum zu rechnen (mobius matmul, etc.)

    Parameter:
    ----------
    hidden_dim : int
        Größe der Eingabe- und Ausgabe-Features.
    num_heads : int
        Anzahl an Attention-Heads (Multi-Head).
    manifold : geoopt.Manifold
        z. B. PoincareBall(c=1.0) oder Lorentz.
    dropout : float
        Dropout-Wahrscheinlichkeit in der Attention-Berechnung.
    use_hyp_output : bool
        Falls True, wird das finale Output wieder mit expmap0 in den
        hyperbolischen Raum gemappt.
    attention_mode : str
        "simple" => Euklid. Dot-Product via logmap0.
        "mobius" => experimenteller mobius-based attention (Democode).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        manifold: Manifold,
        dropout: float = 0.1,
        use_hyp_output: bool = True,
        attention_mode: str = "simple"
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim muss durch num_heads teilbar sein.")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.manifold = manifold
        self.use_hyp_output = use_hyp_output
        self.attention_mode = attention_mode.lower()

        self.head_dim = hidden_dim // num_heads

        # Projektions-Layer für Q, K, V (klassisches Multi-Head-Setup)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Projektions-Layer für das Ergebnis
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Dropout für Attention-Gewichte
        self.dropout = nn.Dropout(dropout)

        # Optional: Möbius-Operatoren, falls 'mobius'
        self.mobius = sm.Mobius()

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Forward-Pass:
        -------------
        x shape: [batch_size, seq_len, hidden_dim] (hyperbolisch)
        mask: Optional, z. B. [batch_size, seq_len, seq_len]

        Ablauf:
          1) falls attention_mode='simple':
               - logmap0(x) => euklidischer Raum
               - Q, K, V => Dot => Softmax => Weighted Sum => out_proj
               - ggf. expmap0 => hyper
          2) falls attention_mode='mobius': (experimentell)
               - Q, K, V in hyperbolischem Raum konstruieren
               - "mobius dot" o. Ä. (hier stark vereinfacht)
               - Weighted Sum / expmap0 etc.
        """
        B, S, H = x.shape

        if self.attention_mode == "mobius":
            return self._mobius_attention(x, mask)
        else:
            # Standard "simple" attention
            return self._simple_attention(x, mask)

    # ----------------------------------------------------------
    # Variante 1: "Simple" => Euklidische Dot-Product-Attention
    # ----------------------------------------------------------
    def _simple_attention(self, hyp_x: torch.Tensor, mask=None) -> torch.Tensor:
        # 1) logmap0 => euklidisch
        euc_x = self.manifold.logmap0(hyp_x)

        # 2) Q, K, V (linear)
        Q = self.q_proj(euc_x)  # [B, S, H]
        K = self.k_proj(euc_x)
        V = self.v_proj(euc_x)

        # Aufteilen in Heads
        Q = Q.view(-1, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, nH, S, head_dim]
        K = K.view(-1, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(-1, S, self.num_heads, self.head_dim).transpose(1, 2)

        # 3) Dot-Product: Q*K^T / sqrt(dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, nH, S, S]

        # Mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 4) Weighted Sum
        context = torch.matmul(attn_probs, V)  # [B, nH, S, head_dim]

        # 5) Zurück auf [B, S, H]
        context = context.transpose(1, 2).contiguous().view(B, S, H)

        # 6) out_proj => euklidisch
        out_euc = self.out_proj(context)

        # 7) Ggf. expmap0 => hyper
        if self.use_hyp_output:
            out_hyp = self.manifold.expmap0(out_euc)
            return out_hyp
        else:
            return out_euc

    # ----------------------------------------------------------
    # Variante 2: "mobius" => Experimentelle Möbius-/Hyp-Attention
    # ----------------------------------------------------------
    def _mobius_attention(self, hyp_x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Sehr vereinfachte Demo einer "mobius-based" Attention:
          - Q, K, V durch 'mobius_matvec' oder 'mobius_add' anstatt
            euklidischer MatrixMultiplikation
          - "Dot-Product" kann man durch geodätische Distanz
            oder "mobius inner product" approximieren

        ACHTUNG: Stark vereinfacht und nicht repräsentativ für
        eine echte Lorentz-/Riemann-Attention.
        """
        B, S, H = hyp_x.shape

        # 1) Q, K, V in Hyp-Raum
        #    Idee: anstatt logmap0 => euk => linear,
        #    könnte man "mobius_matvec" (Matrix-Vekt Multiplikation in Hyp).
        #    Hier nur Demo:

        # Erzeuge euk. Gewichte
        Wq = self.q_proj.weight  # shape [H, H]
        bq = self.q_proj.bias
        Wk = self.k_proj.weight
        bk = self.k_proj.bias
        Wv = self.v_proj.weight
        bv = self.v_proj.bias

        # Flatten x => [B*S, H]
        x_flat = hyp_x.view(B*S, H)

        # "Mobius" Multiplikation (Demo):
        Q_hyp = self._mobius_linear(x_flat, Wq, bq)
        K_hyp = self._mobius_linear(x_flat, Wk, bk)
        V_hyp = self._mobius_linear(x_flat, Wv, bv)

        # -> [B, S, H]
        Q_hyp = Q_hyp.view(B, S, H)
        K_hyp = K_hyp.view(B, S, H)
        V_hyp = V_hyp.view(B, S, H)

        # 2) Aufteilen in Heads
        Q_hyp = Q_hyp.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K_hyp = K_hyp.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V_hyp = V_hyp.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # 3) "Dot-Product": anstelle euklidisch:
        #    man könnte Distanz interpretieren => exp(-dist^2), etc.
        #    Hier nur naive: logmap0 => euk => matmul => softmax
        Q_euc = self.manifold.logmap0(Q_hyp)
        K_euc = self.manifold.logmap0(K_hyp)

        attn_scores = torch.matmul(Q_euc, K_euc.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Weighted Sum -> in Euklid
        V_euc = self.manifold.logmap0(V_hyp)
        context_euc = torch.matmul(attn_probs, V_euc)

        # 4) Zurück in Hyp-Raum
        context_hyp = self.manifold.expmap0(context_euc)

        # 5) out_proj => euk
        out_euc = self.out_proj(self.manifold.logmap0(context_hyp))
        # 6) optional expmap0
        if self.use_hyp_output:
            out_hyp = self.manifold.expmap0(out_euc)
            return out_hyp
        else:
            return out_euc

    def _mobius_linear(self, x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Demo-Funktion für "mobius" lineare Transformation:
        x, W, b euklidisch, aber wir interpretieren x in Hyp-Raum und
        wenden mobius_add / mobius_matvec an.
        """
        # x_hyp = manifold.expmap0(x) falls x euk.
        # Hier: x schon in Hyp-Raum (?), wir "logmap0" falls wir unsicher sind
        # => Demo: euk_x = logmap0(x), dann normal linear, dann expmap0.
        euk_x = self.manifold.logmap0(x)
        # euk_x shape: [N, H], W: [H, H], b: [H]
        lin = euk_x @ W.t() + b

        # expmap0 zurück => Hyp
        hyp_lin = self.manifold.expmap0(lin)
        return hyp_lin

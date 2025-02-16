```markdown
# modules/ — Hyperbolic Transformer Modules

In diesem Ordner befinden sich die **Bausteine** („Modules“) für den Hyperbolic Transformer. Jede Python-Datei implementiert oder exportiert eine spezifische Komponente (z. B. PreNet, PostNet, Attention, Pyramid-Layer etc.), die im Hauptmodell (`model.py`) zu einer umfassenden hyperbolischen Transformer-Architektur zusammengefügt werden.

---

## 1. `prenet.py` — **PreNet**
- **Zweck:**  
  Die PreNet-Klasse übernimmt die Vorverarbeitung der Eingabedaten. Typischerweise:
  1. Lineare Projektion von `input_dim` → `hidden_dim`.
  2. (Optional) Normalisierung (LayerNorm).
  3. **expmap0** ins hyperbolische Manifold.

- **Typischer Aufruf:**  
  ```python
  x = self.prenet(x)  # [batch_size, seq_len, input_dim] -> [batch_size, seq_len, hidden_dim (hyperbolisch)]
  ```

---

## 2. `postnet.py` — **PostNet**
- **Zweck:**  
  Rücktransformation aus dem hyperbolischen Raum zurück ins euklidische Raum oder direkt zum finalen Output.  
  1. **logmap0** für die Koordinatenumwandlung.
  2. Lineare Projektion von `hidden_dim` → `output_dim`.
  3. (Optional) Aktivierung, Norm, Dropout.

- **Typischer Aufruf:**  
  ```python
  output = self.postnet(x)  # [batch_size, seq_len, hidden_dim (hyp)] -> [batch_size, seq_len, output_dim (euklidisch)]
  ```

---

## 3. `hyperbolic_cube.py` — **HyperbolicCube**
- **Zweck:**  
  Mehrstufiges (pyramidenartiges) Feed-Forward-Netz innerhalb des hyperbolischen Raums.  
  1. Linearer Schritt (Euklidisch).  
  2. (Optional) Aktivierung (ReLU / GELU etc.).  
  3. **expmap0** ins hyperbolische Manifold.  
  4. Wiederholen für mehrere Layer-Dimensionen (etwa `[512, 256, 128, 256, 512]`).

- **Typischer Aufruf:**  
  ```python
  x = self.hyperbolic_cube(x)  # Nutzt in jeder Schicht expmap0
  ```

---

## 4. `hyperbolic_attention.py` — **HyperbolicAttention**
- **Zweck:**  
  Experimentelle Multi-Head-Attention, die Eingaben aus dem hyperbolischen Raum per **logmap0** in den euklidischen Raum bringt, dort Q/K/V und Dot-Product-Attention berechnet, und optional **expmap0** zurück in den hyperbolischen Raum durchführt.

- **Ablauf (vereinfacht):**  
  1. `logmap0(x)`  
  2. Euklidisches Dot-Product-Attention (Q·K / sqrt(d)).  
  3. **expmap0** der Ergebnisse für die Rückkehr ins hyperbolische Setting.  

- **Typischer Aufruf:**  
  ```python
  out = self.hyperbolic_attention(x)  # x: [batch, seq, hidden_dim (hyp)]
  ```

---

## 5. `positional_encoding.py` — **HyperbolicPositionalEncoding**
- **Zweck:**  
  Implementiert eine Positionskodierung, ähnlich dem sinus-cosinus-Verfahren aus dem Original-Transformer, wobei man optional die Ergebnisse in den hyperbolischen Raum mappt.  
  - `self.pe`: vordefinierte Sinus-/Cos-Funktionen für `max_len` Sequenzlänge.  
  - Summiert sie mit `x`, gefolgt von Dropout und ggf. `expmap0`.

- **Typischer Aufruf:**  
  ```python
  x = self.pos_encoding(x)
  ```

---

## 6. `specialized_layer.py` — **HyperbolicFeedForward**
- **Zweck:**  
  Ein **zweischichtiges** (FFN-)Modul, das abwechselnd logmap0/expmap0 verwendet, um euklidische Operationen (Linears, Aktivierung) in das hyperbolische Setting zu integrieren.  
  - **Residual**: ggf. “mobius addition” oder eine vereinfachte Log-Exp-Add-Variante.  
  - Kann optional Norm & Dropout integrieren.

- **Typischer Ablauf:**  
  1. `logmap0` → Linear(… ) → ReLU → Dropout → `expmap0`.  
  2. Wieder `logmap0` → Linear(… ) → Dropout.  
  3. (Optional) Residual + Norm.  
  4. (Optional) `expmap0` als finaler Schritt.

---

## Wie diese Module zusammenarbeiten

In der **Hauptarchitektur** (z. B. `HyperbolicTransformer` in `model.py`) werden diese Komponenten so zusammengesetzt:

1. **PreNet**: `input_dim` → `hidden_dim`, plus expmap0.  
2. (Optional) **HyperbolicPositionalEncoding**: sinus/cos-Position + (ggf.) expmap0.  
3. **HyperbolicAttention** & **HyperbolicFeedForward**: in jedem Transformer-Layer.  
4. **PostNet**: Letzte logmap0 + lineare Projektion zum finalen Output.

Das Resultat ist eine **transformerähnliche** Pipeline, nur dass wichtige Operationen (zumindest teilweise) im hyperbolischen Raum stattfinden.

---

## Tipps und Hinweise

1. **Numerische Stabilität**  
   - Beim häufigen Hin- und Her-Wechsel zwischen logmap0 / expmap0 können Rundungsfehler auftreten. Achte auf Toleranzen oder verwende “mobius addition” für Residuals (siehe Forschungsliteratur).

2. **Residual-Verbindungen**  
   - Klassische Residualverbindungen (x + Sublayer(x)) gelten euklidisch. In hyperbolischen Netzen braucht man i. d. R. anstelle dessen Möbius-Additionen. Die hier gezeigten Layer sind nur **Beispiel**-Implementierungen.

3. **geoopt**  
   - Diese Bibliothek (`geoopt`) liefert die Funktionen `expmap0`, `logmap0`, `dist` etc. für verschiedene Manifolds, u. a. `PoincareBall`.  
   - Sie unterstützt auch RiemannianAdam als Optimizer.

4. **Anpassung**  
   - Du kannst die Layer aus diesen Modulen für **Encoder**, **Decoder** oder reine **Feed-Forward**-Architekturen verwenden.   
   - Je nach Ziel (Klassifikation, Reconstruction, Embedding-Simulation, etc.) brauchst du vielleicht andere Output-Aktivierungen (z. B. Softmax).

---

## Fazit

Der Ordner `modules/` bildet den **Baukasten** für deinen **HyperbolicTransformer**:  
- **Prä- und PostNet** kümmern sich um das Mapping (Input ↔ Hidden ↔ Output).  
- **HyperbolicCube**, **HyperbolicAttention**, **HyperbolicFeedForward** stellen die inneren Bausteine für komplexe Deep Networks im hyperbolischen Raum bereit.  
- **PositionalEncoding** rundet das Ganze für Sequenzmodelle ab.

Nutze diese Module in deinem Projekt, um flexible, modulare und (je nach Bedarf) **hyperbolisch inspirierte** Deep-Learning-Architekturen umzusetzen.
```
```markdown
# sdm/ — Sparse Distributed Memory & VSA

In diesem Ordner befinden sich die **Implementierungen** für:
1. **Sparse Distributed Memory** (Kanervas Modell)  
2. **Vector-Symbolic Architecture** (VSA)  
3. **Hilfsfunktionen** (z. B. Hamming-Distanz, Cosine Similarity, Binär-/Bipolar-Konvertierungen)

---

## Ordnerstruktur

```
sdm/
├── __init__.py
├── sdm_module.py
├── vsa_module.py
├── sdm_utils.py
├── tests/
│   ├── __init__.py
│   ├── test_sdm_module.py
│   └── test_vsa_module.py
└── readme.md
```

### Kurze Beschreibung

- **`__init__.py`**  
  Kennzeichnet das Verzeichnis als Python-Paket und re-exportiert zentrale Klassen/Funktionen wie `SparseDistributedMemory`, `VSA`, oder `hamming_distance`, sodass sie bequem via `from sdm import ...` importierbar sind.

- **`sdm_module.py`**  
  Enthält die Klasse `SparseDistributedMemory`, die ein assoziatives Gedächtnis nach Kanerva (1988) implementiert.  
  - **write_memory(...)**: Speichert einen bipolaren Eingabevektor in die nächstgelegenen “Hard Locations.”  
  - **read_memory(...)**: Ruft den gespeicherten Inhalt assoziativ wieder ab, indem es die k ähnlichsten Adressen (nach Hamming-Distanz) auswählt.  

- **`vsa_module.py`**  
  Definiert eine Basis-Klasse `VSA` für Vector-Symbolic Architectures:  
  - **random_hypervector()**: Erzeugt einen zufälligen Hypervektor (+1/-1).  
  - **bind(...)**: Elementweise Multiplikation zweier Hypervektoren (Binding).  
  - **superpose(...)**: Elementweise Addition und Sign-Normalisierung (Superposition).  
  - **store_memory(...)** & **retrieve_memory(...)**: Einfaches Cleanup Memory auf Basis von Cosine Similarity.  

- **`sdm_utils.py`**  
  Enthält Hilfsfunktionen:  
  - **hamming_distance(a, b)**: Hamming-Distanz zweier binärer/bipolarer Vektoren.  
  - **random_bipolar_vector(size)**: Zufällige Vektoren in {+1, -1}.  
  - **binarize_sign(...)**, **to_bipolar(...)**, **to_binary(...)**: Umwandlungsfunktionen von Werten oder Vektoren.  
  - **cosine_similarity(...)**: Klassischer Kosinus zur Ähnlichkeitsbestimmung.  
  - **approximate_knn(...)**: (Platzhalter) Exakte KNN-Suche mit Hamming-Distanz (in realen Systemen könnte man hier Annoy, FAISS usw. einbinden).  

- **`tests/`**  
  Enthält Testskripte (z. B. `test_sdm_module.py`, `test_vsa_module.py`), um die Implementierungen automatisch zu überprüfen.  
  - **`test_sdm_module.py`** könnte beispielsweise prüfen, ob die Schreib-/Lese-Funktionen von `SparseDistributedMemory` korrekt arbeiten.  
  - **`test_vsa_module.py`** testet `VSA`, Binding-/Superposition-Logik und die Cleanup-Funktion.

---

## Beispiel-Nutzung

### 1. Sparse Distributed Memory

```python
from sdm import SparseDistributedMemory
import numpy as np

# Erzeuge ein SDM mit 1000-dimensionalen Adressen und 5000 Hard Locations
sdm = SparseDistributedMemory(address_size=1000, memory_size=5000, activation_percent=0.1)

# Erzeuge einen bipolaren Eingabevektor
input_vec = np.random.choice([1, -1], size=1000)

# Schreiben
sdm.write_memory(input_vec)

# Lesen
retrieved = sdm.read_memory(input_vec)
similarity = np.mean(input_vec == retrieved)
print(f"Retrieval Similarity: {similarity * 100:.2f}%")
```

### 2. Vector-Symbolic Architecture

```python
from sdm import VSA

# Instanz mit 1000-dimensionalen Hypervektoren
vsa = VSA(dimension=1000)

# Erzeuge zufällige Vektoren für "dog", "cat", "chases"
dog = vsa.random_hypervector()
cat = vsa.random_hypervector()
chases = vsa.random_hypervector()

# Binding: dog_chases
dog_chases = vsa.bind(dog, chases)
vsa.store_memory("dog_chases", dog_chases)

# Abfrage
query = vsa.bind(dog, chases)
retrieved_label, sim = vsa.retrieve_memory(query)
print(f"Retrieved Label: {retrieved_label}, Similarity: {sim:.4f}")
```

### 3. SDM-Utils

```python
from sdm import (
    random_bipolar_vector,
    hamming_distance,
    binarize_sign,
    cosine_similarity
)

a = random_bipolar_vector(8)
b = random_bipolar_vector(8)

dist = hamming_distance(a, b)
cos_sim = cosine_similarity(a, b)
print("Hamming Dist:", dist, "Cosine Sim:", cos_sim)
```

---

## Installation & Tests

1. **Installation** (lokal):
   ```bash
   cd sdm
   pip install -e .
   ```
   (Falls du ein Setup-Skript oder Poetry verwendest.)

2. **Unittests ausführen**:
   ```bash
   pytest tests
   ```

---

## Referenzen & Hintergrund

- **Pentti Kanerva**: *Sparse Distributed Memory* (1988), MIT Press.  
- **John S. Anderson**: *Vector Symbolic Architectures* in *The Handbook of Brain Theory and Neural Networks*.  
- Moderne Arbeiten zu **Hyperdimensional Computing** (z. B. HD Computing in NeurIPS, IEEE etc.).

**Viel Erfolg** bei der Nutzung von `sdm/` für assoziative Speicherverfahren und hochdimensionale Symbolrepräsentation! 
```
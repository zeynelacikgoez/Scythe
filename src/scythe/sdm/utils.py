# sdm_utils.py

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

def hamming_distance(vector_a: np.ndarray, vector_b: np.ndarray) -> int:
    """
    Berechnet die Hamming-Distanz zweier binärer Vektoren in {+1, -1}
    oder {0, 1} (je nach Nutzung).

    Parameter:
    ----------
    vector_a, vector_b : np.ndarray
        Gleich lange Arrays, binär bzw. bipolar.

    Rückgabe:
    ---------
    int : Anzahl der Positionen, in denen vector_a != vector_b
    """
    if vector_a.shape != vector_b.shape:
        raise ValueError("hamming_distance: Vektoren haben unterschiedliche Shapes.")
    return np.sum(vector_a != vector_b)


def random_bipolar_vector(size: int) -> np.ndarray:
    """
    Erzeugt einen Zufallsvektor der Länge 'size',
    mit Elementen in {+1, -1}.

    Parameter:
    ----------
    size : int
        Länge des zu erzeugenden Vektors.

    Rückgabe:
    ---------
    np.ndarray : shape=(size,), Elemente in {+1, -1}
    """
    return np.random.choice([1, -1], size=size)


def binarize_sign(array: np.ndarray) -> np.ndarray:
    """
    Nimmt ein Array mit reellen Werten (positiv, negativ, evtl. 0)
    und mappt es elementweise auf {+1, -1}.
    Dabei wird sign(0)=0 auf +1 abgebildet.

    Parameter:
    ----------
    array : np.ndarray
        Array beliebiger Form mit reellen Werten

    Rückgabe:
    ---------
    np.ndarray : Selbe Shape, nur mit +1 / -1
    """
    sign_vec = np.sign(array)  # +, -, 0
    sign_vec[sign_vec == 0] = 1
    return sign_vec


def to_bipolar(array: np.ndarray) -> np.ndarray:
    """
    Konvertiert ein Array mit Werten in {0, 1} in {+1, -1}.

    Parameter:
    ----------
    array : np.ndarray
        Array (z. B. shape=(n,)) oder beliebiger Shape,
        dessen Elemente in {0, 1} liegen.

    Rückgabe:
    ---------
    np.ndarray : Selbe Shape, aber Elemente in {+1, -1}.
    """
    return np.where(array == 1, 1, -1)


def to_binary(array: np.ndarray) -> np.ndarray:
    """
    Konvertiert ein Array mit Werten in {+1, -1} in {0, 1}.

    Parameter:
    ----------
    array : np.ndarray
        Array mit Werten in {+1, -1}.

    Rückgabe:
    ---------
    np.ndarray : Selbe Shape, aber Werte in {0, 1}.
    """
    return np.where(array == 1, 1, 0)


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    Berechnet den Cosine Similarity zwischen zwei Vektoren (gleiche Länge).

    Parameter:
    ----------
    vector_a, vector_b : np.ndarray
        Reelle oder bipolare Vektoren der gleichen Länge.

    Rückgabe:
    ---------
    float : cos(theta) = (a·b)/(||a||*||b||)
    """
    if vector_a.shape != vector_b.shape:
        raise ValueError("cosine_similarity: Vektoren haben unterschiedliche Shapes.")

    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


# --------------------------------------------------------------
# FAISS-Unterstützung (optional), um approximate_knn zu beschleunigen
# --------------------------------------------------------------
FAISS_INDEX_CACHE = None
FAISS_DATA_CACHE = None

def build_faiss_index(vectors: np.ndarray, metric: str = "hamming") -> faiss.Index:
    """
    Erzeugt einen FAISS-Index für 'vectors'.
    metric: "hamming" oder "cosine" oder "l2"

    Für "hamming" würde man typically binäre Indizes brauchen,
    aber hier realisieren wir einen Workaround mit "L2" + Tricks.

    Für "cosine" => erst normalisieren => inner product

    ACHTUNG: "Hamming" an sich kann man nur approximate simulieren
    oder Bits packen. Hier nur als Demo.
    """
    d = vectors.shape[1]
    # Standard: L2-Suche => IndexFlatL2
    # Cosine => IndexFlatIP + normalisierte Vektoren
    if metric == "cosine":
        index = faiss.IndexFlatIP(d)  # inner product
    else:
        index = faiss.IndexFlatL2(d)

    # Eventuell GPU:
    # index = faiss.index_cpu_to_all_gpus(index)

    # Normierung bei Cosine
    data_index = vectors.copy()
    if metric == "cosine":
        norms = np.linalg.norm(data_index, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        data_index = data_index / norms

    index.add(data_index.astype('float32'))
    return index


def approximate_knn(
    query_vector: np.ndarray,
    addresses: np.ndarray,
    k: int = 10,
    metric: str = "hamming",
    use_faiss: bool = True
) -> np.ndarray:
    """
    *Platzhalter*-Funktion für eine Approximate-KNN-Suche (ANN).
    Wenn use_faiss=True und FAISS verfügbar, wird FAISS für
    die Distanzsuche genutzt (entweder L2 oder IP/Cosine).
    Ansonsten wird ein simpler Hamming-Loop (exakt) genutzt.

    Parameter:
    ----------
    query_vector : np.ndarray
        shape=(d,). Vektor in {+1, -1} oder reell, je nach usage.
    addresses : np.ndarray
        shape=(n, d). Alle Adressen / Referenzvektoren.
    k : int
        Anzahl der nächsten Nachbarn.
    metric : str
        "hamming", "cosine" oder "l2".
    use_faiss : bool
        Falls True, nutze FAISS (sofern installiert). Sonst Fallback.

    Rückgabe:
    ---------
    np.ndarray : Indizes der k nächsten Nachbarn nach Distanz/Ähnlichkeit.
    """
    n, d = addresses.shape
    if d != query_vector.shape[0]:
        raise ValueError("Dimension mismatch: addresses und query_vector passen nicht.")

    # 1) Wenn wir FAISS verwenden wollen und es verfügbar ist
    #    => index cachen
    if use_faiss and FAISS_AVAILABLE:
        global FAISS_INDEX_CACHE
        global FAISS_DATA_CACHE

        # Prüfe, ob der Cache noch passt
        if FAISS_DATA_CACHE is not None and np.array_equal(FAISS_DATA_CACHE, addresses):
            index = FAISS_INDEX_CACHE
        else:
            # Baue neu
            index = build_faiss_index(addresses, metric=("cosine" if metric == "cosine" else "l2"))
            FAISS_INDEX_CACHE = index
            FAISS_DATA_CACHE = addresses.copy()

        query = query_vector.astype('float32').reshape(1, -1)

        # Falls Cosine => normalisieren
        if metric == "cosine":
            norm_q = np.linalg.norm(query)
            if norm_q < 1e-9:
                norm_q = 1e-9
            query = query / norm_q

        # FAISS-Suche: L2 oder IP
        # IP ~ negative L2 => Not exact "hamming"
        D, I = index.search(query, k)
        # I => shape (1, k)
        return I[0]

    else:
        # 2) Fallback: exakte Hamming-Schleife
        # "metric='hamming'" => wir gehen von bipolaren Vektoren aus {+1, -1}
        # "cosine" => exakte Cosine
        # "l2" => exakte L2
        if metric == "cosine":
            # exakte Cosine
            query_norm = np.linalg.norm(query_vector)
            if query_norm < 1e-9:
                # triviale fallback => jeder Distanz/Sim
                query_norm = 1e-9
            sims = []
            for idx, row in enumerate(addresses):
                row_norm = np.linalg.norm(row)
                if row_norm < 1e-9:
                    row_norm = 1e-9
                dot = np.dot(query_vector, row)
                sim = dot / (query_norm * row_norm)
                sims.append((sim, idx))
            # sort by sim descending => highest first
            sims.sort(key=lambda x: x[0], reverse=True)
            closest_indices = [pair[1] for pair in sims[:k]]
            return np.array(closest_indices)

        elif metric == "l2":
            # exakte L2
            distances = []
            for idx, row in enumerate(addresses):
                dist = np.linalg.norm(query_vector - row)
                distances.append((dist, idx))
            distances.sort(key=lambda x: x[0])
            closest_indices = [pair[1] for pair in distances[:k]]
            return np.array(closest_indices)

        else:
            # "hamming"
            distances = []
            for idx, row in enumerate(addresses):
                dist = hamming_distance(query_vector, row)
                distances.append((dist, idx))
            distances.sort(key=lambda x: x[0])
            closest_indices = [pair[1] for pair in distances[:k]]
            return np.array(closest_indices)


# --------------------------------------------------------------------------
# Wenn man dieses Skript direkt ausführt, kleiner Demo-Test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Zufälliger Bipolar-Vektor
    vec_a = random_bipolar_vector(8)
    vec_b = random_bipolar_vector(8)

    # 2) Hamming-Distanz
    dist = hamming_distance(vec_a, vec_b)
    print("vec_a:", vec_a)
    print("vec_b:", vec_b)
    print("Hamming Distance:", dist)

    # 3) Sign-Binarisierung
    some_values = np.array([3.2, -0.5, 0.0, 2.0])
    sign_bin = binarize_sign(some_values)
    print("some_values:", some_values)
    print("binarize_sign:", sign_bin)

    # 4) to_bipolar / to_binary
    bin_array = np.array([0, 1, 1, 0])
    bipolar = to_bipolar(bin_array)
    print("bin_array:", bin_array)
    print("bipolar:", bipolar)
    print("back to binary:", to_binary(bipolar))

    # 5) Cosine-Sim
    cos_sim = cosine_similarity(vec_a, vec_b)
    print("Cosine Similarity:", cos_sim)

    # 6) Approximate KNN
    addresses = np.random.choice([1, -1], size=(10, 8))  # 10 Adressen
    query = random_bipolar_vector(8)
    knn_indices = approximate_knn(query, addresses, k=3, metric="hamming", use_faiss=False)
    print("KNN Indices (fallback/hamming):", knn_indices)

    # Falls du FAISS installiert hast, teste:
    if FAISS_AVAILABLE:
        # Example with "cosine"
        float_addresses = np.random.randn(10, 8).astype('float32')
        float_query = np.random.randn(8).astype('float32')
        knn_indices_faiss = approximate_knn(float_query, float_addresses, k=3, metric="cosine", use_faiss=True)
        print("KNN Indices (FAISS/cosine):", knn_indices_faiss)
    else:
        print("[sdm_utils] FAISS not installed, skipping FAISS demo.")

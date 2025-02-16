# sdm_module.py

import numpy as np
from typing import Optional
from sdm.sdm_utils import hamming_distance, approximate_knn

try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class SparseDistributedMemory:
    """
    SparseDistributedMemory (SDM) nach Kanerva (1988).
    --------------------------------------------------
    SDM speichert Vektoren in einem hochdimensionalen Raum. Jede 'Address'
    in self.addresses ist ein zufällig gewählter binärer Vektor (+1 oder -1).
    Beim Schreiben wird das Eingabe-Muster auf die k nächsten (nach Hamming-Distanz
    oder Approx-KNN) Hard Locations verteilt, beim Lesen ebenso.

    NEU:
      - Optionale PCA-Unterstützung (use_pca=True), um Input-Vektoren
        vor dem Schreiben zu transformieren (falls deine Original-Embeddings
        nicht die gleiche Dimension wie 'address_size' haben).
      - Keine automatische Memory-Reset bei jedem Pipeline-Aufruf.
      - Wahlweise exakte Hamming-Schleife oder approximate_knn().

    Parameter:
    ----------
    address_size : int
        Länge jedes Adressvektors (z. B. 256).
    memory_size : int
        Anzahl der Hard Locations (z. B. 1000).
    activation_percent : float
        Prozentsatz der Hard Locations, die beim Schreiben/Lesen aktiviert werden (k= activation_percent * memory_size).
    init_mode : str
        "random" => zufällig +1/-1 Adressen,
        "zeros"  => experimentell, alle +1 als Adresse.
    use_pca : bool
        Falls True, wird intern PCA benutzt.
    pca_components : int
        Anzahl Hauptkomponenten, die wir beibehalten (typisch = address_size).
    use_approx_knn : bool
        Falls True, versuchen wir in read/write via approximate_knn anstatt manueller Distanzschleife.
        In sdm_utils.py wird FAISS oder Fallback verwendet.
    """

    def __init__(
        self,
        address_size: int = 256,
        memory_size: int = 1000,
        activation_percent: float = 0.1,
        init_mode: str = "random",
        use_pca: bool = False,
        pca_components: int = 256,
        use_approx_knn: bool = False
    ):
        self.address_size = address_size
        self.memory_size = memory_size
        self.activation_percent = activation_percent
        self.use_approx_knn = use_approx_knn

        # Hard Locations (Adressen)
        if init_mode == "random":
            self.addresses = np.random.choice([1, -1], size=(memory_size, address_size))
        elif init_mode == "zeros":
            self.addresses = np.ones((memory_size, address_size), dtype=int)
        else:
            raise ValueError(f"Unbekanntes init_mode: {init_mode}")

        # Speicher-Matrix: Akkumuliert Summen für jede Hard Location
        self.memory = np.zeros((memory_size, address_size), dtype=float)

        # PCA-Setup
        self.use_pca = use_pca and SKLEARN_AVAILABLE
        self.pca_components = pca_components
        self._pca_fitted = False

        if self.use_pca:
            # Falls scikit-learn nicht installiert, wirft es an anderer Stelle ImportError
            self._pca_model = PCA(n_components=self.pca_components)
        else:
            self._pca_model = None

        if use_pca and not SKLEARN_AVAILABLE:
            raise ImportError("Sklearn ist nicht installiert, aber 'use_pca=True' gesetzt.")

    def fit_pca(self, data_vectors: np.ndarray):
        """
        Optionale Methode, um PCA auf einem Satz von data_vectors (z. B. Embeddings) zu fitten,
        bevor wir in den SDM schreiben.

        Parameter:
        ----------
        data_vectors : np.ndarray, shape=(N, original_dim)
        """
        if not self.use_pca:
            print("[SDM] use_pca=False => fit_pca wird ignoriert.")
            return

        self._pca_model.fit(data_vectors)
        self._pca_fitted = True
        print("[SDM] PCA ist gefittet. Original-Dim:", data_vectors.shape[1], "->", self.pca_components)

    def transform_pca(self, vector: np.ndarray) -> np.ndarray:
        """
        Wendet den gefitteten PCA-Transform auf 'vector' an.
        Annahme: vector shape = (original_dim,).

        Rückgabe:
        ---------
        np.ndarray shape = (pca_components,)
        """
        if not self.use_pca or not self._pca_fitted:
            # Keine PCA oder nicht gefittet => gib vector unverändert zurück
            return vector

        reshaped = vector.reshape(1, -1)
        pc = self._pca_model.transform(reshaped)
        return pc.flatten()

    def write_memory(self, input_vector: np.ndarray) -> None:
        """
        Schreibt input_vector (binär: +1/-1 oder reell) in die k
        nächsten Hard Locations und akkumuliert dort die Werte.

        1) Optional PCA transform
        2) Evtl. binarisieren in {+1, -1}, je nachdem, was wir wollen
        3) Distanz oder approximate_knn => k Indizes
        4) memory[idx] += input_vector
        """
        if input_vector.shape[0] == 0:
            print("[SDM] WARN: write_memory -> input_vector hat length=0")
            return

        # Optional PCA
        if self.use_pca:
            input_vector = self.transform_pca(input_vector)

        # Sicherheitscheck
        if input_vector.shape[0] != self.address_size:
            raise ValueError(
                f"[SDM] input_vector-Dimension {input_vector.shape[0]} != address_size {self.address_size}"
            )

        # k = anzahl aktivierter Hard Locations
        k = int(self.activation_percent * self.memory_size)
        k = max(1, k)  # mind. 1

        # Indizes der k nächsten
        closest_indices = None
        if self.use_approx_knn:
            # approximate_knn => wir wählen "hamming"
            # oder "cosine" / "l2" - je nach Belieben
            # da wir z. B. bipolare Adressen in self.addresses haben,
            # "hamming" am naheliegendsten
            closest_indices = self._knn_indices(input_vector, k, metric="hamming")
        else:
            # Exakte Hamming-Schleife
            distances = np.array([
                hamming_distance(input_vector, addr)
                for addr in self.addresses
            ])
            # sort asc
            closest_indices = np.argsort(distances)[:k]

        # Akkumuliere in memory
        for idx in closest_indices:
            self.memory[idx] += input_vector

    def read_memory(self, query_vector: np.ndarray) -> np.ndarray:
        """
        Liest aus dem SDM, indem die k nächsten Adressen zum query_vector gesucht,
        deren gespeicherte Inhalte summiert und per sign(...) binarisiert werden.

        Parameter:
        ----------
        query_vector : np.ndarray, shape=(address_size,)

        Rückgabe:
        ---------
        np.ndarray : Rekonstruierter binärer Vektor {+1, -1}.
        """
        if query_vector.shape[0] == 0:
            print("[SDM] WARN: read_memory -> query_vector length=0 => return +1?")
            return np.ones((self.address_size,), dtype=np.float32)

        # optional PCA
        if self.use_pca:
            query_vector = self.transform_pca(query_vector)

        if query_vector.shape[0] != self.address_size:
            raise ValueError(
                f"[SDM] query_vector-Dimension {query_vector.shape[0]} != address_size {self.address_size}"
            )

        k = int(self.activation_percent * self.memory_size)
        k = max(1, k)

        # k nächste Adressen
        if self.use_approx_knn:
            closest_indices = self._knn_indices(query_vector, k, metric="hamming")
        else:
            # exakte Hamming
            distances = np.array([
                hamming_distance(query_vector, addr)
                for addr in self.addresses
            ])
            closest_indices = np.argsort(distances)[:k]

        # Summiere den Inhalt
        retrieved_sum = np.sum(self.memory[closest_indices], axis=0)
        # Binarisierung
        output_vector = np.sign(retrieved_sum)
        output_vector[output_vector == 0] = 1
        return output_vector

    def _knn_indices(self, vector: np.ndarray, k: int, metric: str = "hamming") -> np.ndarray:
        """
        Hilfsfunktion, die approximate_knn aus sdm_utils nutzt,
        um die Indizes der k nächsten Nachbarn zurückzugeben.
        """
        # addresses shape=(memory_size, address_size)
        # vector shape=(address_size,)
        indices = approximate_knn(
            query_vector=vector,
            addresses=self.addresses,
            k=k,
            metric=metric,
            use_faiss=False  # oder True, wenn du FAISS für Bipolar-Hamming willst
        )
        return indices

    def reset_memory(self) -> None:
        """
        Setzt die Speicher-Matrix zurück.
        """
        self.memory = np.zeros((self.memory_size, self.address_size), dtype=float)

    def debug_info(self, n: int = 5) -> None:
        """
        Gibt Debug-Information zu den ersten n Hard Locations aus.
        """
        print("== SDM DEBUG INFO ==")
        for i in range(min(n, self.memory_size)):
            print(f"Hard Location {i} - Address: {self.addresses[i]}")
            print(f"                Memory: {self.memory[i]}")
            print("")


if __name__ == "__main__":
    # Beispiel:
    sdm = SparseDistributedMemory(
        address_size=16,
        memory_size=32,
        activation_percent=0.2,
        init_mode="random",
        use_pca=False,
        use_approx_knn=False
    )

    # Schreibe und Lese
    input_vec = np.random.choice([1, -1], size=16)
    sdm.write_memory(input_vec)
    out_vec = sdm.read_memory(input_vec)
    sim = np.mean(input_vec == out_vec)
    print(f"[sdm_module main] Similarity input vs out={sim*100:.2f}%")

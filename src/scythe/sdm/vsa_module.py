# vsa_module.py

import numpy as np

class VSA:
    """
    Vector Symbolic Architecture (VSA):
    -----------------------------------
    - Repräsentation und Manipulation von Informationen in hochdimensionalen, 
      meist binären (+1/-1 oder 0/1) Hypervektoren.
    - "Binding" (Elementweise Multiplikation) stellt eine 'Verknüpfung' zweier 
      Konzepte dar, die zu einem neuen, disjunkten Vektor führt.
    - "Superposition" (Elementweise Addition + Normalisierung) bündelt mehrere 
      Vektoren in einem gemeinsamen Vektor, der allen ähnlich ist.
    - "Cosine Similarity" zur Abruf- oder Erkennungs-Logik ("Clean-Up Memory").

    Parameter:
    ----------
    dimension : int
        Dimension der Hypervektoren (z. B. 1000).

    Attribute:
    ----------
    memory : dict
        Einfaches Key-Value-Store, in dem man Vektoren ablegen kann (label -> hypervector).
        Dient als 'Cleanup Memory'.
    """

    def __init__(self, dimension: int = 1000):
        self.dimension = dimension
        self.memory = {}

    def random_hypervector(self) -> np.ndarray:
        """
        Erzeugt einen zufälligen bipolar-Hypervektor (Größe = self.dimension),
        Elemente in {+1, -1}.

        Rückgabe:
        ---------
        np.ndarray : shape = (dimension,), Werte = {+1, -1}
        """
        return np.random.choice([1, -1], size=self.dimension)

    def bind(self, vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
        """
        Implementiert das 'Binding' (⊗) in VSA. 
        Hier einfach: elementweise Multiplikation => Resultat unähnlich zu v1, v2.

        Parameter:
        ----------
        vector1, vector2 : np.ndarray, shape=(dimension,), Werte in {+1, -1}

        Rückgabe:
        ---------
        np.ndarray : shape=(dimension,), Bindungsergebnis
        """
        if vector1.shape != vector2.shape:
            raise ValueError("Binding: Vektoren haben unterschiedliche Shapes.")
        return vector1 * vector2

    def superpose(self, *vectors: np.ndarray) -> np.ndarray:
        """
        Implementiert 'Superposition' (⊕): Elementweise Summe + Normalisieren auf {+1, -1}.

        Parameter:
        ----------
        *vectors : Beliebig viele Hypervektoren (gleiche Dimension).

        Rückgabe:
        ---------
        np.ndarray : shape=(dimension,), Werte in {+1, -1}
        """
        if len(vectors) == 0:
            raise ValueError("Superposition: Keine Eingabevektoren übergeben.")

        summed = np.sum(np.array(vectors), axis=0)  # elementwise sum
        # Normalisierung auf +1 / -1
        sign_vec = np.sign(summed)
        # Falls sign(0) = 0 => ersetze durch +1
        sign_vec[sign_vec == 0] = 1
        return sign_vec

    def store_memory(self, label: str, hypervector: np.ndarray) -> None:
        """
        Speichert einen gelabelten Hypervektor im Memory (dictionary).
        So kann man später per retrieve_memory(label) o. ä. wieder darauf zugreifen.

        Parameter:
        ----------
        label : str
            Name oder Schlüssel für den gespeicherten Vektor.
        hypervector : np.ndarray
            Ein Hypervektor (shape = (dimension,))
        """
        if hypervector.shape[0] != self.dimension:
            raise ValueError("store_memory: Hypervector hat falsche Dimension.")
        self.memory[label] = hypervector

    def retrieve_memory(self, query_vector: np.ndarray):
        """
        Sucht den 'nächsten' gespeicherten Vektor (per Cosine Similarity) im Memory.

        Parameter:
        ----------
        query_vector : np.ndarray, shape=(dimension,)

        Rückgabe:
        ---------
        (best_label, best_similarity) : (str, float)
            Label des ähnlichsten Vektors und die zugehörige Cosine-Similarity
        """
        if not self.memory:
            return None, -1.0  # Kein Memory

        best_label = None
        best_similarity = -np.inf

        # Cosine Similarity
        q_norm = np.linalg.norm(query_vector)
        for label, stored_vec in self.memory.items():
            dot = np.dot(query_vector, stored_vec)
            s_norm = np.linalg.norm(stored_vec)
            sim = dot / (q_norm * s_norm) if (q_norm > 0 and s_norm > 0) else -1.0
            if sim > best_similarity:
                best_similarity = sim
                best_label = label

        return best_label, best_similarity

    def debug_memory(self) -> None:
        """
        Listet alle gespeicherten Labels und Vektoren (evtl. gekürzt) auf.
        """
        print("=== VSA Memory Debug ===")
        for label, vec in self.memory.items():
            print(f"Label: {label}, Vec[:10]={vec[:10]}..., Norm={np.linalg.norm(vec):.2f}")
        print("")

# Testbeispiel
if __name__ == "__main__":
    vsa = VSA(dimension=12)  # kleine Dimension zum Demonstrieren

    # Erzeuge zufällige Vektoren
    dog_vec = vsa.random_hypervector()
    cat_vec = vsa.random_hypervector()
    chases_vec = vsa.random_hypervector()

    # Binding: dog_chases
    dog_chases_vec = vsa.bind(dog_vec, chases_vec)

    # Speichern
    vsa.store_memory("dog_chases", dog_chases_vec)

    # Abfrage (dog * chases)
    query = vsa.bind(dog_vec, chases_vec)
    retrieved_label, similarity = vsa.retrieve_memory(query)
    print(f"Retrieved: {retrieved_label}, Sim={similarity:.4f}")

    # Superposition (z. B. dog + cat)
    dog_cat_vec = vsa.superpose(dog_vec, cat_vec)
    print("Superposed dog+cat:", dog_cat_vec)

    # Memory-Debug
    vsa.debug_memory()

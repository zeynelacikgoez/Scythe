import numpy as np
import logging

# Aus deinem SDM-Paket:
from scythe.sdm import SparseDistributedMemory, to_bipolar

logger = logging.getLogger(__name__)

class HyperMemoryInterface:
    """
    Ein einfaches Interface für 'HyperMemory'-Manager,
    das Speichern und Abfragen von States (als Vektoren) erlaubt.
    """

    def store_state(self, state_vector: np.ndarray) -> None:
        """
        Speichert (bzw. schreibt) einen Zustand in den Hyper-Speicher.
        """
        raise NotImplementedError

    def retrieve_nearest_state(self, query_vector: np.ndarray) -> np.ndarray:
        """
        Liest den ähnlichsten (oder k-nächsten) Zustand aus dem Hyper-Speicher.
        Kann None zurückgeben, falls leer oder unbrauchbar.
        """
        raise NotImplementedError


class SdmMemoryManager(HyperMemoryInterface):
    """
    Beispiel-Implementierung, die unser SparseDistributedMemory verwendet.
    """

    def __init__(self,
                 address_size: int = 64,
                 memory_size: int = 500,
                 activation_percent: float = 0.1,
                 init_mode: str = "random"):
        """
        Erzeugt ein SDM-Objekt mit den angegebenen Parametern.
        """
        self.sdm = SparseDistributedMemory(
            address_size=address_size,
            memory_size=memory_size,
            activation_percent=activation_percent,
            init_mode=init_mode
        )
        logger.info(f"[SdmMemoryManager] SDM init: addr_size={address_size}, mem_size={memory_size}, act%={activation_percent}")

    def store_state(self, state_vector: np.ndarray) -> None:
        """
        Speichert state_vector im SDM.
        Dabei wandeln wir (falls nötig) reelle Werte in bipolare (+1/-1) um.
        """
        if state_vector.ndim != 1:
            raise ValueError("[SdmMemoryManager.store_state] state_vector muss 1D sein.")
        # z.B. in +1/-1 wandeln (Heuristik: x>0 => +1, sonst -1)
        bipolar_vec = to_bipolar(state_vector > 0)
        self.sdm.write_memory(bipolar_vec)

    def retrieve_nearest_state(self, query_vector: np.ndarray) -> np.ndarray:
        """
        Liest den 'nächsten' Zustand aus dem SDM. Hier wird (zur Vereinfachung)
        read_memory() genutzt, was den 'gemittelten' Vektor zurückgibt.

        Rückgabe: bipolares +1/-1 Vektor
        """
        if query_vector.ndim != 1:
            raise ValueError("[SdmMemoryManager.retrieve_nearest_state] query_vector muss 1D sein.")
        bipolar_query = to_bipolar(query_vector > 0)

        retrieved = self.sdm.read_memory(bipolar_query)
        return retrieved


def scenario_to_vector(scenario: dict) -> np.ndarray:
    """
    Beispiel-Funktion, die ein Szenario (z. B. demands, max_resource, factories)
    in einen reellen Vektor kodiert, um es im SDM/VSA ablegbar zu machen.

    Du kannst das anpassen, je nachdem was du an Info speichern willst.
    """
    # Minimalbeispiel:
    # demands => sum(demands), max_resource => scalar,
    # factories => sum(max_production)
    # => array([ sum_demands, max_resource, sum_of_max_prod ])
    sum_demands = 0.0
    if "demands" in scenario and scenario["demands"]:
        sum_demands = sum(scenario["demands"])

    max_res = float(scenario.get("max_resource", 0.0))

    sum_fact = 0.0
    for f in scenario.get("factories", []):
        sum_fact += f.get("max_production", 0)

    vec = np.array([sum_demands, max_res, sum_fact], dtype=float)
    return vec

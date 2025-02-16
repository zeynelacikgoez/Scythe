# tests/__init__.py

"""
Das Verzeichnis tests/ enthält Unittests für die SDM/VSA-Module:
- test_sdm_module.py : Testet SparseDistributedMemory-Funktionalität
- test_vsa_module.py : Testet VSA-Klasse und Operationen (Binding, Superposition, Memory)

Hier können auch gemeinsame Fixtures oder Konfigurationen 
für Pytest definiert werden.
"""

import pytest

@pytest.fixture(scope="session")
def global_test_seed():
    """
    Optional: Setzt einen globalen Zufalls-Seed für alle Tests.
    Man kann diesen Fixture in anderen Test-Dateien aufrufen, um Reproduzierbarkeit zu erhöhen.
    
    Usage in test files:
    
    def test_something(global_test_seed):
        # global_test_seed wird bei Session-Start ausgeführt
        pass
    """
    import random
    import numpy as np

    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
    print(f"[test/__init__.py] Global seed set to {seed}")

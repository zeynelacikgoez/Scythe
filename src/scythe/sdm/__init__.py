# sdm/__init__.py

"""
Das sdm-Paket:
--------------
Bietet Implementierungen und Hilfsfunktionen für:
 - Sparse Distributed Memory (Kanervas SDM) in sdm_module.py
 - Vector-Symbolic Architectures (VSA) in vsa_module.py
 - Gemeinsame Hilfsfunktionen (Hamming-Distanz, Binarisierung etc.) in sdm_utils.py

Zusätzliche Tests liegen im Unterordner tests/.
"""

from .sdm_module import SparseDistributedMemory
from .vsa_module import VSA
from .sdm_utils import (
    hamming_distance,
    random_bipolar_vector,
    binarize_sign,
    to_bipolar,
    to_binary,
    cosine_similarity,
    approximate_knn
)

__all__ = [
    "SparseDistributedMemory",
    "VSA",
    "hamming_distance",
    "random_bipolar_vector",
    "binarize_sign",
    "to_bipolar",
    "to_binary",
    "cosine_similarity",
    "approximate_knn"
]

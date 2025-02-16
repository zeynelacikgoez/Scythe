# test_vsa_module.py

import pytest
import numpy as np
from sdm import VSA

@pytest.fixture
def small_vsa():
    """
    Pytest Fixture, die ein VSA-Objekt mit kleiner dimension für Tests erstellt.
    """
    vsa = VSA(dimension=12)  # kleine Dimension zum Schnelltesten
    return vsa

def test_vsa_random_hv(small_vsa):
    """
    Testet, ob random_hypervector() die richtige Shape und Werte in {+1, -1} erzeugt.
    """
    vsa = small_vsa
    hv = vsa.random_hypervector()
    assert hv.shape == (12,), "Hypervector hat falsche Dimension"
    # Überprüfen, ob alle Werte +1 oder -1 sind
    unique_vals = np.unique(hv)
    assert set(unique_vals).issubset({-1, 1}), f"Unexpected values im Hypervector: {unique_vals}"

def test_vsa_bind(small_vsa):
    """
    Testet die Binding-Operation.
    """
    vsa = small_vsa
    vec_a = vsa.random_hypervector()
    vec_b = vsa.random_hypervector()

    bound = vsa.bind(vec_a, vec_b)
    assert bound.shape == (12,), "Gebundener Vektor hat falsche Dimension."

    # Binding in {+1, -1} => elementwise product => Prüfen, ob +1/-1
    unique_vals = np.unique(bound)
    assert set(unique_vals).issubset({-1, 1}), "Gebundener Vektor enthält Werte außerhalb {+1, -1}"

def test_vsa_superpose(small_vsa):
    """
    Testet die Superposition (⊕) Operation.
    """
    vsa = small_vsa
    vec_a = vsa.random_hypervector()
    vec_b = vsa.random_hypervector()
    vec_c = vsa.random_hypervector()

    superposed = vsa.superpose(vec_a, vec_b, vec_c)
    assert superposed.shape == (12,), "Superposed Vektor hat falsche Dimension."

    # Superpose sollte ebenfalls in {+1, -1} resultieren
    unique_vals = np.unique(superposed)
    assert set(unique_vals).issubset({-1, 1}), "Superposition ergab Werte außerhalb {+1, -1}"

def test_vsa_memory_store_retrieve(small_vsa):
    """
    Testet das Speichern (store_memory) und Abfragen (retrieve_memory) von Hypervektoren.
    """
    vsa = small_vsa
    dog = vsa.random_hypervector()
    cat = vsa.random_hypervector()
    chases = vsa.random_hypervector()

    # Binding
    dog_chases = vsa.bind(dog, chases)
    cat_chases = vsa.bind(cat, chases)

    # Memory: Speichern
    vsa.store_memory("dog_chases", dog_chases)
    vsa.store_memory("cat_chases", cat_chases)

    # Abfrage dog ⊗ chases
    query_dog_chases = vsa.bind(dog, chases)
    retrieved_label_dc, sim_dc = vsa.retrieve_memory(query_dog_chases)

    assert retrieved_label_dc == "dog_chases", f"Falscher Label abgerufen: {retrieved_label_dc}, Sim={sim_dc:.4f}"
    assert sim_dc > 0.95, f"Erwartete hohe Similarity, aber sim={sim_dc:.4f}"

    # Abfrage cat ⊗ chases
    query_cat_chases = vsa.bind(cat, chases)
    retrieved_label_cc, sim_cc = vsa.retrieve_memory(query_cat_chases)

    assert retrieved_label_cc == "cat_chases", f"Falscher Label abgerufen: {retrieved_label_cc}, Sim={sim_cc:.4f}"
    assert sim_cc > 0.95, f"Erwartete hohe Similarity, aber sim={sim_cc:.4f}"

def test_vsa_memory_empty_retrieve(small_vsa):
    """
    Testet den Fall, wenn man retrieve_memory ohne gespeicherte Einträge aufruft.
    Sollte (None, -1.0) liefern.
    """
    vsa = small_vsa
    dummy = vsa.random_hypervector()
    label, sim = vsa.retrieve_memory(dummy)
    assert label is None, f"Erwartet 'None' wenn memory leer, bekam Label={label}"
    assert sim == -1.0, f"Erwartet sim=-1.0 wenn memory leer, bekam={sim:.4f}"

# test_sdm_module.py

import pytest
import numpy as np
from sdm import SparseDistributedMemory

@pytest.fixture
def default_sdm():
    """
    Pytest-Fixture: Erstellt ein SDM-Objekt mit Standardparametern.
    Wird in mehreren Tests wiederverwendet.
    """
    sdm = SparseDistributedMemory(
        address_size=16,   # kleine Dimension für Schnelltests
        memory_size=32,
        activation_percent=0.2,
        init_mode="random"
    )
    return sdm

def test_sdm_initialization(default_sdm):
    """
    Testet, ob das SDM korrekt initialisiert wurde.
    """
    sdm = default_sdm
    # Prüfe Shape von addresses und memory
    assert sdm.addresses.shape == (32, 16), "Addresses hat falsche Shape."
    assert sdm.memory.shape == (32, 16), "Memory hat falsche Shape."
    assert 0 <= sdm.activation_percent <= 1.0, "activation_percent liegt nicht im [0,1]-Bereich."

def test_sdm_write_read_single(default_sdm):
    """
    Schreibt einen einzelnen Vektor ins SDM und prüft,
    ob das Lesen das gleiche Muster liefert.
    """
    sdm = default_sdm
    # Erzeuge einen zufälligen bipolaren Vektor
    input_vec = np.random.choice([1, -1], size=16)
    # Schreiben
    sdm.write_memory(input_vec)
    # Lesen
    retrieved = sdm.read_memory(input_vec)

    # Ähnlichkeit (in +1/-1) => direkter Vergleich
    similarity = np.mean(input_vec == retrieved)
    assert similarity > 0.9, f"Erwartete hohe Ähnlichkeit, aber similarity={similarity:.2f}"

def test_sdm_write_read_multiple(default_sdm):
    """
    Schreibt mehrere Vektoren ins SDM und prüft,
    ob die Abfrage einiger Vektoren ungefähr passt.
    """
    sdm = default_sdm

    num_vectors = 5
    vecs = [np.random.choice([1, -1], size=16) for _ in range(num_vectors)]

    # Schreiben
    for vec in vecs:
        sdm.write_memory(vec)

    # Lesen und prüfen, ob wir die Vektoren halbwegs zurückbekommen
    sim_scores = []
    for vec in vecs:
        out = sdm.read_memory(vec)
        sim = np.mean(out == vec)
        sim_scores.append(sim)

    # Wir erwarten zumindest eine "merkbare" Ähnlichkeit
    avg_sim = np.mean(sim_scores)
    assert avg_sim > 0.7, f"Durchschnittliche Ähnlichkeit zu niedrig: {avg_sim:.2f}"

def test_sdm_reset_memory(default_sdm):
    """
    Testet, ob das Zurücksetzen der memory-Matrix funktioniert.
    """
    sdm = default_sdm
    input_vec = np.random.choice([1, -1], size=16)

    # Einmal schreiben
    sdm.write_memory(input_vec)
    # Anschließend resetten
    sdm.reset_memory()

    # Jetzt sollte memory nur Nullen haben
    assert np.allclose(sdm.memory, 0.0), "SDM-Memory wurde nicht korrekt zurückgesetzt."

def test_sdm_activation_percent_edge_cases():
    """
    Testet Edge-Cases für activation_percent: 0.0 und 1.0
    """
    # Case 1: activation_percent=0.0 => k=0 => es wird nichts geschrieben/gelesen
    sdm_zero = SparseDistributedMemory(address_size=10, memory_size=20, activation_percent=0.0)
    vec = np.random.choice([1, -1], size=10)
    sdm_zero.write_memory(vec)
    # Lesen => summiert nichts => reines sign(0) => +1
    out_zero = sdm_zero.read_memory(vec)
    all_plus_ones = np.ones_like(vec)
    assert np.all(out_zero == all_plus_ones), "Erwartet reinen +1-Vektor bei activation_percent=0.0"

    # Case 2: activation_percent=1.0 => k=memory_size => alle Hard Locations werden immer genommen
    sdm_full = SparseDistributedMemory(address_size=10, memory_size=20, activation_percent=1.0)
    sdm_full.write_memory(vec)
    out_full = sdm_full.read_memory(vec)
    # Da alle Hard Locations genommen werden, sind alle Writes "gemittelt"
    # => Möglicherweise sehr ähnlich an vec
    sim_full = np.mean(out_full == vec)
    assert sim_full > 0.7, f"Erwartete ausreichende Ähnlichkeit bei activation_percent=1.0, aber sim={sim_full:.2f}"

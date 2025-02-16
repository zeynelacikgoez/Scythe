# tests/test_plan_optim.py

import pytest
from plan_optimizer.plan_optim import optimize_plan

def test_minimal_scenario():
    """
    Testet, ob 'optimize_plan' für ein einfaches, lösbares Szenario
    eine sinnvolle Lösung (solution != None) und ein endliches objective_value zurückgibt.
    """
    scenario_data = {
        "factories": [
            {"name": "FactoryA", "max_production": 30},
            {"name": "FactoryB", "max_production": 40},
        ],
        "demands": [20],         # Gesamtnachfrage=20
        "max_resource": 50,      # Ausreichend, um die Demand zu decken
    }

    solution, obj_val, report = optimize_plan(
        data=scenario_data,
        solver_type="ortools",
        use_heuristics=False,
        multi_objective_mode=False,
        scenario_generation=False,
        verbose=False
    )
    assert solution is not None, "Erwartet eine valide Lösung, bekam None."
    assert obj_val < float('inf'), f"Objektivwert ist unendlich: {obj_val}"
    # Überprüfen, ob im Report etwas wie 'Feasibility: Feasible' oder 'Feasibility: True' steht
    assert "Feasibility: Feasible" in report or "Feasibility: True" in report, \
        "Erwartete feasible-Lösung im Report."


def test_infeasible_scenario():
    """
    Testet ein unlösbares Szenario, z. B. Demand größer als 
    max_resource * sum(max_production). Erwartet:
      - solution=None
      - obj_val=inf
      - der Report enthält 'Keine valide Lösung / infeasible'
    """
    scenario_data = {
        "factories": [
            {"name": "FactoryA", "max_production": 5},
            {"name": "FactoryB", "max_production": 5},
        ],
        "demands": [30],         # Gesamtnachfrage=30
        "max_resource": 10,      # max Prod = 10 => unlösbar
    }

    solution, obj_val, report = optimize_plan(
        data=scenario_data,
        solver_type="ortools",
        use_heuristics=False,
        multi_objective_mode=False,
        scenario_generation=False,
        verbose=False
    )

    # Falls die Lösung nicht None ist, dann sollte sie im Report NICHT 'Feasible' sein
    if solution is not None:
        assert "Feasible" not in report, "Erwartete 'Infeasible', aber Report sagt 'Feasible'?"
    else:
        # Bei None => typ. obj_val = inf
        assert obj_val == float('inf'), f"ObjVal sollte inf bei None-Lösung sein, bek={obj_val}"
        assert "Keine valide Lösung / infeasible" in report, (
            "Der Report sollte explizit 'Keine valide Lösung / infeasible' enthalten."
        )


def test_multi_objective_mode():
    """
    Testet, ob 'multi_objective_mode=True' Sinn ergibt und kein Fehler auftritt.
    """
    scenario_data = {
        "factories": [
            {"name": "F1", "max_production": 20},
        ],
        "demands": [10],
        "max_resource": 20,
    }

    solution, obj_val, report = optimize_plan(
        data=scenario_data,
        solver_type="ortools",
        use_heuristics=False,
        multi_objective_mode=True,
        scenario_generation=False,
        verbose=False
    )
    assert solution is not None, "Mit multi_objective_mode=True sollte es dennoch eine Lösung geben."
    assert obj_val < float('inf'), "ObjVal < inf erwartet."


def test_heuristics_improvement():
    """
    Testet, ob 'use_heuristics=True' das Ergebnis verbessern oder 
    zumindest nicht verschlechtern kann (z. B. local_search).
    """
    scenario_data = {
        "factories": [
            {"name": "BigFactory", "max_production": 2000},
        ],
        "demands": [1500],
        "max_resource": 2000,
    }

    # 1) Ohne Heuristik
    sol_noheur, val_noheur, rep_noheur = optimize_plan(
        data=scenario_data,
        solver_type="ortools",
        use_heuristics=False,
        multi_objective_mode=False,
        scenario_generation=False,
        verbose=False
    )

    # 2) Mit Heuristik
    sol_heur, val_heur, rep_heur = optimize_plan(
        data=scenario_data,
        solver_type="ortools",
        use_heuristics=True,
        multi_objective_mode=False,
        scenario_generation=False,
        verbose=False
    )

    # Beide Varianten sollten eine Lösung liefern
    assert sol_noheur is not None and sol_heur is not None, "Beide Varianten sollten eine Lösung liefern."

    # Die Heuristik sollte das Ergebnis typischerweise verbessern oder gleichwertig sein
    assert val_heur <= val_noheur + 1e-9, (
        f"Erwartet val_heur <= val_noheur, got {val_heur} vs {val_noheur}"
    )


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__]))

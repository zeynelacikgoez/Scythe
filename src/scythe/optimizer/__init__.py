# plan_optimizer/__init__.py

"""
Das plan_optimizer-Paket umfasst folgende Module:

- plan_optim.py        : Hauptlogik der Planungs- und Optimierungsläufe
- constraints.py       : Definition und Anwendung von Nebenbedingungen
- solver_interface.py  : Abstrakte Schnittstelle zu (MIP-)Solvern
- heuristics.py        : Heuristische Verfahren (Local Search, Genetic Algorithm, etc.)
- scenario_generator.py: Erzeugung verschiedener Szenarien (Base/Worst/Best/Random)
- validation.py        : Validierung und Kennzahlberechnung für Lösungen
- multi_objective.py   : Kombination mehrerer Zielfunktionen, Pareto-Front-Logik
- post_processing.py   : Reporting, CSV-Export, Visualisierung

Dieses __init__.py stellt die wichtigsten Funktionen und Klassen 
nach außen hin bereit, sodass man sie bequem importieren kann.
"""

from .plan_optim import optimize_plan
from .constraints import apply_constraints
from .solver_interface import SolverWrapper
from .heuristics import (
    run_heuristic_if_needed,
    local_search,
    genetic_algorithm,
    calculate_objective,
    mutate_solution,
    crossover
)
from .scenario_generator import ScenarioGenerator
from .validation import validate_solution, calculate_solution_metrics
from .multi_objective import combine_objectives, extract_pareto_front
from .post_processing import (
    generate_plan_report,
    save_solution_to_csv,
    plot_solution_distribution
)

__all__ = [
    "optimize_plan",
    "apply_constraints",
    "SolverWrapper",
    "run_heuristic_if_needed",
    "local_search",
    "genetic_algorithm",
    "calculate_objective",
    "mutate_solution",
    "crossover",
    "ScenarioGenerator",
    "validate_solution",
    "calculate_solution_metrics",
    "combine_objectives",
    "extract_pareto_front",
    "generate_plan_report",
    "save_solution_to_csv",
    "plot_solution_distribution"
]

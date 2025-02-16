# plan_optimizer/plan_optim.py

from .constraints import apply_constraints
from .solver_interface import SolverWrapper
from .heuristics import run_heuristic_if_needed
from .validation import validate_solution
from .post_processing import generate_plan_report
from .scenario_generator import ScenarioGenerator
from .multi_objective import combine_objectives, extract_pareto_front


def optimize_plan(
    data,
    solver_type: str = "ortools",
    use_heuristics: bool = False,
    multi_objective_mode: bool = False,
    scenario_generation: bool = True,
    verbose: bool = True
):
    """
    Haupteinstiegspunkt für die Planungs- und Optimierungsschicht.

    Ablauf:
    1) (Optional) Erzeuge verschiedene Szenarien (BaseCase/WorstCase etc.) via ScenarioGenerator
    2) Für jedes Szenario:
       - Erstelle Solver (z. B. OR-Tools)
       - Wende Constraints an (siehe constraints.py)
       - Löse
       - (Optional) Heuristik
       - Prüfe Validität
       - (Optional) Multi-Objective
       - Bester Plan wird gemerkt
    3) Falls keine valide Lösung gefunden => infeasible-Meldung im Report
    """

    # 1) Szenarioerzeugung
    if scenario_generation:
        scenario_list = ScenarioGenerator(data).create_scenarios()
        if verbose:
            print(f"[plan_optim] Generiere {len(scenario_list)} Szenarien ...")
    else:
        scenario_list = [data]

    best_solution = None
    best_objective_value = float('inf')
    best_scenario_info = None

    # 2) Schleife über Szenarien
    for idx, scenario in enumerate(scenario_list):
        if verbose:
            print(f"[plan_optim] Starte Optimierung für Szenario {idx+1} ...")

        solver = SolverWrapper(solver_type=solver_type)
        solver.init_model(scenario)

        apply_constraints(solver, scenario)

        solution, objective_value = solver.solve()

        if verbose:
            print(f"  -> Lösung gefunden mit Objective = {objective_value:.4f}")

        # 2a) Heuristiken
        if use_heuristics and solution is not None:
            solution, new_obj = run_heuristic_if_needed(solution, objective_value)
            if verbose and new_obj < objective_value:
                print(f"  -> Heuristik verbesserte Lösung auf {new_obj:.4f}")
            objective_value = min(objective_value, new_obj)

        # 2b) Validierung
        is_valid = validate_solution(solution, scenario)
        if not is_valid:
            if verbose:
                print("  -> Lösung ist ungültig / Constraints verletzt.")
            continue

        # 2c) Multi-Objective
        if multi_objective_mode and solution is not None:
            # Beispiel: wir haben Cost als single value
            mo_values = [(objective_value, "Cost")]
            combined_value = combine_objectives(mo_values)
            objective_value = combined_value

        # 2d) Beste Lösung merken
        if solution is not None and objective_value < best_objective_value:
            best_solution = solution
            best_objective_value = objective_value
            best_scenario_info = scenario

    # 3) Reporting
    if best_solution is not None:
        report = generate_plan_report(best_solution, best_objective_value, best_scenario_info)
        if verbose:
            print("[plan_optim] Beste Lösung gewählt ...")
    else:
        # Keine valide Lösung => infeasible
        report = (
            "Keine valide Lösung / infeasible.\n"
            "Hinweis: Das Szenario kann nicht realisiert werden.\n"
        )
        if verbose:
            print("[plan_optim] Keine valide Lösung über alle Szenarien => infeasible.")

    return best_solution, best_objective_value, report

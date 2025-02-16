# multi_objective.py

from math import inf


def combine_objectives(objectives, weights=None, sense="minimize"):
    """
    Kombiniert mehrere Zielfunktionswerte (z. B. 'Cost', 'CO2', etc.)
    zu einem einzigen Skalar via Weighted Sum.

    Parameters
    ----------
    objectives : list of tuples
        Liste von (obj_value, name).
        Beispiel: [(120.0, "Cost"), (50.0, "CO2")].
        'obj_value' = Wert einer Zielfunktion,
        'name' = identifier, z. B. "Cost", "CO2" o. ä.

    weights : dict (str -> float) oder None
        Gewichte pro Zielfunktion, z. B. {"Cost": 0.8, "CO2": 0.2}.
        Falls None, werden alle Zielfunktionen gleich gewichtet (1.0).

    sense : str
        "minimize" oder "maximize" für den Gesamtwert.
        (Hinweis: Falls du verschiedene Richtungen pro Ziel hast,
         wird es komplizierter; man müsste sie manuell invertieren.)

    Returns
    -------
    combined_value : float
        Der kombinierte Skalarwert, der dann z. B. in einer einobjektiven Solve-Phase
        genutzt oder zum Ranking herangezogen werden kann.

    Beispiel
    --------
    >>> objs = [(120.0, "Cost"), (50.0, "CO2")]
    >>> w = {"Cost": 0.7, "CO2": 0.3}
    >>> combine_objectives(objs, w, sense="minimize")
    99.0
    """
    if not objectives:
        return 0.0

    # Falls keine Gewichte, setze Default=1.0
    if weights is None:
        weights = {}
    # Sammele Keys aus objectives
    # -> (obj_val, name)
    # => Falls name in weights, nimm das, sonst 1.0
    combined_value = 0.0
    for (obj_val, name) in objectives:
        w = weights.get(name, 1.0)
        combined_value += w * obj_val

    # Falls sense="maximize", invertiere das Ergebnis
    if sense.lower() == "maximize":
        combined_value = -combined_value

    return combined_value


def extract_pareto_front(solutions, objective_names=None, sense="minimize"):
    """
    Extrahiert die Pareto-Front einer Menge von Lösungen mit mehreren Zielfunktionen.

    Parameters
    ----------
    solutions : list of dict
        Jede Lösung ist ein Dictionary, das pro Zielfunktion einen Wert enthält:
        z. B. [{"Cost": 120, "CO2": 50, "Solution": {...}},
               {"Cost": 100, "CO2": 60, "Solution": {...}}, ...]

    objective_names : list of str or None
        Liste der Objectives, die zu beachten sind. Falls None, werden alle
        numerischen Keys (außer "Solution") interpretiert.

    sense : str
        "minimize" oder "maximize" (für alle Objectives).
        (Für unterschiedliche Richtungen pro Objective bräuchte man
         eine komplexere Logik.)

    Returns
    -------
    front : list of dict
        Die nicht-dominierten Lösungen (Pareto-Front).

    Beispiel
    --------
    >>> sols = [
    ...   {"Cost": 120, "CO2": 50, "Solution": {...}},
    ...   {"Cost": 100, "CO2": 60, "Solution": {...}},
    ...   {"Cost": 110, "CO2": 40, "Solution": {...}}
    ... ]
    >>> front = extract_pareto_front(sols, objective_names=["Cost", "CO2"], sense="minimize")
    """
    if not solutions:
        return []

    if objective_names is None:
        example_sol = solutions[0]
        objective_names = [k for k in example_sol.keys() if k != "Solution"]

    def dominates(sol_a, sol_b):
        """
        Prüft, ob 'sol_a' 'sol_b' dominiert.
        sense='minimize': sol_a dominiert sol_b, wenn a_i <= b_i für alle i
                          und a_j < b_j für mind. ein j.
        sense='maximize': sol_a >= b_i und a_j > b_j ...
        """
        better_or_equal = True
        strictly_better = False

        if sense.lower() == "minimize":
            for obj_name in objective_names:
                if sol_a[obj_name] > sol_b[obj_name]:
                    better_or_equal = False
                    break
                if sol_a[obj_name] < sol_b[obj_name]:
                    strictly_better = True
        else:  # "maximize"
            for obj_name in objective_names:
                if sol_a[obj_name] < sol_b[obj_name]:
                    better_or_equal = False
                    break
                if sol_a[obj_name] > sol_b[obj_name]:
                    strictly_better = True

        return better_or_equal and strictly_better

    pareto_front = []
    for sol in solutions:
        # Check, ob sol von einer Lösung in pareto_front dominiert wird
        dominated = False
        dominated_indices = []
        for i, pf_sol in enumerate(pareto_front):
            if dominates(pf_sol, sol):
                dominated = True
                break
            if dominates(sol, pf_sol):
                dominated_indices.append(i)

        if not dominated:
            # Entferne alle pf_sol, die nun von 'sol' dominiert werden
            for idx in reversed(dominated_indices):
                del pareto_front[idx]
            pareto_front.append(sol)

    return pareto_front


def generate_weighted_solutions(
    base_scenario,
    solver_func,
    objective_info,
    weight_grid=None,
    sense="minimize"
):
    """
    Beispiel-Funktion, die wiederholt 'solver_func' aufruft,
    um eine Reihe von Gewichtungen (weight_grid) auszuprobieren
    und Lösungen zu sammeln. Anschließend kann man
    'extract_pareto_front' anwenden.

    Parameters
    ----------
    base_scenario : dict
        Grunddaten/Constraints. Je nach Problem z. B. "factories", "demands".
    solver_func : callable
        Eine Funktion, die (scenario, weights) entgegennimmt und
        eine Lösung + Objective-Werte zurückgibt.
        (z. B. partial(optimize_plan, ...))
    objective_info : list of str
        Namen der Zielfunktionen, z. B. ["Cost", "CO2"].
    weight_grid : list of dict, optional
        Liste von z. B. {"Cost":0.8, "CO2":0.2}, {"Cost":0.5, "CO2":0.5}, ...
        Falls None, generieren wir ein Raster an Werten.
    sense : str
        "minimize" oder "maximize"

    Returns
    -------
    solutions : list of dict
        Jede dict enthält die Zielfunktionswerte + die "Solution".
        Beispiel:
         [
           {"Cost":120, "CO2":50, "Solution": {...}, "Weights": {...}},
           {"Cost":100, "CO2":60, ...}
           ...
         ]
    """
    if weight_grid is None:
        # Generiere z. B. simple Grid for 2-objective
        # z. B. [ (1,0), (0.8,0.2), (0.5,0.5), (0.2,0.8), (0,1) ]
        weight_grid = []
        steps = [0.0, 0.2, 0.5, 0.8, 1.0]
        if len(objective_info) == 2:
            for w1 in steps:
                w2 = 1.0 - w1
                weight_grid.append({objective_info[0]: w1, objective_info[1]: w2})
        else:
            # Falls du mehr als 2 Obj. hast, generiere anders
            raise NotImplementedError("Auto-Grid nur für 2 Objektive implementiert")

    solutions_collected = []

    for wdict in weight_grid:
        # solver_func soll das Problem lösen mit den entsprechenden Gewichten
        # und uns objective-Werte zurückliefern
        solution_dict = solver_func(base_scenario, wdict, sense)
        # z. B. => {
        #   "Solution": ...,
        #   "Cost": 120,
        #   "CO2": 50
        # }
        # wir fügen "Weights" als Info hinzu:
        solution_dict["Weights"] = wdict
        solutions_collected.append(solution_dict)

    return solutions_collected


def example_solver_func(scenario, weights, sense="minimize"):
    """
    Demo-Funktion, wie man solver_func definieren könnte.
    In echt würdest du hier 'optimize_plan' oder
    dein MIP-Solver-Aufruf machen und
    am Ende die Werte ["Cost", "CO2"] extrahieren.
    """
    # (Pseudo) 'solve' => Cost random, CO2 random
    import random
    cost_val = 100 + random.randint(-20, 20)
    co2_val = 40 + random.randint(-10, 10)

    # Im realen Leben: solver => solution => cost, co2

    return {
        "Solution": {"some_var": 42},
        "Cost": cost_val,
        "CO2": co2_val,
    }

if __name__ == "__main__":
    # Kleines Demo
    objs = [(120.0, "Cost"), (50.0, "CO2")]
    w = {"Cost": 0.7, "CO2": 0.3}
    val = combine_objectives(objs, w, "minimize")
    print("[multi_objective] Weighted Sum =", val)

    sol_list = [
        {"Cost": 120, "CO2": 50, "Solution": {"x": 1}},
        {"Cost": 100, "CO2": 60, "Solution": {"x": 2}},
        {"Cost": 110, "CO2": 40, "Solution": {"x": 3}},
        {"Cost": 130, "CO2": 45, "Solution": {"x": 4}}
    ]
    pf = extract_pareto_front(sol_list, ["Cost", "CO2"], "minimize")
    print("[multi_objective] Pareto Front:", pf)

    # Test generate_weighted_solutions
    base_scen = {}
    # -> Schätze wir haben 2 objectives: "Cost", "CO2"
    w_solutions = generate_weighted_solutions(
        base_scenario=base_scen,
        solver_func=example_solver_func,
        objective_info=["Cost", "CO2"],
        weight_grid=[{"Cost": 1.0, "CO2": 0.0},
                     {"Cost": 0.5, "CO2": 0.5},
                     {"Cost": 0.0, "CO2": 1.0}]
    )
    print("[multi_objective] Weighted solutions:\n", w_solutions)
    pf2 = extract_pareto_front(w_solutions, objective_names=["Cost", "CO2"], sense="minimize")
    print("[multi_objective] Weighted solutions' Pareto Front:\n", pf2)

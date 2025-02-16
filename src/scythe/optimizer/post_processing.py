# plan_optimizer/post_processing.py

import csv
import os
import math
import matplotlib.pyplot as plt

from plan_optimizer.validation import calculate_solution_metrics, validate_solution

def generate_plan_report(solution: dict, objective_value: float, scenario: dict = None) -> str:
    """
    Erzeugt einen textuellen Bericht über die gefundene Lösung.
    
    Parameters
    ----------
    solution : dict or None
        Das Dictionary mit den Variablennamen und -werten (falls valide Lösung).
        Falls None => infeasible.
    objective_value : float
        Wert der Zielfunktion (oft Cost). Kann inf sein, wenn infeasible.
    scenario : dict, optional
        Das verwendete Szenario (z. B. demands, factories ...),
        um im Bericht mehr Kontext zu liefern.

    Returns
    -------
    report : str
        Zusammenhängender Textbericht.
    """

    report_lines = []
    report_lines.append("----- Plan Optimization Report -----\n")

    # Falls keine valide Lösung existiert
    if solution is None:
        report_lines.append("** Keine valide Lösung / infeasible **\n")
        return "".join(report_lines)

    # Wenn eine Lösung vorliegt, normaler Report
    report_lines.append(f"Objective Value: {objective_value:.2f}\n")

    # Ggf. Validierungsinfo
    feasible = True
    if scenario is not None:
        feasible = validate_solution(solution, scenario)
    report_lines.append(f"Feasibility: {'Feasible' if feasible else 'Infeasible'}\n")

    # Zusätzliche Metriken (z. B. total_production, demand_gap, etc.)
    if scenario is not None:
        metrics = calculate_solution_metrics(solution, scenario)
        report_lines.append("\n== Metrics ==\n")
        for k, v in metrics.items():
            if isinstance(v, float):
                report_lines.append(f"{k}: {v:.2f}\n")
            else:
                report_lines.append(f"{k}: {v}\n")
        report_lines.append("\n")

        scenario_name = scenario.get("scenario_name", "NoName")
        report_lines.append(f"Scenario Name: {scenario_name}\n")

    # Variablenwerte der Lösung auflisten
    report_lines.append("\n== Solution Variables ==\n")
    sorted_vars = sorted(solution.items(), key=lambda x: x[0])
    for var_name, var_val in sorted_vars:
        report_lines.append(f"{var_name}: {var_val:.2f}\n")

    report_lines.append("\n--- End of Report ---\n")

    return "".join(report_lines)


def save_solution_to_csv(solution: dict, file_path: str, objective_value: float = None, scenario_name: str = None):
    """
    Schreibt die Variablenwerte und optional den Zielwert / Szenario-Name 
    in eine CSV-Datei.

    Parameters
    ----------
    solution : dict
        Variablenname -> Wert
    file_path : str
        Ausgabedatei
    objective_value : float, optional
        Falls angegeben, wird zusätzlich notiert
    scenario_name : str, optional
        Falls angegeben, wird zusätzlich notiert
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(["Variable", "Value"])

        if scenario_name:
            writer.writerow(["ScenarioName", scenario_name])
        if objective_value is not None:
            writer.writerow(["ObjectiveValue", f"{objective_value:.2f}"])

        for var_name, var_val in sorted(solution.items()):
            writer.writerow([var_name, f"{var_val:.2f}"])


def plot_solution_distribution(solution: dict, title: str = "Solution Distribution", show_plot: bool = True):
    """
    Zeigt ein Balkendiagramm der Variablenwerte (nur positive > 0).
    """
    filtered = {k: v for k, v in solution.items() if v > 0}

    if not filtered:
        print("Keine Variablen mit positivem Wert für den Plot.")
        return

    var_names = list(filtered.keys())
    var_values = list(filtered.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(var_names, var_values)
    ax.set_title(title)
    ax.set_ylabel("Value")
    plt.xticks(rotation=45, ha='right')

    if show_plot:
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)

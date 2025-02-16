# plan_optimizer/validation.py

import math

def validate_solution(solution: dict, scenario: dict) -> bool:
    """
    Prüft, ob die gegebene Lösung alle relevanten Constraints einhält.
    Gibt True zurück, wenn alles in Ordnung, ansonsten False.

    Parameters
    ----------
    solution : dict
        Dictionary, das Variablenname -> Wert (float/int) enthält.
        Beispiel: { "x_factory_FactoryA": 50.0, "x_factory_FactoryB": 80.0, "t_FactoryA_to_Region1": 20.0, ... }
    scenario : dict
        Datenstruktur mit den Planungsparametern, z. B. demands, factories, max_resource, etc.

    Returns
    -------
    bool
        True, falls keine Constraints verletzt wurden, sonst False.
    """

    # 1) Produktionsgrenzen-Check
    factories = scenario.get("factories", [])
    for fac in factories:
        fac_name = fac["name"]
        max_prod = fac.get("max_production", math.inf)
        var_name = f"x_factory_{fac_name}"
        if var_name in solution:
            prod_value = solution[var_name]
            # Produktionsmenge darf nicht negativ sein und max_prod nicht überschreiten
            if prod_value < 0:
                return False
            if prod_value > max_prod:
                return False

    # 2) Ressourcencheck: Summe aller Produktionsvariablen <= max_resource
    max_resource = scenario.get("max_resource", math.inf)
    total_production = 0.0
    for fac in factories:
        var_name = f"x_factory_{fac['name']}"
        if var_name in solution:
            total_production += solution[var_name]
    if total_production > max_resource:
        return False

    # 3) Nachfrageerfüllung: Summe der Produktionsmengen >= Summe demands (falls nur 1 Produkt)
    demands = scenario.get("demands", [])
    if len(demands) > 0:
        total_demand = sum(demands)
        if total_production < total_demand:
            return False

    # 4) Transportcheck (sehr einfaches Beispiel)
    transport_capacity = scenario.get("transport_capacity", math.inf)
    routes = scenario.get("transport_routes", [])
    total_transport = 0.0
    for route in routes:
        var_name = f"t_{route['from']}_to_{route['to']}"
        if var_name in solution:
            transported = solution[var_name]
            if transported < 0:
                return False
            # Check max_truck_load pro Route
            max_load = route.get("max_truck_load", math.inf)
            if transported > max_load:
                return False
            total_transport += transported
    if total_transport > transport_capacity:
        return False

    # Alles bestanden -> True
    return True


def calculate_solution_metrics(solution: dict, scenario: dict) -> dict:
    """
    Liefert einige Kennzahlen zur Lösung. Dient zum Reporting oder zur Analyse.
    Wird oft im Anschluss an 'validate_solution' aufgerufen.

    Beispielhafte Kennzahlen:
    - total_production
    - total_demand
    - resource_usage
    - transport_used
    - demand_gap (falls nicht erfüllt)
    - feasible (True/False)

    Returns
    -------
    dict
        Dictionary mit Metriken zur Lösung
    """

    factories = scenario.get("factories", [])
    total_production = 0.0
    for fac in factories:
        var_name = f"x_factory_{fac['name']}"
        if var_name in solution:
            total_production += solution[var_name]

    total_demand = sum(scenario.get("demands", []))
    resource_limit = scenario.get("max_resource", math.inf)

    routes = scenario.get("transport_routes", [])
    total_transport = 0.0
    for route in routes:
        var_name = f"t_{route['from']}_to_{route['to']}"
        if var_name in solution:
            total_transport += solution[var_name]

    # Demand-Gap = (total_demand - total_production), falls production < demand
    demand_gap = max(0, total_demand - total_production)

    metrics = {
        "total_production": total_production,
        "total_demand": total_demand,
        "resource_usage": total_production,
        "resource_limit": resource_limit,
        "transport_used": total_transport,
        "transport_capacity": scenario.get("transport_capacity", math.inf),
        "demand_gap": demand_gap,
        "feasible": validate_solution(solution, scenario)
    }

    return metrics

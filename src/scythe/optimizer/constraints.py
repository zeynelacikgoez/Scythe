# plan_optimizer/constraints.py

def apply_constraints(solver, scenario):
    """
    Wendet alle definierten Constraints auf das jeweilige Solver-Modell an.
    'solver' ist eine Instanz von SolverWrapper,
    'scenario' enthält die Input-Daten für dieses Szenario, z. B.:

    {
      "factories": [
         {"name": "FactoryA", "max_production": 100},
         {"name": "FactoryB", "max_production": 200},
         ...
      ],
      "demands": [10, 20, 15],
      "max_resource": 300,
      "transport_capacity": 100,
      "transport_routes": [...],
      "inventory_limits": {
          "period1": {"min": 10, "max": 100},
          "period2": {"min": 0,  "max": 200},
      },
      ...
    }
    """
    _apply_production_constraints(solver, scenario)
    _apply_resource_constraints(solver, scenario)
    _apply_demand_satisfaction(solver, scenario)
    _apply_transport_constraints(solver, scenario)
    _apply_inventory_constraints(solver, scenario)   # NEU: Inventar
    _apply_custom_constraints(solver, scenario)      # OPTIONAL: weitere


# --------------------------------------------------------------------------
# 1) Produktionsgrenzen pro Fabrik
# --------------------------------------------------------------------------
def _apply_production_constraints(solver, scenario):
    """
    Beispiel: Obergrenze pro Betrieb / Fabrik.
    Angenommen, scenario['factories'] ist eine Liste von Dicts:
      [
        {'name': 'FactoryA', 'max_production': 100},
        {'name': 'FactoryB', 'max_production': 200},
        ...
      ]
    Wir erzeugen Variablen x_factory[i] = Produktionsmenge.
    """
    factories = scenario.get('factories', [])
    if not factories:
        return

    solver.model_x_factory = {}  # Dictionary, var_name -> var
    for fac in factories:
        fac_name = fac['name']
        max_prod = fac.get('max_production', None)
        var_name = f"x_factory_{fac_name}"
        x_var = solver.add_variable(name=var_name, lb=0, ub=max_prod, var_type='continuous')
        solver.model_x_factory[var_name] = x_var


# --------------------------------------------------------------------------
# 2) Ressourcen-Constraint (z. B. Gesamtressourcen)
# --------------------------------------------------------------------------
def _apply_resource_constraints(solver, scenario):
    """
    Beispiel: scenario['max_resource'] = 500
    => Summe aller x_factory[...] <= 500
    """
    max_resource = scenario.get('max_resource', None)
    if max_resource is None:
        return

    x_factory_dict = getattr(solver, 'model_x_factory', None)
    if not x_factory_dict:
        return

    all_factory_vars = list(x_factory_dict.values())
    constraint_expr = sum(all_factory_vars)
    solver.add_constraint(constraint_expr, "<=", max_resource, name="ResourceConstraint")


# --------------------------------------------------------------------------
# 3) Nachfrageerfüllung (Summe Produktion >= Summe Demand)
# --------------------------------------------------------------------------
def _apply_demand_satisfaction(solver, scenario):
    """
    Beispiel: Summenproduktion >= sum(demands).
    """
    demands = scenario.get('demands', [])
    if not demands:
        return

    total_demand = sum(demands)

    x_factory_dict = getattr(solver, 'model_x_factory', None)
    if not x_factory_dict:
        return

    total_production = sum(x_factory_dict.values())
    solver.add_constraint(total_production, ">=", total_demand, name="DemandSatisfaction")


# --------------------------------------------------------------------------
# 4) Transport-Constraints (z. B. Kapazität, route-limits)
# --------------------------------------------------------------------------
def _apply_transport_constraints(solver, scenario):
    """
    Beispiel: scenario['transport_capacity'] = 300
    scenario['transport_routes'] = [
      {'from': 'FactoryA', 'to': 'Region1', 'max_truck_load': 50},
      ...
    ]
    """
    transport_cap = scenario.get('transport_capacity', None)
    routes = scenario.get('transport_routes', [])
    if not transport_cap or not routes:
        return

    solver.model_transport_vars = {}
    for route in routes:
        route_name = f"t_{route['from']}_to_{route['to']}"
        max_load = route.get('max_truck_load', None)
        t_var = solver.add_variable(name=route_name, lb=0, ub=max_load, var_type='continuous')
        solver.model_transport_vars[route_name] = t_var

    # Globale Transportkapazität
    all_t_vars = list(solver.model_transport_vars.values())
    solver.add_constraint(sum(all_t_vars), "<=", transport_cap, name="TransportCapacity")


# --------------------------------------------------------------------------
# 5) Inventar-Constraints (NEU)
# --------------------------------------------------------------------------
def _apply_inventory_constraints(solver, scenario):
    """
    Beispiel: scenario['inventory_limits'] = {
        "period1": {"min": 10, "max": 100},
        "period2": {"min": 0,  "max": 200},
        ...
    }
    Dann legen wir Variablen inv_periodX an und fordern min <= inv_periodX <= max.
    In einer echten Anwendung würdest du die Lagerbilanz: inv_t+1 = inv_t + production_t - demand_t definieren.
    """
    inv_data = scenario.get('inventory_limits', None)
    if not inv_data:
        return

    solver.model_inventory_vars = {}

    for period, limits in inv_data.items():
        var_name = f"inv_{period}"
        min_val = limits.get('min', 0)
        max_val = limits.get('max', None)  # None = unbounded
        inv_var = solver.add_variable(name=var_name, lb=min_val, ub=max_val, var_type='continuous')
        solver.model_inventory_vars[var_name] = inv_var

    # Beispiel für eine einfache Lagerbilanz:
    # Falls scenario["inventory_balance"] = True, erzwinge:
    #   inv_period2 = inv_period1 + Sum(production in period1) - Sum(demand in period1)
    # Hier nur Dummy-Logik als Demonstration:
    if scenario.get("inventory_balance", False):
        # Wir gehen von 2 Perioden aus, rein beispielhaft
        inv_var1 = solver.model_inventory_vars.get("inv_period1")
        inv_var2 = solver.model_inventory_vars.get("inv_period2")
        if inv_var1 and inv_var2:
            # Summiere Produktion in period1 => x_factory[...]?
            # Summiere demand in period1 => ???

            # Dummy: inv_period2 >= inv_period1 - 5
            # => solver.add_constraint(inv_var2 - inv_var1, ">=", -5, name="InvBalance")
            solver.add_constraint(inv_var2 - inv_var1, ">=", -5, name="InvBalance_min")
            solver.add_constraint(inv_var2 - inv_var1, "<=", 5, name="InvBalance_max")


# --------------------------------------------------------------------------
# 6) Custom Constraints (optional)
# --------------------------------------------------------------------------
def _apply_custom_constraints(solver, scenario):
    """
    Platz für benutzerdefinierte Constraints, z. B. Wartungszeiten, 
    Schichtplanung, Emissionsgrenzen, etc.

    Beispiel: scenario["co2_limit"] = 500
      => Summierter CO2-Ausstoß (abhängig von x_factory_A, etc.) <= 500
    """
    co2_limit = scenario.get("co2_limit", None)
    if co2_limit is not None:
        # Wir bräuchten eine Logik, wie CO2 pro Produktion berechnet wird.
        # z. B. scenario["factories"] enthält "co2_per_unit"
        x_factory_dict = getattr(solver, 'model_x_factory', None)
        if x_factory_dict:
            total_co2_expr = 0
            for fac in scenario.get("factories", []):
                fac_name = fac["name"]
                co2_rate = fac.get("co2_per_unit", 0)
                var_name = f"x_factory_{fac_name}"
                x_var = x_factory_dict.get(var_name)
                if x_var:
                    total_co2_expr += co2_rate * x_var

            solver.add_constraint(total_co2_expr, "<=", co2_limit, name="CO2Constraint")

    # Weitere mögliche individuelle Constraints ...

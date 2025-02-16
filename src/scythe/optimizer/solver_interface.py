# plan_optimizer/solver_interface.py

import math
try:
    from ortools.linear_solver import pywraplp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

# Optional: Gurobi, CPLEX, etc. hier importieren:
# import gurobipy as gp
# import cplex

class SolverWrapper:
    """
    Eine Abstraktionsschicht um verschiedene LP/MIP-Solver herum.
    Erzeugt Variablen, Constraints und löst das Modell.

    Aktuell implementiert: OR-Tools (SCIP).
    Erweiterbar für Gurobi, CPLEX usw.
    """

    def __init__(self, solver_type="ortools", sense="minimize"):
        """
        Parameters
        ----------
        solver_type : str
            Bezeichnung des Solvers, z.B. 'ortools', 'gurobi', 'cplex', ...
        sense : str
            'minimize' oder 'maximize'
        """
        self.solver_type = solver_type
        self.sense = sense.lower()
        self.model = None
        self.variables = []
        self.constraints = []
        self.objective_expr = None  # Expression oder Liste von Expressions
        self._is_model_built = False

    def init_model(self, data=None):
        """
        Erzeugt das interne Modellobjekt, z. B. den OR-Tools-Solver.
        """
        if self.solver_type == "ortools":
            if not ORTOOLS_AVAILABLE:
                raise ImportError("OR-Tools ist nicht installiert oder konnte nicht importiert werden.")
            # Mit SCIP (Mixed Integer Programming):
            self.model = pywraplp.Solver.CreateSolver('SCIP')
            if not self.model:
                raise RuntimeError("Konnte keinen OR-Tools (SCIP) Solver erstellen.")
        else:
            # Hier könntest du Gurobi, CPLEX o. ä. initialisieren:
            raise NotImplementedError(f"Solver '{self.solver_type}' wird noch nicht unterstützt.")
        self._is_model_built = True

    def add_variable(self, name, lb=0.0, ub=None, var_type='continuous'):
        """
        Deklariert eine Variable im Modell.

        Parameters
        ----------
        name : str
            Name der Variable
        lb : float
            Untere Grenze (default: 0)
        ub : float oder None
            Obere Grenze oder None = keine Begrenzung
        var_type : str
            'continuous' oder 'integer' (ganzzahlige Variable).
            (Falls boole'sch -> 'integer' mit ub=1)

        Returns
        -------
        var : Solver-spezifisches Variablenobjekt
        """
        if not self._is_model_built:
            raise RuntimeError("Bitte zuerst init_model() aufrufen, bevor Variablen angelegt werden.")

        if ub is None:
            ub = self.model.infinity()

        if self.solver_type == "ortools":
            if var_type == 'continuous':
                var = self.model.NumVar(lb, ub, name)
            elif var_type == 'integer':
                var = self.model.IntVar(lb, ub, name)
            else:
                raise ValueError("var_type muss 'continuous' oder 'integer' sein.")
        else:
            # Andere Solver-Implementierung
            raise NotImplementedError(f"Solver '{self.solver_type}' wird noch nicht unterstützt.")

        self.variables.append(var)
        return var

    def add_constraint(self, left_expr, sense, right_value, name=None):
        """
        Fügt dem Modell eine Nebenbedingung hinzu.

        Parameters
        ----------
        left_expr : Ausdruck (z. B. sum(...) oder var)
        sense : str
            Vergleichsoperator: '<=', '>=', '=='
        right_value : float
            Konstante auf der rechten Seite
        name : str
            Optionaler Name für die Constraint
        """
        if not self._is_model_built:
            raise RuntimeError("Bitte init_model() aufrufen, bevor Constraints angelegt werden.")

        if self.solver_type == "ortools":
            if sense == "<=":
                ct = self.model.Add(left_expr <= right_value, name=name)
            elif sense == ">=":
                ct = self.model.Add(left_expr >= right_value, name=name)
            elif sense == "==":
                ct = self.model.Add(left_expr == right_value, name=name)
            else:
                raise ValueError(f"Unbekannter Vergleichsoperator: {sense}")
            self.constraints.append(ct)
        else:
            # Andere Solver-Logik
            raise NotImplementedError(f"Solver '{self.solver_type}' wird noch nicht unterstützt.")

    def set_objective(self, expr, name="obj"):
        """
        Legt die Zielfunktion fest.

        Parameters
        ----------
        expr : Ausdruck
            MIP- / LP-Ausdruck, der minimiert oder maximiert werden soll
        name : str
            Name der Zielfunktions-Definition
        """
        if not self._is_model_built:
            raise RuntimeError("Bitte init_model() aufrufen, bevor Ziele definiert werden.")

        if self.solver_type == "ortools":
            if self.sense == "minimize":
                self.model.Minimize(expr)
            else:
                self.model.Maximize(expr)
            self.objective_expr = expr
        else:
            raise NotImplementedError(f"Solver '{self.solver_type}' wird noch nicht unterstützt.")

    def solve(self):
        """
        Führt die Optimierung durch und gibt (solution, objective_value) zurück.

        Returns
        -------
        solution : dict
            Mapping Variablenname -> Wert
        objective_value : float
            Wert der Zielfunktion
        """
        if not self._is_model_built:
            raise RuntimeError("Das Modell wurde noch nicht initialisiert.")

        if self.solver_type == "ortools":
            status = self.model.Solve()
            if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
                # OR-Tools kann FEASIBLE zurückgeben, wenn Zeitlimit o. Ä. erreicht
                obj_val = self.model.Objective().Value()
                solution = {}
                for var in self.variables:
                    solution[var.name()] = var.solution_value()
                return solution, obj_val
            else:
                # Keine optimale oder zulässige Lösung
                return None, math.inf
        else:
            raise NotImplementedError(f"Solver '{self.solver_type}' wird noch nicht unterstützt.")

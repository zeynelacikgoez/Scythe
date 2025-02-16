# plan_optimizer/scenario_generator.py

import copy
import random

class ScenarioGenerator:
    """
    Erzeugt unterschiedliche Szenarien auf Basis der Eingabedaten.
    Beispielsweise:
    - Base Case: Originalwerte
    - Worst Case: Höhere Nachfrage, geringere Ressourcen
    - Best Case: Niedrigere Nachfrage, höhere Ressourcen
    - Zufällige / Stresstest-Szenarien

    'data' kann eine beliebige Struktur sein, z. B. ein Dictionary mit Schlüsseln wie
      {
        "factories": [
           {"name": "FactoryA", "max_production": 100},
           {"name": "FactoryB", "max_production": 200}
        ],
        "demands": [10, 20, 15],
        "max_resource": 300,
        "transport_capacity": 150,
        ...
      }
    """

    def __init__(self, data):
        """
        Konstruktor, der die Originaldaten speichert.
        """
        self.data = data

    def create_scenarios(self, generate_random=False, random_scenario_count=2):
        """
        Erstellt eine Liste von Szenarien.

        Parameter:
        ----------
        generate_random : bool
            Falls True, werden zusätzlich zu Base/Worst/Best noch zufällige
            Szenarien erzeugt (z. B. Stresstest).
        random_scenario_count : int
            Anzahl der zu erzeugenden Zufallsszenarien.

        Returns:
        --------
        scenarios : list
            Liste mit verschiedenen Versionen der Daten (base_case, worst_case, best_case, etc.).
        """
        scenarios = []

        # 1) Base Case
        base = self._create_base_case()
        scenarios.append(base)

        # 2) Worst Case
        worst = self._create_worst_case()
        scenarios.append(worst)

        # 3) Best Case
        best = self._create_best_case()
        scenarios.append(best)

        # 4) Optionale zufällige Szenarien
        if generate_random:
            for i in range(random_scenario_count):
                rnd = self._create_random_scenario(i)
                scenarios.append(rnd)

        return scenarios

    def _create_base_case(self):
        """
        Kopiert die Originaldaten als 'Base Case'.
        """
        base_case = copy.deepcopy(self.data)
        base_case["scenario_name"] = "BaseCase"
        return base_case

    def _create_worst_case(self):
        """
        Erhöht z. B. die Nachfrage, reduziert Ressourcen, etc.
        """
        worst_case = copy.deepcopy(self.data)
        worst_case["scenario_name"] = "WorstCase"

        # Beispiel: Steigere Nachfrage um +50%
        if "demands" in worst_case:
            worst_case["demands"] = [int(d * 1.5) for d in worst_case["demands"]]

        # Reduziere max_resource um 20%
        if "max_resource" in worst_case and worst_case["max_resource"] is not None:
            worst_case["max_resource"] = int(worst_case["max_resource"] * 0.8)

        # Reduziere transport_capacity um 25%
        if "transport_capacity" in worst_case and worst_case["transport_capacity"] is not None:
            worst_case["transport_capacity"] = int(worst_case["transport_capacity"] * 0.75)

        # Reduziere max_production in jeder Fabrik um 10%
        if "factories" in worst_case:
            for f in worst_case["factories"]:
                if "max_production" in f and f["max_production"] is not None:
                    f["max_production"] = int(f["max_production"] * 0.9)

        return worst_case

    def _create_best_case(self):
        """
        Verringert Nachfrage, erhöht Ressourcen usw.
        """
        best_case = copy.deepcopy(self.data)
        best_case["scenario_name"] = "BestCase"

        # Nachfrage um 25% verringern
        if "demands" in best_case:
            best_case["demands"] = [max(int(d * 0.75), 0) for d in best_case["demands"]]

        # Ressourcen um 30% erhöhen
        if "max_resource" in best_case and best_case["max_resource"] is not None:
            best_case["max_resource"] = int(best_case["max_resource"] * 1.3)

        # Transportkapazität um 50% steigern
        if "transport_capacity" in best_case and best_case["transport_capacity"] is not None:
            best_case["transport_capacity"] = int(best_case["transport_capacity"] * 1.5)

        # max_production jeder Fabrik um 20% steigern
        if "factories" in best_case:
            for f in best_case["factories"]:
                if "max_production" in f and f["max_production"] is not None:
                    f["max_production"] = int(f["max_production"] * 1.2)

        return best_case

    def _create_random_scenario(self, idx):
        """
        Erzeugt ein zufälliges Szenario (z. B. Variation von Nachfrage, Ressourcen, etc.).
        """
        rnd_case = copy.deepcopy(self.data)
        rnd_case["scenario_name"] = f"RandomCase_{idx+1}"

        # Zufälliger Faktor für demands: 0.5 - 2.0 (also bis zu halbe oder doppelte Nachfrage)
        if "demands" in rnd_case:
            factor = random.uniform(0.5, 2.0)
            rnd_case["demands"] = [int(d * factor) for d in rnd_case["demands"]]

        # Zufällige Modifikation für max_resource: ± 30%
        if "max_resource" in rnd_case and rnd_case["max_resource"] is not None:
            # Beispiel: wir erhöhen oder verringern um bis zu 30%
            change_factor = random.uniform(0.7, 1.3)
            rnd_case["max_resource"] = int(rnd_case["max_resource"] * change_factor)

        # Zufällige Modifikation für transport_capacity
        if "transport_capacity" in rnd_case and rnd_case["transport_capacity"] is not None:
            change_factor = random.uniform(0.5, 1.5)
            rnd_case["transport_capacity"] = int(rnd_case["transport_capacity"] * change_factor)

        # Zufällige Modifikation bei jeder Fabrik
        if "factories" in rnd_case:
            for f in rnd_case["factories"]:
                if "max_production" in f and f["max_production"] is not None:
                    # Variation +/- 25%
                    fac_var = random.uniform(0.75, 1.25)
                    f["max_production"] = int(f["max_production"] * fac_var)

        return rnd_case

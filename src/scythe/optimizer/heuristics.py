# plan_optimizer/heuristics.py

import random
import copy

def run_heuristic_if_needed(solution: dict, objective_value: float):
    """
    Entscheidet anhand einfacher Regeln, ob eine Heuristik ausgeführt werden soll.
    Und führt sie ggf. aus. Gibt ein (solution, objective_value)-Tupel zurück.
    """
    if solution is None:
        # Keine Lösung vorhanden - wir können keine Heuristik anwenden
        return solution, objective_value

    # Beispiel-Regel: Wenn die Objective sehr hoch ist, versuchen wir Local Search
    if objective_value > 1000:
        improved_sol, improved_obj = local_search(solution, objective_value)
        if improved_obj < objective_value:
            return improved_sol, improved_obj
        else:
            return solution, objective_value
    else:
        # Alternative: Genetic Algorithm, wenn Objective mittelgroß
        if 500 < objective_value <= 1000:
            improved_sol, improved_obj = genetic_algorithm(solution, objective_value)
            if improved_obj < objective_value:
                return improved_sol, improved_obj
            else:
                return solution, objective_value

    # Wenn keine Bedingung greift, geben wir die Originallösung zurück
    return solution, objective_value


def local_search(solution: dict, objective_value: float, max_iterations: int = 50):
    """
    Sehr vereinfachter Local-Search-Ansatz:
    1. Kopiere die aktuelle Lösung.
    2. Iteriere mehrfach:
       - Wähle zufällig eine Variable.
       - Ändere sie leicht (Delta).
       - Falls die Objective besser wird, übernehme die Änderung.
       - Sonst mache ein Rollback.
    3. Gib die verbesserte (oder unveränderte) Lösung zurück.
    
    Hinweise:
    - In einem echten Planungsproblem müssen wir sicherstellen, 
      dass wir Constraints nicht verletzen. Dies ist hier stark vereinfacht.
    - Wir benötigen eine `calculate_objective(...)`-Funktion, um
      die Auswirkung der Änderung auf die Kosten/Zielfunktion zu beurteilen.
    """
    best_solution = copy.deepcopy(solution)
    best_obj = objective_value

    for _ in range(max_iterations):
        # Wähle zufällig eine Variable aus dem Lösungs-Dict
        var_name = random.choice(list(best_solution.keys()))
        old_value = best_solution[var_name]

        # Wähle ein kleines Delta, z. B. +/- 5%
        delta = old_value * 0.05  
        if delta == 0:
            delta = 1  # falls old_value = 0 war

        # Zufällige Richtung (+ oder -)
        direction = random.choice([+1, -1])
        candidate_value = old_value + direction * delta

        # Setze kurzzeitig den neuen Wert
        best_solution[var_name] = candidate_value

        # Berechne neue Objective
        new_obj = calculate_objective(best_solution)

        # Ist es besser?
        if new_obj < best_obj:
            # Verbesserung akzeptieren
            best_obj = new_obj
        else:
            # Rollback
            best_solution[var_name] = old_value

    return best_solution, best_obj


def genetic_algorithm(solution: dict, objective_value: float, population_size: int = 10, generations: int = 5):
    """
    Ein stark vereinfachter genetischer Algorithmus, der auf einer 
    kleinen Population nahe der aktuellen Lösung arbeitet.

    Ablauf:
    1. Erstelle eine Startpopulation, indem du die gegebene Lösung 
       leicht veränderst (Mutation).
    2. Bewerte jede Lösung (Objective).
    3. Wähle die besten Lösungen aus (Selektion).
    4. Erzeuge neue Lösungen durch 'Crossover' und Mutation.
    5. Wiederhole für 'generations' viele Generationen.
    6. Gib die beste gefundene Lösung zurück.

    Auch hier gilt: Constraints, realistische Mutation/Crossover und 
    exakte Objektivfunktion sind nur als Platzhalter implementiert.
    """
    # Startpopulation
    population = []
    base_sol = copy.deepcopy(solution)
    base_obj = objective_value

    population.append((base_sol, base_obj))

    # Erzeuge weitere zufällige Abkömmlinge
    for _ in range(population_size - 1):
        mutated_sol, mutated_obj = mutate_solution(base_sol)
        population.append((mutated_sol, mutated_obj))

    for gen in range(generations):
        # Sortiere nach bester Objective
        population.sort(key=lambda x: x[1])
        # Selektionsschritt: die besten 50% überleben
        survivors = population[: population_size // 2]

        # Crossover: Erzeuge neuen Nachwuchs, bis wir wieder population_size haben
        offspring = []
        while len(survivors) + len(offspring) < population_size:
            # Wähle zwei zufällige Eltern
            parents = random.sample(survivors, 2)
            child_sol, child_obj = crossover(parents[0][0], parents[1][0])
            # Mutation auf Kind
            child_sol, child_obj = mutate_solution(child_sol, parent_obj=child_obj)
            offspring.append((child_sol, child_obj))

        # Neue Population = Survivors + Offspring
        population = survivors + offspring

    # Am Ende erneut sortieren und beste Lösung zurückgeben
    population.sort(key=lambda x: x[1])
    best_sol, best_obj = population[0]
    return best_sol, best_obj


# --------------------------------------------
# Hilfsfunktionen, die die "Objektivfunktion"
# und einfache (platzhalterhafte) Operationen
# realisieren.
# --------------------------------------------

def calculate_objective(solution: dict) -> float:
    """
    Dummy-Funktion zur Berechnung der Objective:
    - Hier nur eine einfache Summe aller Variablen (als Beispiel).
    - In einer echten Anwendung liest man aus 'solution' 
      die Variablen und berechnet die realen Kosten, 
      z. B. production_cost + transport_cost etc.
    """
    # Summiere einfach die Beträge (Absolutwerte), als "Kosten"
    cost = sum(abs(v) for v in solution.values())
    return cost

def mutate_solution(solution: dict, parent_obj: float = None):
    """
    Nimmt eine bestehende Lösung und ändert sie zufällig leicht.
    Gibt (neue_solution, neue_objective) zurück.
    """
    child = copy.deepcopy(solution)

    # Mutationsstärke
    mutation_strength = 0.1  

    var_name = random.choice(list(child.keys()))
    original = child[var_name]
    delta = original * mutation_strength
    if delta == 0:
        delta = 1.0
    direction = random.choice([+1, -1])
    child[var_name] = original + direction * delta

    new_obj = calculate_objective(child)
    return child, new_obj

def crossover(sol1: dict, sol2: dict):
    """
    Kreuzt zwei Elternlösungen, indem wir die Variablen 
    teils von sol1, teils von sol2 übernehmen.
    """
    child = {}
    for var_name in sol1.keys():
        if random.random() < 0.5:
            child[var_name] = sol1[var_name]
        else:
            child[var_name] = sol2[var_name]
    child_obj = calculate_objective(child)
    return child, child_obj

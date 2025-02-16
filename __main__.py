#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
__main__.py
-----------
Dies ist der Einstiegspunkt für das scythe-Framework. 
Wenn du `python -m scythe` ausführst, wird diese Datei aufgerufen.

Hier kannst du Kommandozeilenargumente parsen und entsprechende 
Aktionen ausführen (z. B. Orchestrator starten, Datenpipelines anstoßen, etc.).
"""

import sys
import argparse

# Angenommen, dein Orchestrator liegt unter scythe.orchestrator.main_orchestrator
from scythe.orchestrator.orchestrator import main as orchestrator_main


def parse_arguments():
    """
    Parst die Kommandozeilenargumente und gibt das resultierende 
    args-Objekt zurück.
    """
    parser = argparse.ArgumentParser(
        description="Scythe: Agent-Based Simulation & Plan-Optimizer Framework"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Pfad zu den Eingabedaten (z. B. CSV-Dateien)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Ausführlichere Konsolenausgaben."
    )
    parser.add_argument(
        "--run_pipeline",
        action="store_true",
        help="Starte die gesamte Pipeline inklusive Transformer-Training & SDM-Demo."
    )
    # Füge hier weitere Argumente hinzu, falls erforderlich

    return parser.parse_args()

def main():
    """
    Hauptfunktion für den Einstiegspunkt 'python -m scythe'.
    """
    args = parse_arguments()

    # Optional: Falls du 'verbose' für Logging etc. verwenden willst
    if args.verbose:
        print("[__main__] Starte Scythe im verbose-Modus...")

    # Beispiel: Wenn man explizit die Pipeline starten soll
    if args.run_pipeline:
        # Rufe deine Orchestrator-Logik auf
        result = orchestrator_main(data_dir=args.data_dir)
        if args.verbose:
            print("[__main__] Pipeline beendet. Ergebnis:", result)
    else:
        # Anderenfalls nur eine kurze Info ausgeben
        print("[__main__] Keine Aktion ausgewählt. Nutze --help für Optionen.")

    return 0  # normaler Exit-Code

if __name__ == "__main__":
    sys.exit(main())

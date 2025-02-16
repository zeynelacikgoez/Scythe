#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
from pathlib import Path

# Importiere die zentrale Konfigurationsklasse und die Pipeline-Funktion
from scythe.orchestrator.config import OrchestratorConfig
from scythe.orchestrator.pipeline import run_entire_pipeline


def parse_arguments():
    """
    Parst die Kommandozeilenargumente und gibt ein args-Objekt zurück.
    """
    parser = argparse.ArgumentParser(
        description="Scythe: Agent-Based Simulation & Plan-Optimizer Framework"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Pfad zu den Eingabedaten (z. B. CSV-Dateien). Überschreibt den Standardpfad in der Konfiguration."
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
    return parser.parse_args()


def main():
    """
    Hauptfunktion für den Einstiegspunkt 'python -m scythe'.
    Nutzt die OrchestratorConfig, um alle relevanten Pfade zu setzen,
    und startet optional die gesamte Pipeline.
    """
    args = parse_arguments()
    verbose = args.verbose

    # Erzeuge die Konfigurationsinstanz (setzt standardmäßig relative Pfade relativ zum Projekt)
    config = OrchestratorConfig()

    # Falls ein data_dir als Argument übergeben wurde, diesen Wert übernehmen
    if args.data_dir:
        config.data_dir = Path(args.data_dir).resolve()

    if verbose:
        print("[main_orchestrator] Starte Scythe im verbose-Modus...")
        print("[main_orchestrator] Aktuelle Konfiguration:")
        print(config)

    # Falls die Pipeline gestartet werden soll, rufe run_entire_pipeline auf
    if args.run_pipeline:
        # Übergebe den (absoluten) Datenpfad als String an die Pipeline-Funktion
        result = run_entire_pipeline(str(config.data_dir))
        if verbose:
            print("[main_orchestrator] Pipeline beendet. Ergebnis:")
            print(result)
    else:
        print("[main_orchestrator] Keine Aktion ausgewählt. Nutze --help für Optionen.")

    return 0  # normaler Exit-Code


if __name__ == "__main__":
    sys.exit(main())

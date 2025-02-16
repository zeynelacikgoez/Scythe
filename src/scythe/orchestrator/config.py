from pathlib import Path
import os

class OrchestratorConfig:
    """
    OrchestratorConfig:
    -------------------
    Zentralisiert alle Konfigurationsparameter für den Pipeline-Prozess:
      - Pfade für Daten, Outputs und Checkpoints (relativ zum Projekt-Root)
      - Parameter für SDM, Hyperbolic Transformer und Plan Optimizer
      - Weitere Einstellungen (z. B. Debug-Modus)

    Die Pfade und einige Parameter können über Umgebungsvariablen überschrieben werden.
    """

    def __init__(self, project_root: str = None):
        # Bestimme das Projekt-Root: Falls nicht explizit angegeben, gehe davon aus,
        # dass config.py in src/scythe/orchestrator liegt und das Root 3 Ebenen darüber liegt.
        if project_root is None:
            project_root = Path(__file__).resolve().parent.parent.parent
        else:
            project_root = Path(project_root)
        self.project_root = project_root

        # Setze Standardpfade (können über Umgebungsvariablen überschrieben werden)
        self.data_dir = Path(os.getenv("SCYTHE_DATA_DIR", self.project_root / "data"))
        self.output_dir = Path(os.getenv("SCYTHE_OUTPUT_DIR", self.project_root / "outputs"))
        self.checkpoints_dir = Path(os.getenv("SCYTHE_CHECKPOINTS_DIR", self.project_root / "checkpoints"))

        # Dateinamen für Trainings- und Validierungsdaten
        self.train_data_file = os.getenv("SCYTHE_TRAIN_DATA_FILE", "train.csv")
        self.val_data_file = os.getenv("SCYTHE_VAL_DATA_FILE", "val.csv")

        # Erstelle die nötigen Verzeichnisse, falls sie noch nicht existieren
        self.ensure_directories()

        # ========== SDM-Parameter ==========
        self.sdm_address_size = 256
        self.sdm_memory_size = 1000
        self.sdm_activation_percent = 0.1
        self.sdm_init_mode = "random"

        # ========== Hyperbolic Transformer Parameter ==========
        self.ht_input_dim = 256
        self.ht_hidden_dim = 512
        self.ht_output_dim = 256
        self.ht_curvature_init = 1.0
        self.ht_num_layers = 2
        self.ht_num_heads = 8
        self.ht_ff_dim = 2048
        self.ht_dropout = 0.1
        self.ht_use_pos_enc = True
        self.ht_epochs = 2
        self.ht_use_gpu = os.getenv("SCYTHE_HT_USE_GPU", "True").lower() in ["true", "1", "yes"]

        # ========== Plan Optimizer Parameter ==========
        self.plan_solver_type = os.getenv("SCYTHE_PLAN_SOLVER_TYPE", "ortools")
        self.plan_use_heuristics = os.getenv("SCYTHE_PLAN_USE_HEURISTICS", "False").lower() in ["true", "1", "yes"]

        # ========== Weitere Einstellungen ==========
        self.debug_mode = os.getenv("SCYTHE_DEBUG_MODE", "True").lower() in ["true", "1", "yes"]

    def ensure_directories(self):
        """Erstellt die für Output und Checkpoints erforderlichen Verzeichnisse."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        return (
            f"OrchestratorConfig(\n"
            f"  project_root='{self.project_root}',\n"
            f"  data_dir='{self.data_dir}',\n"
            f"  output_dir='{self.output_dir}',\n"
            f"  checkpoints_dir='{self.checkpoints_dir}',\n"
            f"  train_data_file='{self.train_data_file}',\n"
            f"  val_data_file='{self.val_data_file}',\n\n"
            f"  sdm_address_size={self.sdm_address_size}, sdm_memory_size={self.sdm_memory_size},\n"
            f"  sdm_activation_percent={self.sdm_activation_percent}, sdm_init_mode='{self.sdm_init_mode}',\n\n"
            f"  ht_input_dim={self.ht_input_dim}, ht_hidden_dim={self.ht_hidden_dim},\n"
            f"  ht_output_dim={self.ht_output_dim}, ht_curvature_init={self.ht_curvature_init},\n"
            f"  ht_num_layers={self.ht_num_layers}, ht_num_heads={self.ht_num_heads},\n"
            f"  ht_ff_dim={self.ht_ff_dim}, ht_dropout={self.ht_dropout}, ht_use_pos_enc={self.ht_use_pos_enc},\n"
            f"  ht_epochs={self.ht_epochs}, ht_use_gpu={self.ht_use_gpu},\n\n"
            f"  plan_solver_type='{self.plan_solver_type}', plan_use_heuristics={self.plan_use_heuristics},\n"
            f"  debug_mode={self.debug_mode}\n"
            f")"
        )

if __name__ == "__main__":
    config = OrchestratorConfig()
    print(config)

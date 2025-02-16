from pathlib import Path
from typing import Optional
import torch

from scythe.transformer.config import HyperConfig
from scythe.transformer.model import HyperbolicTransformer

def run_inference(
    input_data: torch.Tensor, 
    checkpoint_path: str, 
    map_location: Optional[str] = None
) -> torch.Tensor:
    """
    Lädt ein trainiertes HyperbolicTransformer-Modell und führt einen Forward-Pass
    auf den Eingabedaten aus.

    Parameter:
    ----------
    input_data : torch.Tensor
        Eingabedaten, z. B. in der Form [batch_size, seq_len, input_dim].
    checkpoint_path : str
        Pfad zur gespeicherten Checkpoint-Datei, z. B. "checkpoints/model_epoch10.pt".
    map_location : Optional[str]
        Falls das Modell auf einem anderen Gerät (z. B. CPU) geladen werden soll,
        kann hier z. B. "cpu" angegeben werden.

    Rückgabe:
    ---------
    torch.Tensor
        Die Ausgabe des Modells, z. B. in der Form [batch_size, seq_len, output_dim].
    """
    # 1) Konfiguration laden
    cfg = HyperConfig()

    # 2) Modell instanzieren anhand der Konfiguration
    model = HyperbolicTransformer(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.output_dim,
        curvature=cfg.curvature_init,
        num_layers=getattr(cfg, "num_layers", 2),
        num_heads=getattr(cfg, "num_heads", 8),
        ff_dim=getattr(cfg, "ff_dim", 2048),
        dropout=getattr(cfg, "dropout", 0.1),
        use_pos_enc=getattr(cfg, "use_pos_enc", True)
    )

    # 3) Checkpoint-Pfad als Path-Objekt verwenden und prüfen
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"[inference] Checkpoint not found: {checkpoint_file}")

    # 4) Checkpoint laden
    checkpoint = torch.load(str(checkpoint_file), map_location=map_location)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[inference] Loaded model_state_dict from {checkpoint_file}")
    else:
        model.load_state_dict(checkpoint)
        print(f"[inference] Loaded model directly from {checkpoint_file}")

    # 5) Modell in den Evaluierungsmodus setzen und auf das passende Gerät verschieben
    model.eval()
    device = torch.device("cuda" if (cfg.use_gpu and torch.cuda.is_available()) else "cpu")
    model.to(device)

    # 6) Eingabedaten auf das Gerät verschieben
    input_data = input_data.to(device)

    # 7) Forward-Pass ohne Gradientenberechnung durchführen
    with torch.no_grad():
        outputs = model(input_data)

    return outputs

def run_inference_example():
    """
    Beispiel-Funktion, die zeigt, wie man 'run_inference' aufrufen kann.
    """
    batch_size = 2
    seq_len = 10
    input_dim = 300  # muss zu cfg.input_dim passen
    dummy_input = torch.randn(batch_size, seq_len, input_dim)

    # Checkpoint-Pfad – idealerweise relativ zum Projektverzeichnis
    checkpoint_path = str(Path("checkpoints") / "model_epoch10.pt")

    outputs = run_inference(dummy_input, checkpoint_path, map_location="cpu")
    print("[inference_example] Output shape:", outputs.shape)
    print("[inference_example] Outputs:", outputs)

if __name__ == "__main__":
    run_inference_example()

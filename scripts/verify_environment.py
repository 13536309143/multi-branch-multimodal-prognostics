"""Check installed versions and CPU/CUDA matrix operations without training."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mbdnn.utils import environment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with torch.no_grad():
    matrix = torch.eye(4, device=device)
    assert torch.equal(matrix @ matrix, matrix)
print(json.dumps({**environment(), "selected_device": str(device),
                  "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                  "matrix_operation_verified": True, "optimizer_updates": 0}, indent=2))

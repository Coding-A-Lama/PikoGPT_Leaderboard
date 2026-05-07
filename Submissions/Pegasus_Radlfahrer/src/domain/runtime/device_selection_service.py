import torch


class DeviceSelectionService:
    """Selects the best available torch device with CUDA > MPS > CPU priority."""

    def select_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

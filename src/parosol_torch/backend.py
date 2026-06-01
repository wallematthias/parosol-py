from __future__ import annotations

from dataclasses import dataclass
from importlib import util
from typing import Any


@dataclass(frozen=True)
class TorchBackendInfo:
    """Runtime capabilities for the optional torch backend package."""

    torch_installed: bool
    torch_version: str | None
    mps_available: bool
    cuda_available: bool
    default_device: str | None
    status: str

    @property
    def available(self) -> bool:
        """Return true when any accelerated torch device is available."""

        return self.mps_available or self.cuda_available


def backend_info() -> TorchBackendInfo:
    """Inspect optional torch acceleration without requiring torch at import time."""

    if util.find_spec("torch") is None:
        return TorchBackendInfo(
            torch_installed=False,
            torch_version=None,
            mps_available=False,
            cuda_available=False,
            default_device=None,
            status="torch is not installed; install parosol-py[torch]",
        )

    import torch

    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    cuda_available = bool(torch.cuda.is_available())
    default_device = _default_device(
        mps_available=mps_available,
        cuda_available=cuda_available,
    )
    status = (
        f"torch {torch.__version__} available on {default_device}"
        if default_device is not None
        else f"torch {torch.__version__} installed but no accelerator is available"
    )
    return TorchBackendInfo(
        torch_installed=True,
        torch_version=str(torch.__version__),
        mps_available=mps_available,
        cuda_available=cuda_available,
        default_device=default_device,
        status=status,
    )


def is_available(device: str | None = None) -> bool:
    """Return whether the requested torch device is available."""

    info = backend_info()
    if device is None:
        return info.available
    normalized = device.strip().lower()
    if normalized == "mps":
        return info.mps_available
    if normalized == "cuda":
        return info.cuda_available
    if normalized == "cpu":
        return info.torch_installed
    return False


def require_backend(device: str | None = None) -> Any:
    """Import torch and validate a device for future solver code.

    The numerical GPU backend is intentionally not implemented here yet. This
    helper exists so callers and tests can fail early with a clear message.
    """

    info = backend_info()
    if not info.torch_installed:
        raise RuntimeError(info.status)
    if device is not None and not is_available(device):
        raise RuntimeError(f"torch device '{device}' is not available: {info.status}")
    import torch

    return torch


def _default_device(*, mps_available: bool, cuda_available: bool) -> str | None:
    if mps_available:
        return "mps"
    if cuda_available:
        return "cuda"
    return None

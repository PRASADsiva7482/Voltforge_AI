"""Production Gen1 decoder architecture for the VoltForge-owned language model."""

from .config import (
    ARCHITECTURE_ID,
    DEFAULT_ALLOCATION_LIMIT,
    Gen1Config,
    Gen1ConfigError,
    ParameterAllocationError,
)
from .model import (
    CheckpointContractError,
    Gen1Output,
    KVCache,
    VoltForgeGen1,
)

__all__ = [
    "ARCHITECTURE_ID",
    "DEFAULT_ALLOCATION_LIMIT",
    "CheckpointContractError",
    "Gen1Config",
    "Gen1ConfigError",
    "Gen1Output",
    "KVCache",
    "ParameterAllocationError",
    "VoltForgeGen1",
]

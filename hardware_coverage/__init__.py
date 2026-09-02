"""Project-owned hardware coverage reporting for the VoltForge AI boundary."""

from hardware_coverage.component_service import (
    get_component_coverage,
    validate_component_coverage,
)
from hardware_coverage.service import get_hardware_coverage, validate_hardware_coverage

__all__ = [
    "get_component_coverage",
    "get_hardware_coverage",
    "validate_component_coverage",
    "validate_hardware_coverage",
]

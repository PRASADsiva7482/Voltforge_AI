"""Deployment continuity contracts and content-free verification helpers."""

from operations.continuity import (
    ContinuityError,
    build_registry_source_inventory,
    build_restore_receipt,
    validate_inventory,
    verify_backup_receipt,
    verify_owner_objectives_receipt,
    verify_restored_tree,
    verify_restore_receipt,
    verify_rotation_receipt,
)

__all__ = [
    "ContinuityError",
    "build_registry_source_inventory",
    "build_restore_receipt",
    "validate_inventory",
    "verify_backup_receipt",
    "verify_owner_objectives_receipt",
    "verify_restored_tree",
    "verify_restore_receipt",
    "verify_rotation_receipt",
]

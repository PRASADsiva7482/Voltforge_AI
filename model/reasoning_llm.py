"""Legacy import alias for deterministic tools; this module is not an LLM.

New callers use engine.deterministic_tools. Original unverified templates and
hardware tables are retained only in the LLM-TASK-016 source snapshot.
"""
from engine.deterministic_tools import DeterministicElectronicsTools

ElectronicsReasoningEngine = DeterministicElectronicsTools

__all__ = ["ElectronicsReasoningEngine"]

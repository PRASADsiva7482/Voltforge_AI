"""Versioned, provenance-bound local electronics knowledge."""

from electronics_corpus.schema import (
    CORPUS_CONTRACT_VERSION,
    KnowledgeContractError,
    KnowledgeRecord,
    validate_knowledge_record,
)
from electronics_corpus.store import ElectronicsCorpus, LookupResult, get_electronics_corpus

__all__ = [
    "CORPUS_CONTRACT_VERSION",
    "ElectronicsCorpus",
    "KnowledgeContractError",
    "KnowledgeRecord",
    "LookupResult",
    "get_electronics_corpus",
    "validate_knowledge_record",
]

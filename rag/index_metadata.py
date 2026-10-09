"""Contrato de metadados comum aos índices local e remoto."""

from rag.splitter import CHUNKING_VERSION

METADATA_FIELDS = (
    "kind", "source", "origin_group", "title", "url", "published_at",
    "natureza", "license", "dataset_name", "conclusive_eligible", "chunking_version",
)


def index_metadata(document: dict) -> dict:
    metadata = {field: document.get(field, "") for field in METADATA_FIELDS}
    # Apenas o booleano explicitamente revisado qualifica uma fonte.
    metadata["conclusive_eligible"] = document.get("conclusive_eligible") is True
    metadata["chunking_version"] = CHUNKING_VERSION
    return metadata

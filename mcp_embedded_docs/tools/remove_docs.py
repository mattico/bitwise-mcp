"""Remove documents tool."""

from typing import Optional

from ..config import Config


def remove_docs(doc_id: str, config: Optional[Config] = None) -> str:
    """Remove a document's chunks and vectors from the index.

    Blocking and cheap: no search engine or embedding model is loaded.

    Args:
        doc_id: Document ID to remove (a filename as listed by list_docs is
            also accepted)
        config: Configuration object

    Returns:
        Status message as markdown
    """
    from ..ingestion.pipeline import remove_document

    if config is None:
        config = Config.load()

    report = remove_document(doc_id, config)
    if report is None:
        other_id = _doc_id_for_filename(doc_id, config)
        if other_id:
            report = remove_document(other_id, config)
    if report is None:
        return f"❌ Error: Document not found: {doc_id}"

    return (
        f"✅ Removed {report.filename} (ID: `{report.doc_id}`): "
        f"{report.chunks} chunks, {report.vectors} vectors"
    )


def _doc_id_for_filename(name: str, config: Config) -> Optional[str]:
    """Id of the indexed document whose filename matches `name` (case-insensitive)."""
    from ..indexing.metadata_store import MetadataStore

    db_path = config.index.directory / config.index.metadata_db
    if not db_path.exists():
        return None
    store = MetadataStore(db_path)
    try:
        for doc in store.list_documents():
            if (doc.get("filename") or "").lower() == name.lower():
                return doc["id"]
    finally:
        store.close()
    return None

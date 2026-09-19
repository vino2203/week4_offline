import re


def sanitize_label(label: str) -> str:
    """LLM-extracted labels can contain spaces/slashes that aren't valid Neo4j identifiers."""
    cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", label.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "Entity"
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


def sanitize_graph_documents(graph_documents):
    for doc in graph_documents:
        for node in doc.nodes:
            node.type = sanitize_label(node.type)
        for rel in doc.relationships:
            rel.type = sanitize_label(rel.type)
            rel.source.type = sanitize_label(rel.source.type)
            rel.target.type = sanitize_label(rel.target.type)
    return graph_documents

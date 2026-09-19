from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document

from graph_utils import sanitize_graph_documents, sanitize_label


def test_sanitize_label_removes_special_chars():
    assert sanitize_label("Ci/cd tool") == "Ci_cd_tool"


def test_sanitize_label_collapses_repeated_separators():
    assert sanitize_label("a   b//c") == "a_b_c"


def test_sanitize_label_handles_leading_digit():
    assert sanitize_label("123abc") == "_123abc"


def test_sanitize_label_handles_empty_string():
    assert sanitize_label("   ") == "Entity"


def test_sanitize_label_handles_only_special_chars():
    assert sanitize_label("///") == "Entity"


def test_sanitize_label_leaves_clean_label_untouched():
    assert sanitize_label("Service") == "Service"


def test_sanitize_graph_documents_cleans_node_and_relationship_types():
    source = Node(id="a", type="Ci/cd tool")
    target = Node(id="b", type="Monitoring tool")
    relationship = Relationship(source=source, target=target, type="depends/on")
    graph_document = GraphDocument(
        nodes=[source, target],
        relationships=[relationship],
        source=Document(page_content="irrelevant"),
    )

    sanitize_graph_documents([graph_document])

    assert source.type == "Ci_cd_tool"
    assert target.type == "Monitoring_tool"
    assert relationship.type == "depends_on"
    assert relationship.source.type == "Ci_cd_tool"
    assert relationship.target.type == "Monitoring_tool"

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from graph_utils import sanitize_graph_documents

BASE_DIR = Path(__file__).parent
PDF_PATH = BASE_DIR / "spotify_web_app_architecture.pdf"


def main() -> None:
    load_dotenv(BASE_DIR / ".env")

    loader = PyPDFLoader(str(PDF_PATH))
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    graph_transformer = LLMGraphTransformer(llm=llm)
    graph_documents = graph_transformer.convert_to_graph_documents(chunks)
    sanitize_graph_documents(graph_documents)

    graph = Neo4jGraph(
        url=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        database=os.environ.get("NEO4J_DATABASE"),
    )
    graph.query("MATCH (n) DETACH DELETE n")
    graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)

    node_count = graph.query("MATCH (n) RETURN count(n) AS count")[0]["count"]
    rel_count = graph.query("MATCH ()-[r]->() RETURN count(r) AS count")[0]["count"]

    print(f"Loaded {len(pages)} page(s) from {PDF_PATH.name}")
    print(f"Split into {len(chunks)} chunk(s)")
    print(f"Extracted {len(graph_documents)} graph document(s)")
    print(f"Neo4j now has {node_count} node(s) and {rel_count} relationship(s)")


if __name__ == "__main__":
    main()

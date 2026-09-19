import os
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from graph_utils import sanitize_graph_documents

BASE_DIR = Path(__file__).parent
PDF_PATH = BASE_DIR / "spotify_web_app_architecture.pdf"
INDEX_DIR = BASE_DIR / "faiss_index"


def ingest_vector(pages, embeddings, on_progress: Optional[Callable[[str], None]] = None) -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    for page in pages:
        page_number = page.metadata["page"] + 1
        chunks = splitter.split_documents([page])

        vectorstore = FAISS.from_documents(chunks, embeddings)
        page_dir = INDEX_DIR / f"page_{page_number}"
        vectorstore.save_local(str(page_dir))

        message = f"[vector] Page {page_number}: {len(chunks)} chunk(s) -> {page_dir}"
        print(message)
        if on_progress:
            on_progress(message)


def ingest_graph(pages, llm, on_progress: Optional[Callable[[str], None]] = None) -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)

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

    message_1 = f"[graph] Extracted {len(graph_documents)} graph document(s) from {len(chunks)} chunk(s)"
    message_2 = f"[graph] Neo4j now has {node_count} node(s) and {rel_count} relationship(s)"
    print(message_1)
    print(message_2)
    if on_progress:
        on_progress(message_1)
        on_progress(message_2)


def main() -> None:
    load_dotenv(BASE_DIR / ".env")

    loader = PyPDFLoader(str(PDF_PATH))
    pages = loader.load()
    print(f"Loaded {len(pages)} page(s) from {PDF_PATH.name}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    ingest_vector(pages, embeddings)
    ingest_graph(pages, llm)


if __name__ == "__main__":
    main()

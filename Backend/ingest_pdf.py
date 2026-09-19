from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).parent
PDF_PATH = BASE_DIR / "spotify_web_app_architecture.pdf"
INDEX_DIR = BASE_DIR / "faiss_index"


def main() -> None:
    load_dotenv(BASE_DIR / ".env")

    loader = PyPDFLoader(str(PDF_PATH))
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print(f"Loaded {len(pages)} page(s) from {PDF_PATH.name}")

    for page in pages:
        page_number = page.metadata["page"] + 1
        chunks = splitter.split_documents([page])

        vectorstore = FAISS.from_documents(chunks, embeddings)
        page_dir = INDEX_DIR / f"page_{page_number}"
        vectorstore.save_local(str(page_dir))

        print(f"Page {page_number}: {len(chunks)} chunk(s) -> {page_dir}")


if __name__ == "__main__":
    main()

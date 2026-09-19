import os
import sys
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent
INDEX_DIR = BASE_DIR / "faiss_index"

MAX_RETRIEVAL_ATTEMPTS = 2
MAX_GENERATION_ATTEMPTS = 2
RETRIEVAL_CANDIDATES_K = 8
RERANK_TOP_K = 4
OUT_OF_SCOPE_MESSAGE = "out of scope, Please ask as per the document"

SYNTHESIS_PROMPT = ChatPromptTemplate.from_template(
    "Answer the question using both sources of context below. If a source has "
    "nothing relevant, ignore it.\n\n"
    "Vector search results (semantic text chunks, with page numbers):\n{vector_context}\n\n"
    "Knowledge graph findings (entity relationships):\n{graph_context}\n\n"
    "Question: {question}\n\nAnswer:"
)

# Entity names were auto-extracted by an LLM, so casing/wording in the question
# rarely matches node ids exactly (e.g. "API Gateway" vs "Api Gateway"). Force
# case-insensitive, partial-string matching instead of the default's exact `=`.
CYPHER_GENERATION_PROMPT = PromptTemplate.from_template(
    "Task: Generate a Cypher statement to query a graph database.\n"
    "Instructions:\n"
    "Use only the provided relationship types and properties in the schema.\n"
    "Do not use any other relationship types or properties that are not provided.\n"
    "Entity ids were extracted by an LLM, so never match them with exact equality (=). "
    "Always match node ids case-insensitively and by partial string, e.g. "
    "WHERE toLower(n.id) CONTAINS toLower('search term').\n"
    "Schema:\n{schema}\n"
    "Note: Do not include any explanations or apologies in your responses.\n"
    "Do not respond to any questions that might ask anything else than for you to "
    "construct a Cypher statement.\n"
    "Do not include any text except the generated Cypher statement.\n\n"
    "The question is:\n{question}"
)

DOCUMENT_GRADE_PROMPT = ChatPromptTemplate.from_template(
    "You are grading whether retrieved context is relevant to a user question.\n\n"
    "Retrieved context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Give a binary score 'yes' or 'no'. 'yes' means the context contains information "
    "that helps answer the question."
)

HALLUCINATION_GRADE_PROMPT = ChatPromptTemplate.from_template(
    "You are grading whether an answer is grounded in / supported by the given context.\n\n"
    "Context:\n{context}\n\n"
    "Answer:\n{answer}\n\n"
    "Give a binary score 'yes' or 'no'. 'yes' means every claim in the answer is "
    "supported by the context (no invented facts)."
)

ANSWER_GRADE_PROMPT = ChatPromptTemplate.from_template(
    "You are grading whether an answer actually addresses a user question.\n\n"
    "Question: {question}\n\n"
    "Answer:\n{answer}\n\n"
    "Give a binary score 'yes' or 'no'. 'yes' means the answer resolves the question."
)

TRANSFORM_QUERY_PROMPT = ChatPromptTemplate.from_template(
    "The following search query did not retrieve useful results from the ingested "
    "reference document. Rewrite it as a clearer, more specific search query that "
    "is more likely to retrieve relevant content. Return only the rewritten query, "
    "nothing else.\n\n"
    "Original query: {question}"
)

RERANK_PROMPT = ChatPromptTemplate.from_template(
    "Given the question and the numbered candidate text chunks below (retrieved via a "
    "mix of keyword and semantic search), return the indices of the {top_k} most "
    "relevant chunks, ordered from most to least relevant.\n\n"
    "Question: {question}\n\n"
    "Candidates:\n{candidates}"
)

GUARDRAILS_PROMPT = ChatPromptTemplate.from_template(
    "You are a scope guardrail for a Q&A assistant that only answers questions about "
    "the content of a specific ingested reference document.\n\n"
    "Question: {question}\n\n"
    "Answer: {answer}\n\n"
    "Give a binary score 'yes' or 'no' for out_of_scope. 'yes' means the question "
    "and/or answer is NOT about the ingested document's content (e.g. general "
    "knowledge, unrelated topics, small talk). 'no' means it is on-topic."
)


class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Documents are relevant to the question")


class GradeHallucination(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Answer is grounded in the context")


class GradeAnswer(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Answer addresses the question")


class GuardrailsScope(BaseModel):
    out_of_scope: Literal["yes", "no"] = Field(description="Question/answer is out of the document's scope")


class RerankedChunks(BaseModel):
    ranked_indices: list[int] = Field(description="Indices of the most relevant chunks, most relevant first")


class GraphState(TypedDict):
    original_question: str
    question: str
    vector_context: str
    graph_context: str
    answer: str
    retrieval_attempts: int
    generation_attempts: int
    generation_grade: str
    out_of_scope: bool


def load_merged_vectorstore(embeddings: OpenAIEmbeddings) -> FAISS:
    page_dirs = sorted(INDEX_DIR.glob("page_*"), key=lambda p: int(p.name.split("_")[1]))
    if not page_dirs:
        raise FileNotFoundError(f"No per-page FAISS indexes found in {INDEX_DIR}")

    vectorstore = FAISS.load_local(str(page_dirs[0]), embeddings, allow_dangerous_deserialization=True)
    for page_dir in page_dirs[1:]:
        vectorstore.merge_from(
            FAISS.load_local(str(page_dir), embeddings, allow_dangerous_deserialization=True)
        )
    return vectorstore


def build_hybrid_retriever(vectorstore: FAISS) -> EnsembleRetriever:
    docs = list(vectorstore.docstore._dict.values())

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = RETRIEVAL_CANDIDATES_K

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_CANDIDATES_K})

    return EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])


def build_graph(llm: ChatOpenAI, embeddings: OpenAIEmbeddings):
    vectorstore = load_merged_vectorstore(embeddings)
    hybrid_retriever = build_hybrid_retriever(vectorstore)

    neo4j_graph = Neo4jGraph(
        url=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        database=os.environ.get("NEO4J_DATABASE"),
    )
    cypher_chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=neo4j_graph,
        verbose=False,
        allow_dangerous_requests=True,
        validate_cypher=True,
        cypher_prompt=CYPHER_GENERATION_PROMPT,
    )

    document_grader = DOCUMENT_GRADE_PROMPT | llm.with_structured_output(GradeDocuments)
    hallucination_grader = HALLUCINATION_GRADE_PROMPT | llm.with_structured_output(GradeHallucination)
    answer_grader = ANSWER_GRADE_PROMPT | llm.with_structured_output(GradeAnswer)
    query_transformer = TRANSFORM_QUERY_PROMPT | llm
    synthesis_chain = SYNTHESIS_PROMPT | llm
    guardrails_grader = GUARDRAILS_PROMPT | llm.with_structured_output(GuardrailsScope)
    reranker = RERANK_PROMPT | llm.with_structured_output(RerankedChunks)

    def rerank_documents(question: str, documents: list, top_k: int) -> list:
        if not documents:
            return documents
        numbered = "\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(documents))
        try:
            result = reranker.invoke({"question": question, "candidates": numbered, "top_k": top_k})
            picked = [documents[i] for i in result.ranked_indices if 0 <= i < len(documents)]
            if picked:
                return picked[:top_k]
        except Exception:
            pass
        return documents[:top_k]

    def retrieve(state: GraphState) -> dict:
        candidates = hybrid_retriever.invoke(state["question"])
        vector_hits = rerank_documents(state["question"], candidates, RERANK_TOP_K)
        vector_context = "\n\n".join(
            f"[Page {doc.metadata.get('page', 0) + 1}] {doc.page_content}" for doc in vector_hits
        )
        try:
            graph_context = cypher_chain.invoke({"query": state["question"]})["result"]
        except Exception:
            # The LLM sometimes can't produce valid Cypher for a question that has
            # nothing to do with the graph's schema (e.g. fully off-topic questions
            # the guardrails check is meant to catch) — fail soft, not crash.
            graph_context = "No relevant graph information found."
        return {"vector_context": vector_context, "graph_context": graph_context}

    def grade_documents(state: GraphState) -> dict:
        context = f"{state['vector_context']}\n\n{state['graph_context']}"
        grade = document_grader.invoke({"context": context, "question": state["original_question"]})
        return {"generation_grade": "relevant" if grade.binary_score == "yes" else "irrelevant"}

    def route_after_document_grade(state: GraphState) -> str:
        if state["generation_grade"] == "relevant" or state["retrieval_attempts"] >= MAX_RETRIEVAL_ATTEMPTS:
            return "generate"
        return "transform_query"

    def transform_query(state: GraphState) -> dict:
        rewritten = query_transformer.invoke({"question": state["original_question"]}).content
        return {"question": rewritten, "retrieval_attempts": state["retrieval_attempts"] + 1}

    def generate(state: GraphState) -> dict:
        answer = synthesis_chain.invoke(
            {
                "vector_context": state["vector_context"],
                "graph_context": state["graph_context"],
                "question": state["original_question"],
            }
        ).content
        return {"answer": answer, "generation_attempts": state["generation_attempts"] + 1}

    def grade_generation(state: GraphState) -> dict:
        context = f"{state['vector_context']}\n\n{state['graph_context']}"
        hallucination = hallucination_grader.invoke({"context": context, "answer": state["answer"]})
        if hallucination.binary_score == "no":
            return {"generation_grade": "not_supported"}

        usefulness = answer_grader.invoke({"question": state["original_question"], "answer": state["answer"]})
        if usefulness.binary_score == "no":
            return {"generation_grade": "not_useful"}

        return {"generation_grade": "useful"}

    def route_after_generation_grade(state: GraphState) -> str:
        grade = state["generation_grade"]
        if grade == "useful":
            return "guardrails_check"
        if grade == "not_supported" and state["generation_attempts"] < MAX_GENERATION_ATTEMPTS:
            return "generate"
        if grade == "not_useful" and state["retrieval_attempts"] < MAX_RETRIEVAL_ATTEMPTS:
            return "transform_query"
        return "guardrails_check"

    def guardrails_check(state: GraphState) -> dict:
        scope = guardrails_grader.invoke({"question": state["original_question"], "answer": state["answer"]})
        if scope.out_of_scope == "yes":
            return {"answer": OUT_OF_SCOPE_MESSAGE, "out_of_scope": True}
        return {"out_of_scope": False}

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("generate", generate)
    workflow.add_node("grade_generation", grade_generation)
    workflow.add_node("guardrails_check", guardrails_check)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_document_grade,
        {"generate": "generate", "transform_query": "transform_query"},
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", "grade_generation")
    workflow.add_conditional_edges(
        "grade_generation",
        route_after_generation_grade,
        {"generate": "generate", "transform_query": "transform_query", "guardrails_check": "guardrails_check"},
    )
    workflow.add_edge("guardrails_check", END)

    return workflow.compile()


def build_self_rag_app():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return build_graph(llm, embeddings)


def run_self_rag(question: str, app=None) -> GraphState:
    if app is None:
        app = build_self_rag_app()

    initial_state: GraphState = {
        "original_question": question,
        "question": question,
        "vector_context": "",
        "graph_context": "",
        "answer": "",
        "retrieval_attempts": 0,
        "generation_attempts": 0,
        "generation_grade": "",
        "out_of_scope": False,
    }
    return app.invoke(initial_state)


def main() -> None:
    load_dotenv(BASE_DIR / ".env")

    question = " ".join(sys.argv[1:]) or "What is the architecture of the Spotify web app?"
    result = run_self_rag(question)

    print(f"Question: {question}\n")
    print("=== Vector search context ===")
    print(result["vector_context"])
    print("\n=== Graph findings ===")
    print(result["graph_context"])
    print(f"\n=== Self-RAG trace ===")
    print(f"Retrieval attempts (rewrites): {result['retrieval_attempts']}")
    print(f"Generation attempts: {result['generation_attempts']}")
    print(f"Final generation grade: {result['generation_grade']}")
    print(f"Out of scope (guardrails): {result['out_of_scope']}")
    print("\n=== Final answer ===")
    print(result["answer"])


if __name__ == "__main__":
    main()

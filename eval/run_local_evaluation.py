from __future__ import annotations

# ruff: noqa: E402 -- repository path and isolated settings must precede src imports.

import asyncio
import json
import os
import tempfile
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./evaluation.db")
os.environ.setdefault("ACCESS_TOKEN_SECRET_KEY", "evaluation-only-access")
os.environ.setdefault("REFRESH_TOKEN_SECRET_KEY", "evaluation-only-refresh")
os.environ.setdefault("RESET_PASSWORD_TOKEN_SECRET", "evaluation-only-reset")
os.environ.setdefault("VERIFICATION_TOKEN_SECRET", "evaluation-only-verify")

from src.agents.langgraph_rag import LangGraphRAGService
from src.infrastructure.keyword import SQLiteBM25Index
from src.infrastructure.llm.local_embeddings import LocalHashEmbeddingProvider
from src.infrastructure.llm.local_llm import LocalExtractiveLLM
from src.infrastructure.reranker import LocalCrossEncoderReranker
from src.infrastructure.vector_db.chroma_vectorstore import ChromaVectorStore
from src.rag.models import RAGChunk, RetrievalFilter
from src.rag.pipeline import RAGRetrievalService
from src.shared.interfaces.llm import ChatMessage, MessageRole

TOP_K = 2

def load_cases() -> list[dict]:
    path = ROOT / "eval" / "datasets" / "support-eval.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def corpus() -> list[RAGChunk]:
    documents = {
        "release-3.2": ("nebula-3.2-release-notes.md", "release_note"),
        "ticket-1842": ("incident-ticket-1842.md", "ticket"),
        "timeout-sop": ("connection-timeout-sop.md", "sop"),
    }
    chunks = []
    for doc_id, (name, document_type) in documents.items():
        text = (ROOT / "docs" / "examples" / name).read_text(encoding="utf-8")
        chunks.append(RAGChunk(
            doc_id=doc_id, chunk_id="chunk-0", source=name, text=text, page_number=1,
            document_type=document_type, product_name="Nebula Gateway", product_version="3.2",
            department="payments", knowledge_space_id="nebula-support", visibility="workspace",
        ))
    return chunks


def metrics(results: list[str], expected: list[str], k: int) -> dict[str, float]:
    top = results[:k]
    if not expected:
        return {"hit_at_k": 1.0 if not top else 0.0, "recall_at_k": 1.0, "mrr": 0.0}
    hits = [doc_id for doc_id in top if doc_id in expected]
    first = next((i for i, doc_id in enumerate(top, 1) if doc_id in expected), None)
    return {
        "hit_at_k": float(bool(hits)),
        "recall_at_k": len(set(hits)) / len(set(expected)),
        "mrr": 1.0 / first if first else 0.0,
    }


async def main() -> None:
    cases = load_cases()
    chunks = corpus()
    embeddings = LocalHashEmbeddingProvider()
    with tempfile.TemporaryDirectory(prefix="enterprise-support-eval-") as temporary:
        vector = ChromaVectorStore(persist_dir=temporary, collection_name="evaluation")
        keyword = SQLiteBM25Index(database_path=str(Path(temporary) / "bm25.db"))
        await vector.upsert_chunks(chunks=chunks, embeddings=await embeddings.embed_documents([c.text for c in chunks]))
        await keyword.upsert_chunks(chunks=chunks)
        cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L6-v2"
        local_reranker = LocalCrossEncoderReranker(
            model=cross_encoder_model, timeout_s=120.0,
        )
        modes = {
            "dense_only": RAGRetrievalService(
                embedding_provider=embeddings, vector_store=vector, default_top_k=TOP_K,
                prefetch_k=3, dense_enabled=True, bm25_enabled=False,
            ),
            "bm25_only": RAGRetrievalService(
                embedding_provider=embeddings, vector_store=vector, default_top_k=TOP_K,
                prefetch_k=3, keyword_index=keyword, dense_enabled=False, bm25_enabled=True,
            ),
            "hybrid_rrf": RAGRetrievalService(
                embedding_provider=embeddings, vector_store=vector, default_top_k=TOP_K,
                prefetch_k=3, keyword_index=keyword, dense_enabled=True, bm25_enabled=True,
                dense_fetch_k=3, bm25_fetch_k=3, candidate_k=3, rrf_k=60,
            ),
            "hybrid_plus_reranker": RAGRetrievalService(
                embedding_provider=embeddings, vector_store=vector, default_top_k=TOP_K,
                prefetch_k=3, keyword_index=keyword, dense_enabled=True, bm25_enabled=True,
                dense_fetch_k=3, bm25_fetch_k=3, candidate_k=3, rrf_k=60,
                reranker=local_reranker, reranker_fail_open=False,
            ),
        }
        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "parameters": {
                "chunk_size": "one authored sample document per chunk", "overlap": 0,
                "top_k": TOP_K, "rrf_k": 60, "rerank_count": 3,
                "embedding_model": "local-hash-384-v1", "llm_model": "local-extractive-v1",
                "reranker_model": cross_encoder_model,
            },
            "modes": {},
            "limitations": [
                "Three questions and three documents only; values are pipeline smoke checks, not retrieval quality evidence.",
                "Local hash embeddings and the extractive LLM are deterministic development fallbacks, not production model quality.",
                "citation_integrity only validates citation IDs against retrieved chunks; it does not measure factual entailment or faithfulness.",
                "SQLite FTS5 default tokenization is aimed at the authored English technical samples and error codes, not Chinese search quality.",
            ],
        }
        retrieval_filter = RetrievalFilter(
            product_name="Nebula Gateway", product_version="3.2",
            knowledge_space_id="nebula-support", department="payments",
            allowed_doc_ids=tuple(chunk.doc_id for chunk in chunks),
        )
        for name, service in modes.items():
            case_results = []
            for case in cases:
                retrieved = await service.retrieve(
                    query=case["question"], top_k=TOP_K, filters=retrieval_filter,
                )
                doc_ids = list(dict.fromkeys(item.doc_id for item in retrieved))
                case_results.append({
                    "id": case["id"], "retrieved_doc_ids": doc_ids,
                    **metrics(doc_ids, case["expected_doc_ids"], TOP_K),
                })
            answerable = [item for item, case in zip(case_results, cases) if case["answerable"]]
            report["modes"][name] = {
                "cases": case_results,
                f"hit_at_{TOP_K}": sum(i["hit_at_k"] for i in answerable) / len(answerable),
                f"recall_at_{TOP_K}": sum(i["recall_at_k"] for i in answerable) / len(answerable),
                "mrr": sum(i["mrr"] for i in answerable) / len(answerable),
            }
        hybrid = modes["hybrid_rrf"]
        local_llm = LocalExtractiveLLM()
        ordinary_cases = []
        for case in cases:
            retrieved = await hybrid.retrieve(
                query=case["question"], top_k=TOP_K, filters=retrieval_filter,
            )
            tool_payload = {
                "results": [
                    {
                        "doc_id": item.doc_id, "chunk_id": item.chunk_id,
                        "source": item.source, "text": item.text,
                        "page_number": item.page_number,
                    }
                    for item in retrieved
                ]
            }
            response = await local_llm.generate([
                ChatMessage(role=MessageRole.USER, content=case["question"]),
                ChatMessage(role=MessageRole.TOOL, content=str(tool_payload)),
            ])
            refused = response.content.startswith("I could not find")
            cited = {item.doc_id for item in retrieved[:TOP_K]} if not refused else set()
            expected = set(case["expected_doc_ids"])
            ordinary_cases.append({
                "id": case["id"], "status": "rejected" if refused else "ok",
                "citation_accuracy": len(cited & expected) / len(cited) if cited else float(not expected),
                "citation_integrity": all(
                    item.doc_id in {chunk.doc_id for chunk in retrieved}
                    for item in retrieved[:TOP_K]
                ),
                "refusal_correct": refused == (not case["answerable"]),
            })
        report["modes"]["ordinary_rag"] = {
            "cases": ordinary_cases,
            "citation_accuracy": sum(c["citation_accuracy"] for c in ordinary_cases) / len(ordinary_cases),
            "citation_integrity": sum(
                float(c["citation_integrity"]) for c in ordinary_cases
            ) / len(ordinary_cases),
            "refusal_accuracy": sum(float(c["refusal_correct"]) for c in ordinary_cases) / len(ordinary_cases),
        }
        agentic = LangGraphRAGService(
            retrieval_service=hybrid, llm=local_llm, max_retrieval_attempts=2,
        )
        agent_cases = []
        for case in cases:
            result = await agentic.run(
                question=case["question"], filters={
                    "allowed_doc_ids": list(retrieval_filter.allowed_doc_ids),
                    "product_name": "Nebula Gateway", "product_version": "3.2",
                    "knowledge_space_id": "nebula-support", "department": "payments",
                },
            )
            cited = {item["doc_id"] for item in result.citations}
            expected = set(case["expected_doc_ids"])
            citation_accuracy = len(cited & expected) / len(cited) if cited else float(not expected)
            agent_cases.append({
                "id": case["id"], "status": result.status,
                "citation_accuracy": citation_accuracy,
                "citation_integrity": (
                    bool(result.citations) if case["answerable"] else result.status == "rejected"
                ),
                "refusal_correct": (result.status == "rejected") == (not case["answerable"]),
            })
        report["modes"]["agentic_rag"] = {
            "cases": agent_cases,
            "citation_accuracy": sum(c["citation_accuracy"] for c in agent_cases) / len(agent_cases),
            "citation_integrity": sum(
                float(c["citation_integrity"]) for c in agent_cases
            ) / len(agent_cases),
            "refusal_accuracy": sum(float(c["refusal_correct"]) for c in agent_cases) / len(agent_cases),
        }
        output = ROOT / "eval" / "reports"
        output.mkdir(exist_ok=True)
        (output / "evaluation-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        lines = ["# Local Evaluation Report", "", f"Generated: {report['generated_at']}", ""]
        for name, data in report["modes"].items():
            lines.extend([f"## {name}", "", "```json", json.dumps(data, ensure_ascii=False, indent=2), "```", ""])
        lines.extend(["## Limitations", "", *[f"- {item}" for item in report["limitations"]]])
        (output / "evaluation-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())

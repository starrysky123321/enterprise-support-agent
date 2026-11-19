from __future__ import annotations

# ruff: noqa: E402 -- configure the isolated application before importing its routes.

import argparse
import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update({
    "DATABASE_URL": "sqlite+aiosqlite:///:memory:",
    "ENVIRONMENT": "development",
    "ACCESS_TOKEN_SECRET_KEY": "evaluation-only-access",
    "REFRESH_TOKEN_SECRET_KEY": "evaluation-only-refresh",
    "RESET_PASSWORD_TOKEN_SECRET": "evaluation-only-reset",
    "VERIFICATION_TOKEN_SECRET": "evaluation-only-verify",
    "SEMANTIC_CACHE_ENABLED": "false",
    "QUERY_REFINEMENT_ENABLED": "false",
    "AGENTIC_MAX_RETRIEVAL_ATTEMPTS": "2",
})

from fastapi import FastAPI, Header
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.agents.cache_policy import is_no_answer_fallback
from src.api.v1 import dependencies
from src.api.v1.routes.agent import router
from src.infrastructure.database import Base, get_db
from src.infrastructure.keyword import SQLiteBM25Index
from src.infrastructure.llm.huggingface_embeddings import HuggingFaceEmbeddingProvider
from src.infrastructure.llm.local_embeddings import LocalHashEmbeddingProvider
from src.infrastructure.llm.local_llm import LocalExtractiveLLM
from src.infrastructure.reranker import LocalCrossEncoderReranker
from src.infrastructure.vector_db.chroma_vectorstore import ChromaVectorStore
from src.modules.access.models import KnowledgeSpace, KnowledgeSpaceMember
from src.modules.access.service import PermissionService
from src.modules.documents.models import Document
from src.modules.users.dependencies import active_user
from src.modules.users.models import User
from src.rag.embeddings import EmbeddingProvider
from src.rag.models import RAGChunk, RetrievalFilter
from src.rag.pipeline import RAGRetrievalService
from src.tools import ToolRegistry
from src.tools.retriever_tool import RetrieverTool

DATA = ROOT / "eval" / "datasets"
CORPUS = DATA / "support-corpus.jsonl"
CASES = DATA / "support-eval.jsonl"
EXTRA_CORPUS = DATA / "support-corpus-extra.jsonl"
EXTRA_CASES = DATA / "support-eval-extra.jsonl"
PRODUCT = "Nebula Gateway"
TOP_K = 5
RRF_K = 60
CANDIDATE_K = 16
HASH_MODEL = "local-hash-384-v1"
ACTORS = {"engineer": "payments", "viewer": "operations", "outsider": "sales", "owner": "payments"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_corpus(documents: list[dict], cases: list[dict]) -> list[RAGChunk]:
    chunks = [
        RAGChunk(
            doc_id=doc["id"], chunk_id=chunk["id"], source=doc["id"], text=chunk["text"],
            page_number=page, document_type=doc["document_type"], product_name=doc["product_name"],
            product_version=doc["product_version"], department=doc["department"],
            knowledge_space_id=doc["workspace"], visibility=doc["visibility"],
        )
        for doc in documents
        for page, chunk in enumerate(doc["chunks"], start=1)
    ]
    texts = {(chunk.doc_id, chunk.chunk_id): chunk.text for chunk in chunks}
    document_ids = {doc["id"] for doc in documents}
    if len(texts) != len(chunks) or len(document_ids) != len(documents):
        raise ValueError("Duplicate document or chunk identifier")
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case identifier")
    for case in cases:
        single_doc = case["single_doc_id"]
        forbidden = set(case["forbidden_doc_ids"])
        evidence = case["evidence"]
        if (
            case["actor"] not in ACTORS or bool(evidence) != case["answerable"]
            or (single_doc is not None and single_doc not in document_ids)
            or not forbidden <= document_ids
            or (case["expect_access_denial"] and single_doc not in forbidden)
        ):
            raise ValueError(f"Invalid scope or label in {case['id']}")
        for ref in evidence:
            key = ref["doc_id"], ref["chunk_id"]
            if (
                key not in texts or not ref["quote"] or ref["quote"] not in texts[key]
                or ref["doc_id"] in forbidden or (single_doc and ref["doc_id"] != single_doc)
            ):
                raise ValueError(f"Invalid reference evidence in {case['id']}: {key}")
    return chunks


async def seed_access_data(session: AsyncSession, documents: list[dict]) -> dict[str, User]:
    users = {
        actor: User(
            id=UUID(int=index), email=f"{actor}@example.invalid", hashed_password="",
            department=department, is_active=True, is_verified=True, is_superuser=False,
        )
        for index, (actor, department) in enumerate(ACTORS.items(), start=1)
    }
    owner_id = users["owner"].id
    session.add_all(users.values())
    session.add_all([
        KnowledgeSpace(id=space, name=space, created_by=owner_id)
        for space in {doc["workspace"] for doc in documents}
    ])
    session.add_all([
        KnowledgeSpaceMember(knowledge_space_id="nebula-support", user_id=users[actor].id)
        for actor in ("engineer", "viewer", "owner")
    ])
    session.add_all([
        Document(
            id=doc["id"], owner_user_id=owner_id, source=doc["id"],
            product_name=doc["product_name"], product_version=doc["product_version"],
            document_type=doc["document_type"], visibility=doc["visibility"],
            department=doc["department"], knowledge_space_id=doc["workspace"],
            deleted_at=datetime(2026, 1, 1, tzinfo=UTC) if doc["deleted"] else None,
        )
        for doc in documents
    ])
    await session.commit()
    return users


def build_retrievers(
    embeddings: EmbeddingProvider, vector: ChromaVectorStore, keyword: SQLiteBM25Index,
    reranker: LocalCrossEncoderReranker | None,
) -> dict[str, RAGRetrievalService]:
    modes: dict[str, tuple[bool, bool, LocalCrossEncoderReranker | None]] = {
        "dense_only": (True, False, None),
        "bm25_only": (False, True, None),
        "hybrid_rrf": (True, True, None),
    }
    if reranker is not None:
        modes["hybrid_plus_reranker"] = (True, True, reranker)
    return {
        name: RAGRetrievalService(
            embedding_provider=embeddings, vector_store=vector, keyword_index=keyword,
            default_top_k=TOP_K, prefetch_k=CANDIDATE_K, dense_fetch_k=CANDIDATE_K,
            bm25_fetch_k=CANDIDATE_K, candidate_k=CANDIDATE_K, rrf_k=RRF_K,
            dense_enabled=dense, bm25_enabled=sparse, reranker=ranker, reranker_fail_open=False,
        )
        for name, (dense, sparse, ranker) in modes.items()
    }


def build_app(
    sessions: async_sessionmaker[AsyncSession], users: dict[str, User],
    retrieval: RAGRetrievalService, embeddings: EmbeddingProvider, llm: LocalExtractiveLLM,
) -> FastAPI:
    async def evaluation_db() -> AsyncIterator[AsyncSession]:
        async with sessions() as session:
            yield session

    def evaluation_user(x_eval_actor: str = Header()) -> User:
        return users[x_eval_actor]

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides.update({
        get_db: evaluation_db,
        active_user: evaluation_user,
        dependencies.get_llm: lambda: llm,
        dependencies.get_embedding_provider: lambda: embeddings,
        dependencies.get_rag_retrieval_service: lambda: retrieval,
        dependencies.get_tool_registry: lambda: ToolRegistry([
            RetrieverTool(retrieval_service=retrieval, default_top_k=TOP_K),
        ]),
    })
    return app


def retrieval_score(case: dict, matches: list[tuple[str, str]]) -> dict:
    expected = {(ref["doc_id"], ref["chunk_id"]) for ref in case["evidence"]}
    if not expected:
        return {}
    first = next((rank for rank, match in enumerate(matches, 1) if match in expected), None)
    return {
        "hit_at_5": float(first is not None),
        "recall_at_5": len(set(matches) & expected) / len(expected),
        "mrr_at_5": 1 / first if first else 0.0,
    }


def score_response(case: dict, status_code: int, body: dict) -> dict:
    citations = body.get("citations", [])
    answer = body.get("answer", "")
    refs = case["evidence"]
    expected = {(ref["doc_id"], ref["chunk_id"]) for ref in refs}
    cited = {(item["doc_id"], item["chunk_id"]) for item in citations}
    snippets = {
        (item["doc_id"], item["chunk_id"]): item["snippet"].casefold() for item in citations
    }
    exact_support = [
        ref for ref in refs
        if ref["quote"].casefold() in answer.casefold()
        and ref["quote"].casefold() in snippets.get((ref["doc_id"], ref["chunk_id"]), "")
    ]
    denied = status_code == 404
    refused = body.get("status") == "rejected" or is_no_answer_fallback(answer, exact=True)
    record = {
        "outcome": "access_denied" if denied else "refused" if refused else body.get("status"),
        "answer": answer,
        "citation_ids": sorted(cited),
    }
    if case["expect_access_denial"]:
        record["correct_access_denial"] = denied
    elif case["answerable"]:
        record.update(
            gold_citation_precision=len(cited & expected) / len(cited) if cited else 0.0,
            reference_fact_coverage=len(exact_support) / len(refs),
            false_refusal=refused,
        )
    else:
        record["correct_refusal"] = refused and not denied
    return record


def summarize(rows: list[dict], metrics: tuple[str, ...]) -> dict:
    summary = {"case_count": len(rows)}
    for metric in metrics:
        values = [row[metric] for row in rows if metric in row]
        if values:
            summary[metric] = round(sum(values) / len(values), 4)
            summary[f"{metric}_n"] = len(values)
    latencies = sorted(row["elapsed_ms"] for row in rows)
    summary["p95_ms"] = round(latencies[math.ceil(0.95 * len(latencies)) - 1], 2)
    summary["scope_violation_cases"] = sum(bool(row["out_of_scope_doc_ids"]) for row in rows)
    return summary


async def run_evaluation(args: argparse.Namespace, directory: Path) -> dict:
    corpus_paths = (CORPUS,) if args.baseline else (CORPUS, EXTRA_CORPUS)
    case_paths = (CASES,) if args.baseline else (CASES, EXTRA_CASES)
    documents = [row for path in corpus_paths for row in load_jsonl(path)]
    cases = [row for path in case_paths for row in load_jsonl(path)]
    chunks = build_corpus(documents, cases)
    embeddings = (
        LocalHashEmbeddingProvider() if args.embedding_model == HASH_MODEL
        else HuggingFaceEmbeddingProvider(model_name=args.embedding_model)
    )
    reranker = (
        LocalCrossEncoderReranker(model=args.reranker_model, timeout_s=120.0)
        if args.reranker_model else None
    )
    vector = ChromaVectorStore(persist_dir=str(directory / "chroma"), collection_name="evaluation")
    keyword = SQLiteBM25Index(database_path=str(directory / "bm25.db"))
    await vector.upsert_chunks(
        chunks=chunks, embeddings=await embeddings.embed_documents([chunk.text for chunk in chunks]),
    )
    await keyword.upsert_chunks(chunks=chunks)
    modes = build_retrievers(embeddings, vector, keyword, reranker)
    llm = LocalExtractiveLLM()
    engine = create_async_engine(f"sqlite+aiosqlite:///{directory / 'access.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    retrieval_results: dict[str, list[dict]] = {name: [] for name in modes}
    response_results: dict[str, list[dict]] = {"agentic_rag": [], "ordinary_rag": []}
    try:
        tables = [
            Base.metadata.tables[model.__tablename__]
            for model in (User, KnowledgeSpace, KnowledgeSpaceMember, Document)
        ]
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all, tables=tables)
        async with sessions() as session:
            users = await seed_access_data(session, documents)
            permissions = PermissionService(session)
            scopes = {
                (actor, version): await permissions.allowed_document_ids(
                    user=users[actor], product_name=PRODUCT, product_version=version,
                )
                for actor, version in {(case["actor"], case["version"]) for case in cases}
            }
        first = cases[0]
        warmup_scope = RetrievalFilter(
            allowed_doc_ids=scopes[first["actor"], first["version"]], product_version=first["version"],
        )
        for service in modes.values():
            await service.retrieve(query=first["question"], filters=warmup_scope)
        for case in cases:
            visible = set(scopes[case["actor"], case["version"]])
            forbidden = set(case["forbidden_doc_ids"])
            if forbidden & visible:
                raise RuntimeError(f"Access policy admitted forbidden documents in {case['id']}")
            if {ref["doc_id"] for ref in case["evidence"]} - visible:
                raise RuntimeError(f"Access policy hid reference evidence in {case['id']}")
            filters = RetrievalFilter(
                allowed_doc_ids=tuple(sorted(visible)), product_name=PRODUCT,
                product_version=case["version"], deny_all=not visible,
            )
            for mode, service in modes.items():
                start = perf_counter()
                matches = await service.retrieve(query=case["question"], top_k=TOP_K, filters=filters)
                elapsed_ms = (perf_counter() - start) * 1000
                keys = [(chunk.doc_id, chunk.chunk_id) for chunk in matches]
                retrieval_results[mode].append({
                    "id": case["id"], "category": case["category"], "elapsed_ms": elapsed_ms,
                    "chunk_ids": keys,
                    "out_of_scope_doc_ids": sorted({doc for doc, _ in keys} - visible),
                    **retrieval_score(case, keys),
                })

        app = build_app(sessions, users, modes["hybrid_rrf"], embeddings, llm)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://evaluation") as client:
            for path in ("/agent/ask?use_cache=false", "/agent/ask/agentic"):
                warmup = await client.post(
                    path, headers={"x-eval-actor": first["actor"]},
                    json={"question": first["question"], "doc_id": first["single_doc_id"]},
                )
                warmup.raise_for_status()
            for case in cases:
                visible = set(scopes[case["actor"], case["version"]])
                scope = case["single_doc_id"]
                payload = {"question": case["question"], "product_name": PRODUCT,
                           "product_version": case["version"]}
                paths = {"agentic_rag": "/agent/ask/agentic"}
                if scope is not None:
                    payload["doc_id"] = scope
                    visible &= {scope}
                    paths["ordinary_rag"] = "/agent/ask?use_cache=false"
                for mode, path in paths.items():
                    start = perf_counter()
                    response = await client.post(path, json=payload, headers={"x-eval-actor": case["actor"]})
                    elapsed_ms = (perf_counter() - start) * 1000
                    if response.status_code != 404:
                        response.raise_for_status()
                    if (response.status_code == 404) != case["expect_access_denial"]:
                        raise RuntimeError(f"Unexpected access decision in {mode} for {case['id']}")
                    result = score_response(case, response.status_code, response.json())
                    response_results[mode].append({
                        "id": case["id"], "category": case["category"], "elapsed_ms": elapsed_ms,
                        **result,
                        "out_of_scope_doc_ids": sorted({doc for doc, _ in result["citation_ids"]} - visible),
                    })
    finally:
        await engine.dispose()

    retrieval_metrics = ("hit_at_5", "recall_at_5", "mrr_at_5")
    response_metrics = (
        "gold_citation_precision", "reference_fact_coverage", "correct_refusal",
        "false_refusal", "correct_access_denial",
    )
    paired_ids = {case["id"] for case in cases if case["single_doc_id"]}
    answer_modes = {
        "ordinary_rag_paired": response_results["ordinary_rag"],
        "agentic_rag_paired": [row for row in response_results["agentic_rag"] if row["id"] in paired_ids],
        "agentic_rag_multi_document": [
            row for row in response_results["agentic_rag"] if row["id"] not in paired_ids
        ],
    }
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.machine()}",
            "local_ml": {
                package: version(package)
                for package in ("torch", "sentence-transformers", "transformers")
            } if args.embedding_model != HASH_MODEL or args.reranker_model else {},
        },
        "parameters": {
            "documents": len(documents), "chunks": len(chunks), "cases": len(cases),
            "embedding_model": model_reference(args.embedding_model),
            "reranker_model": model_reference(args.reranker_model),
            "generation_model": llm.model_name, "top_k": TOP_K, "rrf_k": RRF_K,
            "candidate_k": CANDIDATE_K,
            "datasets_sha256": {
                path.name: sha256(path.read_bytes()).hexdigest()
                for path in (*corpus_paths, *case_paths)
            },
        },
        "retrieval": {
            mode: {"summary": summarize(rows, retrieval_metrics), "cases": rows}
            for mode, rows in retrieval_results.items()
        },
        "api": {
            mode: {"summary": summarize(rows, response_metrics), "cases": rows}
            for mode, rows in answer_modes.items()
        },
    }


def model_reference(name: str | None) -> str | None:
    if name is None:
        return None
    path = Path(name).expanduser()
    if path.is_dir():
        if path.parent.name == "snapshots":
            model = path.parent.parent.name.removeprefix("models--").replace("--", "/")
            return f"{model}@{path.name}"
        return path.name
    return name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval and the FastAPI answer routes.")
    parser.add_argument("--baseline", action="store_true", help="Use the original 12 documents and 50 cases")
    parser.add_argument("--embedding-model", default=HASH_MODEL, help="Hugging Face model ID or local snapshot")
    parser.add_argument("--reranker-model", help="Optional local Cross-Encoder model ID or snapshot")
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "reports" / "evaluation-report.json")
    args = parser.parse_args()
    if args.output.suffix != ".json":
        parser.error("--output must be a .json path")
    with tempfile.TemporaryDirectory(prefix="enterprise-support-eval-") as temporary:
        report = asyncio.run(run_evaluation(args, Path(temporary)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summaries = {
        group: {mode: result["summary"] for mode, result in report[group].items()}
        for group in ("retrieval", "api")
    }
    summary_json = json.dumps(summaries, indent=2)
    args.output.with_suffix(".md").write_text(
        f"# Local Evaluation Report\n\nGenerated {report['generated_at']}. "
        "Results use synthetic support documents and local extractive answers. "
        "The accompanying JSON contains the model settings and individual cases.\n\n"
        f"```json\n{summary_json}\n```\n",
        encoding="utf-8",
    )
    print(summary_json)
    if any(summary["scope_violation_cases"] for group in summaries.values() for summary in group.values()):
        raise SystemExit("Evaluation returned documents or citations outside the caller's scope.")

from typing import Literal

from pydantic import BaseModel, Field

DocumentType = Literal["manual", "api_doc", "sop", "ticket", "release_note", "faq", "config_doc"]
Visibility = Literal["private", "department", "workspace"]


class RAGIngestTextRequest(BaseModel):
    text: str = Field(min_length=1, description="Raw text to index.")
    chunking_strategy: str | None = Field(
        default=None,
        description="Optional chunking strategy; default strategy is used if omitted.",
    )
    source: str | None = Field(
        default=None,
        description="Optional source label (file name, URL, or domain key).",
    )
    doc_id: str | None = Field(
        default=None,
        description="Optional document id; generated automatically if omitted.",
    )
    document_type: DocumentType = "manual"
    product_name: str | None = Field(default=None, max_length=255)
    product_version: str | None = Field(default=None, max_length=128)
    department: str | None = Field(default=None, max_length=255)
    knowledge_space_id: str | None = Field(default=None, max_length=255)
    visibility: Visibility = "private"


class RAGIngestTextResponse(BaseModel):
    status: str = Field(description="Ingestion status.")
    doc_id: str = Field(description="Document id used for indexing.")
    chunks_ingested: int = Field(description="Number of indexed chunks.")


class RAGIngestPDFResponse(BaseModel):
    status: str = Field(description="Ingestion status.")
    doc_id: str = Field(description="Document id used for indexing.")
    chunks_ingested: int = Field(description="Number of indexed chunks.")
    pages_total: int = Field(description="Total number of PDF pages.")
    pages_ingested: int = Field(description="Number of pages successfully ingested.")
    skipped_pages: list[int] = Field(
        default_factory=list,
        description="1-based page numbers skipped during extraction.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Extraction warnings encountered during partial ingestion.",
    )


class RAGRetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    product_name: str | None = None
    product_version: str | None = None
    document_types: list[DocumentType] = Field(default_factory=list)
    knowledge_space_id: str | None = None
    top_k: int = Field(default=8, ge=1, le=50)

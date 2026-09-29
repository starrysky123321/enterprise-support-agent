from __future__ import annotations

import hashlib
import json
from typing import Any


def build_permission_cache_key(
    *, user_id: str, permission_scope: list[str], knowledge_space_id: str | None,
    filters: dict[str, Any], model_version: str, prompt_version: str,
    pipeline_version: str,
) -> str:
    """Canonical cache partition key; never includes question or document text."""
    payload = {
        "user_id": user_id,
        "permission_scope": sorted(permission_scope),
        "knowledge_space_id": knowledge_space_id,
        "filters": filters,
        "model_version": model_version,
        "prompt_version": prompt_version,
        "pipeline_version": pipeline_version,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

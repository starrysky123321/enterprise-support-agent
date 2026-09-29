from src.agents.cache_key import build_permission_cache_key


def test_cache_key_is_stable_but_partitioned_by_permission_and_pipeline():
    base = dict(
        user_id="u1", permission_scope=["doc-b", "doc-a"], knowledge_space_id="space-a",
        filters={"product": "Nebula", "version": "3.2"}, model_version="local-v1",
        prompt_version="support-v1", pipeline_version="hybrid-v1",
    )
    assert build_permission_cache_key(**base) == build_permission_cache_key(**base)
    changed = {**base, "permission_scope": ["doc-a"]}
    assert build_permission_cache_key(**base) != build_permission_cache_key(**changed)
    changed_pipeline = {**base, "pipeline_version": "hybrid-v2"}
    assert build_permission_cache_key(**base) != build_permission_cache_key(**changed_pipeline)

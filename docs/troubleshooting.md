# Troubleshooting

## 导入 Settings 失败

应用要求 `DATABASE_URL` 和四个 JWT secret。复制 `.env.example`，不要把真实 secret 提交。测试已在 `tests/conftest.py` 使用隔离值，不依赖开发者 `.env`。

## PostgreSQL 连接失败

Compose 内应用使用主机名 `db`；宿主机运行 Alembic 应使用 `localhost`。确认 `.env` 的 `POSTGRES_USER/POSTGRES_PASSWORD/POSTGRES_DB` 与 `DATABASE_URL` 一致。

## ARQ 任务一直 pending

确认 `redis-cli -u "$REDIS_URL" ping`、worker 进程和 API 使用同一 Redis URL；查看 worker 日志中的 task ID。开发机可设 `INGESTION_MODE=background`，无需 Redis worker。

失败任务且原始 payload 仍在时调用 `POST /ingestion/tasks/{task_id}/retry`。自动重试使用固定首次 job ID 防止重复派发，人工恢复使用新 job ID；如果 payload 已丢失，接口返回 409 并要求重新上传。

## Dense 有结果但 BM25 无结果

确认 `BM25_ENABLED=true`、`BM25_DATABASE_PATH` 可写，并检查文档 task 是否到 `completed`。删除/重传会按 `doc_id:chunk_id` upsert，不应出现重复项。

## 召回其他部门文档

禁止直接调用底层 index 绕过 `PermissionService`。API 应先计算 `allowed_doc_ids`，再构造 `RetrievalFilter`。使用跨部门测试复现并检查文档 visibility、department 和 workspace membership。

## Reranker 故障

无 key 环境可先执行 `uv sync --extra local-ml`，再设 `RERANKER_PROVIDER=local` 并将 `RERANKER_MODEL` 指向 sentence-transformers Cross-Encoder；默认安装刻意不包含 Torch/CUDA。Cohere 模式需 key。若生产希望可用性优先，设 `RERANKER_FAIL_OPEN=true`；此时返回 RRF 顺序并记录 degraded event，不无限重试。`RERANKER_TIMEOUT_S` 限制单次调用。

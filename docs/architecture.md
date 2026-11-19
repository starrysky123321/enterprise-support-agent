# Architecture

## 请求链路

```mermaid
flowchart LR
  U[Authenticated user] --> P[PermissionService]
  P --> F[RetrievalFilter / allowed_doc_ids]
  F --> B[SQLite FTS5 BM25]
  F --> D[Chroma Dense]
  B --> R[RRF fusion]
  D --> R
  R --> X[Optional Cross-Encoder]
  X --> O[Single-document tool-based RAG]
  O --> S[Answer with retrieved evidence]
  X --> G[Bounded LangGraph RAG]
  G --> V[Citation ID integrity check]
  V --> A[Answer or structured rejection]
```

权限条件不是召回后的清理步骤。`PermissionService` 先计算当前用户可访问的文档 ID，`RetrievalFilter` 同时传给 SQLite FTS5 SQL 和 Chroma `where`。第三方旧 VectorStore adapter 不支持 filter 时，兼容路径才会在 adapter 返回后过滤；内置生产 adapter 使用前置过滤。

对于 `CFG-3207 是什么意思` 这类单个错误码的释义查询，Dense 检索使用完整问题，BM25 在原权限和版本范围内按提取出的编号搜索。系统选一个明确解释该编号的分块，在 RRF 截断前保留，并占用最终 Top-K 的一个位置。启用重排时，其余位置按重排后的顺序返回。

## 摄取链路

```text
POST /ingestion/text
  -> 保存 payload + Document + IngestionTask(pending)
  -> 返回 202/task_id
  -> BackgroundTask（开发）或 ARQ/Redis（部署）
  -> parsing -> chunking -> indexing
  -> embedding/Chroma + SQLite FTS5
  -> completed / failed
```

内容 SHA-256 与确定性自动 `doc_id` 防止重复任务。Chunk ID 由分块器确定，Dense 和 BM25 都采用 upsert；worker 重试不会增加重复索引。删除文档同时清理 Chroma 与 BM25。上传文件落在 `INGESTION_STORAGE_DIR`，密钥只从环境读取。

## LangGraph 状态与停止条件

State 只有业务数据：original/current query、messages、filters、attempts、检索/相关文档、answer、citations、groundedness 和 failure reason。LLM 与 retrieval service 由构造函数注入，不进入 State。

`classify -> rewrite -> retrieve -> grade` 后，本地抽取模式检查文档是否给出了问题所需的显式信息。比如问某版本的默认超时时间，要能定位到同时说明默认设置与时长的片段。日期、命令和表名也有对应的检查。普通单文档问答复用该检查，缺少证据时返回无答案结果。

Agentic 路径在证据不足时回到 rewrite，继续在原权限范围内检索，默认最多检索两次。生成式模型会收到逐项检查问题所需事实的提示，明确拒答会走同一重试或拒答分支。API 的引用校验检查引用 ID 与本次证据的对应关系，SSE 在引用校验通过时发送回答与引用。完整的事实级语义核验仍需独立验证。

## 降级策略

- 没有 API key：`LLM_PROVIDER=local` 与 `EMBEDDING_PROVIDER=local` 保证核心链路运行。
- `LLM_PROVIDER=qwen` 通过 DashScope OpenAI-compatible endpoint 使用通义千问；密钥只从 `DASHSCOPE_API_KEY` 读取，和 Embedding Provider 独立切换。
- BM25 或 Dense 可分别关闭；至少一路开启时检索正常。
- Reranker 可选本地 Cross-Encoder 或 Cohere；本地 ML 依赖通过 `local-ml` Extra 安装，默认 Docker/uv 安装不包含 Torch/CUDA。未配置时关闭，运行失败且 `RERANKER_FAIL_OPEN=true` 时返回 RRF 候选并记录原因。
- OpenAI LLM/Embedding 显式配置请求超时和 SDK 最大重试；Reranker 有超时且检索服务至多尝试两次。
- 语义缓存异常不阻断生成；权限缓存指纹包括 user、allowed docs、space、filter、model、prompt 与 pipeline version。
- 普通 `/agent/ask` 保留为 LangGraph 故障时的降级路径。

## 已知边界

- SQLite FTS5 适合单机、CI、英文错误码和英文技术文档；默认 tokenizer 对中文检索不理想。中文生产场景应实现现有 `KeywordIndex` 接口的 OpenSearch/Elasticsearch 中文分词 adapter。
- 本地 hash embedding 只用于无密钥开发与确定性链路验证，不具备可宣称的同义表达/语义泛化能力。语义质量验收必须改用真实 Embedding 模型并重建索引。
- `/ingestion/text` 与 `/ingestion/pdf` 都通过同一任务/worker 状态机处理。
- 普通 RAG 在 pgvector 原有分区字段中写入权限指纹；指纹包含当前用户、文档范围、知识空间、filter、模型、prompt 与 pipeline version。多文档 Agentic 路径当前不缓存。
- ARQ 自动失败最多执行 `INGESTION_MAX_RETRIES + 1` 次；人工 retry 使用新的 Redis job ID，避免被 ARQ 的已完成 job 去重键吞掉，同时两路索引仍按确定性 Chunk ID upsert。

# Evaluation

## 数据格式

`eval/datasets/support-eval.jsonl` 每行包含：问题、期望文档 ID、参考事实、权限范围和是否可回答。示例文档全部为本项目自行编写，存放在 `docs/examples/`。

## 运行

```bash
uv run --extra local-ml python eval/run_local_evaluation.py
```

输出：

- `eval/reports/evaluation-report.json`：机器可读结果与参数
- `eval/reports/evaluation-report.md`：人类可读逐模式报告

脚本实际比较 Dense Only、BM25 Only、Hybrid RRF、Hybrid + 本地 Cross-Encoder、普通单轮 RAG 和 Agentic RAG。本地 reranker 使用 `cross-encoder/ms-marco-MiniLM-L6-v2`，模型按 sentence-transformers 缓存；无 Cohere/OpenAI key 也能执行。

指标定义：Hit@K 表示 Top-K 是否含任一期望文档；Recall@K 是期望文档被找回比例；MRR 使用第一个期望文档的倒数排名；引用准确性是引用文档中属于期望集合的比例；`citation_integrity` 只检查引用是否来自本次检索结果；拒答准确性比较实际 status 与 `answerable`。当前没有事实级 NLI/LLM Judge，因此没有实现或报告真正的 Groundedness/Faithfulness。

2026-08-07 的当前报告只有 3 条案例和 3 篇语料。现已从 Top-3 改为 Top-2，避免每次直接返回全部语料，但样本仍远不足以支持质量结论；即便 Hit@2/Recall@2/MRR 为 1.0，也只能证明这三个固定案例能跑通。普通 RAG 的引用准确性为 0.5、拒答准确性为 0.6667；Agentic RAG 对应为 1.0 和 1.0。`citation_integrity=1.0` 不等于事实忠实度。不得把这些数值描述为生产 Recall、MRR、Groundedness 或延迟指标。

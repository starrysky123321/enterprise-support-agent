# Nebula Gateway 故障诊断演示

运行以下命令复现三个现象。评测用合成资料和临时索引，在进程内调用实际 FastAPI 问答路由。

```bash
uv run python eval/run_local_evaluation.py > /dev/null
uv run python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("eval/reports/evaluation-report.json").read_text())
case_ids = ("timeout-roll-back", "ticket-1842-1", "acl-department", "no-database-secret", "no-rate-exception")
results = {
    case["id"]: case
    for mode in ("agentic_rag_paired", "agentic_rag_multi_document")
    for case in report["api"][mode]["cases"]
}
for case_id in case_ids:
    case = results[case_id]
    print(f"\n{case_id} | {case['outcome']}")
    if case["answer"]:
        print(case["answer"])
    print("Citations:", case["citation_ids"])
PY
```

1. **跨文档诊断**。询问 Nebula 3.2 默认空闲超时和回滚阈值。查看版本说明中的 30 秒默认值，以及 SOP 中连接占用连续 5 分钟超过 80% 的回滚阈值。两项引用分别指向 `release-32:chunk-1` 与 `timeout-sop:chunk-2`。
2. **同样的问题切换身份**。支付部门工程师能查到 INC-1842 工单中的 45 秒上游响应时间。运营部门用户请求同一文档时，API 返回 404，引用为空。
3. **证据不足时拒答**。询问数据库管理员密码，Agentic API 返回拒答且引用为空。也可以看 `no-rate-exception`。系统找到了同主题的限流资料，缺少用户询问的紧急豁免时长，因此拒答。[改动前后的评测](evaluation.md)

[请求链路](architecture.md) · [题集原文](../eval/datasets/support-eval.jsonl)

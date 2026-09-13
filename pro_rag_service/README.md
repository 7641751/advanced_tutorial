# pro_rag_service — 生产级 RAG 服务（进阶教程第 12 讲）

一个**带缓存 + 人工审批 + 流式部署 + 离线评估**的 RAG 服务，
串联 `advanced_tutorial/07-11` 的全部知识点。

## 知识串联

| 模块 | 用到的知识点 | 对应讲次 |
|---|---|---|
| `cache.py` | LLM 缓存（InMemory / SQLite） | 07 |
| `graph.py` | Functional API（@entrypoint/@task）+ interrupt 审批 | 08 + 09 |
| `ingest.py` | 带 year/topic 元数据的入库（供 SelfQuery 过滤） | 10 |
| `server.py` | FastAPI + SSE 流式部署 | 11 |
| `evaluate.py` | 离线评估（固定问题集 + 断言检查） | 11 |

## 架构

```
提问 → retrieve(向量检索) → generate(LLM 生成) → interrupt(人工审批)
                                                     ├─ 批准 → 返回
                                                     └─ 驳回 → approved=False
```

## 项目结构

```
pro_rag_service/
├── config.py            # 配置中心（pydantic-settings，前缀 PRORAG_）
├── cache.py             # LLM 缓存封装（07）
├── ingest.py            # 入库：加载→切分→嵌入→Chroma（带元数据，10）
├── graph.py             # Functional API 编排 + 审批（08+09）
├── server.py            # FastAPI 部署 + SSE（11）
├── evaluate.py          # 离线评估（11）
├── data/
│   ├── knowledge.json   # 示例知识库（5 条，带 year/topic）
│   └── chroma_db/       # 向量库（ingest 后生成）
└── tests/
    └── test_graph.py    # 离线测试（Fake model/retriever，零额度）
```

## 运行指引

前置：项目根目录 `.env` 已配置 `DEEPSEEK_API_KEY`。

```bash
cd advanced_tutorial/pro_rag_service

# 1. 入库（首次必跑；幂等，已有库会跳过）
python ingest.py

# 2. 离线测试（零 API 额度）
python -m pytest tests/ -v

# 3. 启动服务
python server.py

# 4. 另开终端测试
curl "http://127.0.0.1:8000/ask?q=年假有几天"
curl "http://127.0.0.1:8000/ask/stream?q=年假有几天"

# 5. 离线评估
python evaluate.py
```

## 审批流演示（graph.py）

```python
from backend.app.agent.graph import build_app
from langgraph.types import Command

app = build_app(with_approval=True)
cfg = {"configurable": {"thread_id": "demo"}}

# 第一次：停在审批，拿到待审内容
r = app.invoke("年假有几天？", config=cfg)
print(r["__interrupt__"])  # 里面是待审批的回答

# 第二次：人工批准
r = app.invoke(Command(resume={"ok": True}), config=cfg)
print(r["approved"], r["answer"])
```

## 常见问题（FAQ）

**Q1: 缓存如何开关？**
`config.py` 的 `cache_enabled` / `cache_persistent` 控制；或环境变量
`PRORAG_CACHE_ENABLED=false`。持久化缓存落盘到 `.pro_rag_cache.db`。

**Q2: SelfQuery 元数据过滤怎么用？**
`ingest.py` 已把 `year`/`topic` 写入 metadata，检索时可用
`vectorstore.similarity_search(q, filter={"year": {"$eq": 2024}})` 过滤（见教程 10）。

**Q3: 审批流为什么必须配 checkpointer？**
`interrupt()` 依赖状态持久化才能暂停与恢复，`build_app(with_approval=True)`
内部已默认挂 `InMemorySaver`，生产换 `SqliteSaver`。

**Q4: 如何不花额度做回归测试？**
`tests/test_graph.py` 用 `FakeModel` + `FakeRetriever` 依赖注入，覆盖
编译、非审批流、审批批准、审批驳回四条路径。

## 与教程的关系

本项目是 `advanced_tutorial/` 的收官实践：
- 07 缓存 → `cache.py`
- 08 Functional API → `graph.py` 的 @entrypoint/@task
- 09 人工介入 → `graph.py` 的 interrupt 审批
- 10 进阶检索 → `ingest.py` 元数据 + SelfQuery 过滤
- 11 部署可观测 → `server.py` + `evaluate.py`

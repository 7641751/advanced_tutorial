# advanced_tutorial — LangChain / LangGraph 进阶教程

面向已学完基础篇（`langchain_tutorial/`、`langgraph_tutorial/`、`rag_tutorial/`）的开发者。

## 模块地图

| 模块 | 文件 | 内容 |
|---|---|---|
| 1. 核心概念深入 | `01_LangChain核心概念进阶.ipynb` | LCEL 组合术 / Agent 结构化输出 / 长期记忆 Store / 混合检索 |
| | `02_LangGraph设计模式.ipynb` | 子图封装 / Command 路由 / Send 并行归约 / 哨兵节点 / 反模式 |
| | `03_Runtime状态更新与回溯.ipynb` | update_state / 时间旅行 / 检查点分叉 |
| 2. 端到端项目 | `rag_qa_project/` | CRAG 式智能问答系统（.py 项目结构 + 测试 + README） |
| 3. 高级特性 | `04_流式输出与回调机制.ipynb` | 四种 stream_mode / 自定义回调 / LangSmith |
| | `05_自定义工具与错误处理.ipynb` | 参数校验 / ToolException / 重试 / 兜底 / middleware |
| 4. 最佳实践 | `06_最佳实践_模块化配置测试与性能.ipynb` | 分层架构 / 配置管理 / 测试金字塔 / 性能优化 |
| 5. 生产进阶 | `07_LLM缓存与成本控制.ipynb` | 缓存（InMemory/SQLite）/ with_retry / usage_metadata |
| | `08_LangGraph_Functional_API.ipynb` | @entrypoint / @task / 并行 / 选型对比 |
| | `09_Human_in_the_loop.ipynb` | interrupt / Command(resume) / 审批流 |
| | `10_RAG进阶检索三件套.ipynb` | SelfQuery（手写）/ ParentDocument / HyDE |
| | `11_部署与可观测.ipynb` | FastAPI + SSE 部署 / TestClient / LangSmith |
| 6. 端到端项目 | `pro_rag_service/` | 生产级 RAG 服务（缓存 + 审批 + 流式 + 评估） |

## 学习路径建议

```
01 -> 02 -> 03 -> rag_qa_project（动手跑通） -> 04 -> 05 -> 06
              -> 07 -> 08 -> 09 -> 10 -> 11 -> pro_rag_service（动手跑通）
```

## 环境要求

```bash
cd my_langchain_demo
uv add langchain langgraph langchain-deepseek langchain-classic \
      langchain-community langchain-huggingface langchain-chroma \
      pydantic-settings python-dotenv pytest
```

- 根目录 `.env` 需配置 `DEEPSEEK_API_KEY`（真实 LLM，无本地降级）
- 检索章节使用本地 `BAAI/bge-small-zh-v1.5` 嵌入（首次自动经 hf-mirror 下载）
- `rag_qa_project` 完整运行指引见其目录下 `README.md`

## 已验证事项

- 所有 notebook 代码基于当前 `.venv`（langchain 1.3.14 / langgraph 1.2.10）实测 API 编写
- `rag_qa_project`：4 个离线测试全过；入库、结构图、真实问答、重写循环均已端到端验证
- `pro_rag_service`：4 个离线测试全过（编译 / 非审批 / 批准 / 驳回）
- 经典检索器（Ensemble/MultiQuery）从 `langchain_classic.retrievers` 导入（community 包已移除）
- `create_agent` 从 `langchain.agents` 导入（当前 langgraph 版本的 prebuilt 未导出）
- **缓存**：`InMemoryCache` 在 `langchain_core.caches`，`SQLiteCache` 在 `langchain_community.cache`
- **SelfQuery**：`langchain_classic` 的 `from_llm` 有版本兼容 bug，教程 10 改用手写实现
- **ParentDocument** 的 `docstore` 需 `langchain_core.stores`（不是 `langgraph.store`）

## 重新生成 notebook

```bash
cd advanced_tutorial
python _build_advanced.py   # 生成 01-06
python _build_pro.py        # 生成 07-11
```

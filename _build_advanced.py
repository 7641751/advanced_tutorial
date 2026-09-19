# -*- coding: utf-8 -*-
"""生成 advanced_tutorial/ 下的 6 个进阶教学 notebook。

运行方式（在 advanced_tutorial 目录下）：
    python _build_advanced.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


def dump(name, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    (OUT / name).write_text(
        json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written: {name} ({len(cells)} cells)")


# 每个 notebook 的首格初始化代码（保证可独立运行）
HEADER_CODE = '''
# ========== 0. 初始化（每个 notebook 第一格） ==========
import os, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

# HF 镜像必须先于任何 langchain/huggingface 导入设置（详见 rag_qa_project FAQ）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path.cwd().parent  # advanced_tutorial 的上一级 = 项目根目录
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

assert os.getenv("DEEPSEEK_API_KEY"), "请先在根目录 .env 配置 DEEPSEEK_API_KEY"

# 真实 LLM：DeepSeek（本教程要求真实模型）
from langchain_deepseek import ChatDeepSeek

model = ChatDeepSeek(model="deepseek-chat", temperature=0.2)
print("模型就绪:", model.__class__.__name__)
'''

HEADER_MD = '''# 0. 环境准备与运行说明

**前置要求：**
- 根目录 `.env` 已配置 `DEEPSEEK_API_KEY`（本教程用真实 DeepSeek，无本地降级）
- 已安装：`langchain>=1.3`、`langgraph>=1.2`、`langchain-deepseek`、`python-dotenv`
- 使用本地 `bge-small-zh-v1.5` 嵌入的章节首次运行会下载模型（约 100MB，走 hf-mirror 镜像）

**运行说明：**
- 按 cell 顺序执行；除标注外，每个示例消耗少量 API 额度（单次 < 0.01 元量级）
- 本教程面向已学完 `langchain_tutorial/` 与 `langgraph_tutorial/` 基础篇的开发者
- 涉及导入路径的坑（如 `create_agent` 在 `langchain.agents`）已在 FAQ 中汇总'''

FAQ_01 = '''## 5. 常见问题（FAQ）

| 问题 | 原因 | 解决 |
|---|---|---|
| `cannot import name 'EnsembleRetriever' from 'langchain_community'` | community 包已移除经典检索器 | 从 `langchain_classic.retrievers` 导入 |
| `with_structured_output` 报 NotImplementedError | Fake 模型未实现 | 自定义 Fake 时覆盖该方法（见 rag_qa_project/tests） |
| Agent 没记住上一轮对话 | 未配置 checkpointer | `create_agent(checkpointer=InMemorySaver())` + thread_id |
| 长期记忆写入了但读不到 | store 命名空间不一致 | put/get 的 namespace 与 key 必须完全一致 |
| `certificate verify failed ... huggingface.co` | HF_ENDPOINT 设置太晚 | 在任何 langchain/chromadb 导入前设置（见首格注释） |'''


# ========== 01 LangChain 核心概念进阶 ==========
def build_01():
    cells = [
        md('''# 进阶教程（一）：LangChain 核心概念深入

> 面向已掌握基础的开发者，深入 Chains / Agents / Memory / Retrievers 四大件的进阶用法。

## 本讲内容
1. **Chains**：LCEL 组合术（并行 / 动态路由 / 级联兜底）
2. **Agents**：结构化输出 + Agent 作为可组合单元
3. **Memory**：从会话记忆到跨线程长期记忆（Store）
4. **Retrievers**：多查询 / 混合检索 / 检索后压缩'''),
        md(HEADER_MD),
        code(HEADER_CODE),
        md('''## 1. Chains 进阶：LCEL 组合术

LCEL 的核心思想：一切皆 Runnable，用管道组合出数据流。
进阶技巧集中在三件事：**并行、动态路由、容错**。'''),
        md('''### 1.1 RunnableParallel：并行分支与汇总

多个分支同时执行（内部线程池），最后自动合并为 dict——
这是性能优化的第一手段，也常用于 RAG 的"问题改写 + 检索"并行。'''),
        code('''
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from operator import itemgetter

summary_chain = (
    ChatPromptTemplate.from_template("用一句话总结：{text}")
    | model | StrOutputParser()
)
keywords_chain = (
    ChatPromptTemplate.from_template("提取 3 个关键词，逗号分隔：{text}")
    | model | StrOutputParser()
)

parallel = RunnableParallel(
    summary=summary_chain,
    keywords=keywords_chain,
)

result = parallel.invoke({"text": "LangGraph 用状态图编排 LLM 工作流，"
                                  "支持循环、分支与持久化，是构建可靠 Agent 的底层引擎。"})
print("摘要:", result["summary"])
print("关键词:", result["keywords"])'''),
        md('''**预期输出**（大意）：
```
摘要: LangGraph 以状态图方式编排 LLM 工作流……
关键词: LangGraph, 状态图, 工作流
```

两个分支并行执行，总耗时约等于较慢的那个分支，而不是两者之和。'''),
        md('''### 1.2 动态路由：RunnableLambda 分发

按输入内容把请求分发到不同处理链——路由逻辑是纯 Python 函数，
返回的 key 决定走哪条分支。'''),
        code('''
from langchain_core.runnables import RunnableLambda

tech_chain = ChatPromptTemplate.from_template(
    "你是技术专家，回答：{q}") | model | StrOutputParser()
casual_chain = ChatPromptTemplate.from_template(
    "你是闲聊伙伴，轻松回答：{q}") | model | StrOutputParser()

def route(q: str) -> str:
    """路由函数：含技术关键词走 tech，否则走 casual"""
    return "tech" if any(k in q for k in ["LangChain", "LangGraph", "API", "代码"]) \\
           else "casual"

router = RunnableLambda(route)
chain = (
    {"q": RunnablePassthrough()}
    | RunnableLambda(lambda x: print(f"[route] -> {route(x['q'])}") or x)
    | RunnableLambda(lambda x: (tech_chain if route(x["q"]) == "tech"
                                else casual_chain).invoke(x))
)
print(chain.invoke("LangGraph 的 checkpointer 是什么？")[:80])'''),
        md('''### 1.3 级联兜底：with_fallbacks

主链路故障时自动切换备用链路——生产环境的标配。
首选链故意抛错，观察兜底生效：'''),
        code('''
from langchain_core.runnables import RunnableLambda

def flaky(q):
    raise ConnectionError("模拟主服务宕机")

backup_chain = ChatPromptTemplate.from_template("备用通道回答：{q}") \\
    | model | StrOutputParser()

robust = (
    RunnableLambda(flaky)
    .with_fallbacks([{"q": RunnablePassthrough()} | backup_chain])
)
print(robust.invoke("什么是 RAG？")[:100])'''),
        md('''**要点**：`with_fallbacks` 接受 Runnable 列表，依次尝试直到成功。
典型生产组合：DeepSeek 官方 API → 中转站 → 静态回答。'''),
        md('''## 2. Agents 进阶

### 2.1 结构化输出：response_format

让 Agent 输出 Pydantic 校验过的结构化结果，
把"自由文本 Agent"变成"可编程 API"：'''),
        code('''
from pydantic import BaseModel, Field
from langchain.agents import create_agent

class Answer(BaseModel):
    """带依据的回答"""
    answer: str = Field(description="最终回答")
    confidence: float = Field(description="置信度 0-1")
    needs_more_info: bool = Field(description="是否需要追问用户")

agent = create_agent(
    model=model,
    tools=[],
    system_prompt="你严谨的问答助手，回答必须给出置信度。",
    response_format=Answer,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "地球到月球多远？"}]})
parsed = result["structured_response"]
print(type(parsed).__name__)
print("回答:", parsed.answer)
print("置信度:", parsed.confidence)'''),
        md('''**预期输出**（大意）：
```
Answer
回答: 约 38.4 万公里
置信度: 0.98
```

`result["structured_response"]` 是 Pydantic 实例，字段类型由框架保证。'''),
        md('''### 2.2 Agent 作为可组合单元（子代理模式）

Agent 本身是 CompiledStateGraph（Runnable），可以直接嵌入更大的图/链中——
这是多 Agent 系统的基本粒子：'''),
        code('''
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """查询城市天气。Args: city: 城市名"""
    return {"北京": "晴 25C", "上海": "多云 28C"}.get(city, f"{city} 未收录")

weather_agent = create_agent(
    model=model, tools=[get_weather],
    system_prompt="你是天气助手，必须用工具查询。", name="weather_agent")

# 把 Agent 当节点用：外层链负责"意图识别后转发"
def ask_weather(question: str) -> str:
    r = weather_agent.invoke({"messages": [{"role": "user", "content": question}]})
    return r["messages"][-1].content

print(ask_weather("北京天气怎么样？"))'''),
        md('''## 3. Memory 进阶：从会话到长期记忆

| 类型 | 机制 | 生命周期 | 典型用途 |
|---|---|---|---|
| 短期记忆 | checkpointer（按 thread_id 隔离） | 单会话 | 多轮对话上下文 |
| 长期记忆 | BaseStore（按 namespace 隔离） | 跨会话/跨线程 | 用户偏好、事实沉淀 |'''),
        code('''
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver

store = InMemoryStore()

# 模拟"画像采集"环节写入长期记忆（namespace 按 用户/领域 组织）
store.put(("user", "alice", "prefs"), "style", {"text": "回答保持简洁，用中文"})

# 读取记忆并注入 system prompt —— 长期记忆的标准用法
def build_agent_with_memory(user: str):
    pref = store.get(("user", user, "prefs"), "style")
    extra = pref.value["text"] if pref else ""
    return create_agent(
        model=model, tools=[], checkpointer=InMemorySaver(),
        system_prompt=f"你是助手。用户偏好：{extra}")

# 新线程中 agent 自动遵守"几天前"写入的偏好
agent_a = build_agent_with_memory("alice")
r = agent_a.invoke(
    {"messages": [{"role": "user", "content": "介绍 RAG"}]},
    config={"configurable": {"thread_id": "t1"}})
print(r["messages"][-1].content[:100])'''),
        md('''**要点**：短期记忆换 thread_id 即失效；长期记忆跟随 **namespace**
（如 `("user", "alice", "prefs")`）跨线程、甚至跨 Agent 存活。
生产环境把 `InMemoryStore` 换成 `PostgresStore` 等持久化实现即可，接口不变。'''),
        md('''## 4. Retrievers 进阶'''),
        code('''
# 准备 mini 向量库（内存 Chroma + 本地 bge 嵌入，首次运行自动下载模型）
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

docs = [
    Document(page_content="LangGraph 的 checkpointer 在每个节点执行后保存状态快照，"
                          "支持断点续跑与时间旅行调试。", metadata={"topic": "langgraph"}),
    Document(page_content="DeepSeek-V3 采用 MoE 架构，671B 总参数、37B 激活参数，"
                          "在代码与数学推理上对标一线闭源模型。", metadata={"topic": "model"}),
    Document(page_content="RAG 通过检索外部知识再生成，能有效抑制大模型幻觉，"
                          "是知识密集型场景的主流方案。", metadata={"topic": "rag"}),
]

vectorstore = Chroma.from_documents(docs, embeddings, collection_name="mini")
vs_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
print("向量检索:", [d.metadata["topic"] for d in vs_retriever.invoke("什么是幻觉的解药")])'''),
        md('''### 4.1 MultiQueryRetriever：查询扩展

用户提问往往与文档措辞不一致。让 LLM 生成多个查询变体，
分别检索后合并去重——召回率显著提升：'''),
        code('''
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

mq_retriever = MultiQueryRetriever.from_llm(
    retriever=vs_retriever, llm=model)

import logging
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

docs = mq_retriever.invoke("怎么让大模型不胡说八道")
print("多查询检索命中:", [d.metadata["topic"] for d in docs])'''),
        md('''### 4.2 EnsembleRetriever：BM25 + 向量混合检索

关键词检索（BM25）擅长精确术语，向量检索擅长语义泛化，
两者加权融合是 RAG 检索质量的性价比之王：'''),
        code('''
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25 = BM25Retriever.from_documents(docs, k=2)

ensemble = EnsembleRetriever(
    retrievers=[bm25, vs_retriever],   # 权重默认均分
    weights=[0.4, 0.6],
)
print("混合检索:", [d.metadata["topic"] for d in ensemble.invoke("checkpointer 断点续跑")])'''),
        md('''### 4.3 检索后压缩：只把"相关句子"喂给模型

检索命中 ≠ 全文有用。用 LCEL 手写一条"检索→过滤"链，
先粗检索再让 LLM 挑出真正相关的片段，节省 token 且降噪：'''),
        code('''
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate

compress_prompt = ChatPromptTemplate.from_messages([
    ("system", "从候选文档中摘出与问题直接相关的句子，没有则输出'无'。"),
    ("human", "问题：{q}\\n\\n候选文档：\\n{docs}"),
])

def format_docs(ds):
    return "\\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(ds))

compress_chain = (
    {"q": RunnableLambda(lambda x: x["q"]),
     "docs": RunnableLambda(lambda x: format_docs(vs_retriever.invoke(x["q"])))}
    | compress_prompt | model | StrOutputParser()
)
print(compress_chain.invoke({"q": "MoE 架构的参数量是多少"}))'''),
        md(FAQ_01),
    ]
    dump("01_LangChain核心概念进阶.ipynb", cells)


# ========== 02 LangGraph 设计模式 ==========
def build_02():
    cells = [
        md('''# 进阶教程（二）：LangGraph 状态图设计模式

> 设计模式 = 可复用的图结构解决方案。本讲沉淀 4 个高频模式 + 反模式清单。

## 本讲内容
| 模式 | 一句话 | 典型场景 |
|---|---|---|
| 子图封装 | 把循环图打包成一个节点 | 复用、分层设计 |
| Command 路由 | 节点内部决定下一步 | 动态审批流 |
| Send 并行归约 | 运行时动态分发 Map-Reduce | 多角度评审、批处理 |
| 校验-重试 | 哨兵节点把关输出 | 结构化输出兜底 |'''),
        md(HEADER_MD),
        code(HEADER_CODE),
        md('''## 1. 模式一：子图封装（Subgraph as Node）

"生成 → 校验 → 不合格回炉"的循环图，打包后挂进父图当普通节点。
父图代码完全不感知内部循环——**图的节点可以是图**：'''),
        code('''
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class InnerState(TypedDict):
    text: str
    attempts: int

def write(state: InnerState):
    tip = "" if state["attempts"] == 0 else f"（注意修正：{state['tip'] if 'tip' in state else ''}）"
    r = model.invoke(f"写一句不超过 20 字的产品宣传语，主题：{state['text'][:20]}{tip}")
    return {"text": r.content}

def check(state: InnerState):
    ok = len(state["text"]) <= 25
    return {"text": state["text"], "attempts": state["attempts"] + 1,
            **({} if ok else {"tip": "太长了"})}

def route(state: InnerState):
    return "write" if (len(state["text"]) > 25 and state["attempts"] < 3) else "pass"

inner = StateGraph(InnerState)
inner.add_node("write", write)
inner.add_node("check", check)
inner.add_edge(START, "write")
inner.add_edge("write", "check")
inner.add_conditional_edges("check", route, {"write": "write", "pass": END})
slogan_factory = inner.compile()   # 子图编译产物 = Runnable

# ---- 子图作为父图节点 ----
class OuterState(TypedDict):
    product: str
    slogan: str

def slogan_node(state: OuterState):
    r = slogan_factory.invoke({"text": state["product"], "attempts": 0})
    return {"slogan": r["text"]}

outer = StateGraph(OuterState)
outer.add_node("make_slogan", slogan_node)
outer.add_edge(START, "make_slogan")
outer.add_edge("make_slogan", END)
app = outer.compile()
print(app.invoke({"product": "保温杯", "slogan": ""})["slogan"])'''),
        md('''**要点**：
- 子图与父图**状态独立**（InnerState / OuterState），靠节点函数做输入输出适配
- `create_agent(...)` 返回的也是 CompiledStateGraph，同样可以直接 `add_node` 挂载'''),
        md('''## 2. 模式二：Command 路由（节点内决策）

传统 `add_conditional_edges` 把路由逻辑放在节点外；
`Command` 让节点**在返回结果的同时宣布下一步去哪**——
决策与执行内聚，特别适合"执行完才知道下一步"的场景：'''),
        code('''
from langgraph.types import Command
from pydantic import BaseModel, Field

class ReviewResult(BaseModel):
    score: float = Field(description="质量分 0-10")
    comment: str = Field(description="一句话评语")

class FlowState(TypedDict):
    draft: str
    score: float
    history: Annotated[list, lambda old, new: old + new]

def draft_node(state: FlowState) -> Command:
    """写稿节点：根据上一轮评审分数决定重写还是交付"""
    if state.get("score", 0) >= 8:
        return Command(goto=END, update={"draft": state["draft"]})
    # 首次或低分回炉：写一版新文案（模型每次生成不同版本）
    r = model.invoke("写一句不超过 20 字的保温杯文案，朗朗上口、突出保温卖点。")
    return Command(goto="review", update={"draft": r.content})

def review_node(state: FlowState):
    r = model.with_structured_output(ReviewResult).invoke(
        f"给这句文案打分（0-10）并一句话点评：{state['draft']}")
    print(f"  [review] 分数 {r.score}: {r.comment}")
    return {"score": r.score}

b = StateGraph(FlowState)
b.add_node("draft", draft_node)      # 节点内部用 Command 宣布去向
b.add_node("review", review_node)
b.add_edge(START, "draft")
b.add_edge("review", "draft")        # 回边固定，是否再走由 draft 自己决定
flow = b.compile()
result = flow.invoke({"draft": "", "score": 0, "history": []},
                     config={"recursion_limit": 10})   # 循环安全阀
print("最终文案:", result["draft"])'''),
        md('''**要点**：`Command(goto=..., update=...)` 中 update 写入状态、goto 决定路由；
回边 `review -> draft` 静态存在，但 draft 内部可直接 `goto=END` 跳出，
循环控制权收进节点内部，图结构保持极简。'''),
        md('''## 3. 模式三：Send 动态并行（Map-Reduce）

`Send` 在**运行时**根据状态动态决定分发几份、每份带什么参数——
并行度不再写死在图结构里：'''),
        code('''
import operator
from langgraph.types import Send

class MapState(TypedDict):
    topic: str
    angles: list            # 评审角度（运行时才知道有几个）
    reviews: Annotated[list, operator.add]   # reducer：并行结果自动归并

class WorkerState(TypedDict):
    """Send 的 payload：只带本 worker 需要的字段"""
    topic: str
    angle: str

def dispatch(state: MapState):
    """Map：为每个角度发一个 worker（动态并行）"""
    return [Send("worker", {"topic": state["topic"], "angle": a})
            for a in state["angles"]]

def worker(ws: WorkerState):
    r = model.invoke(f"从「{ws['angle']}」角度，用一句话评价：{ws['topic']}")
    print(f"  [{ws['angle']}] 完成")
    return {"reviews": [f"[{ws['angle']}] {r.content}"]}

b = StateGraph(MapState)
b.add_node("worker", worker)
b.add_conditional_edges(START, dispatch, ["worker"])  # 入口直接动态分发
b.add_edge("worker", END)
mapreduce = b.compile()

r = mapreduce.invoke({"topic": "RAG 技术", "angles": ["实用性", "成本", "风险"], "reviews": []})
for line in r["reviews"]:
    print(line)'''),
        md('''**要点**：
- worker 收到的是 **WorkerState**（Send payload），不是完整 MapState
- 多个 worker **并行**执行，各自的返回值经 `operator.add` reducer 自动归并
- 与 fan-out（`add_edge(["a","b"], "join")`）的区别：Send 的分支数量运行时可变'''),
        md('''## 4. 模式四：校验-重试循环（哨兵节点）

LLM 输出不可信？在出口放一个**哨兵节点**做结构化校验，
不合法就带着错误信息回炉，超过次数走兜底——
rag_qa_project 的 grade_documents 正是该模式的检索版：'''),
        code('''
import re
import json as _json
from pydantic import BaseModel, Field, ValidationError

class Report(BaseModel):
    title: str = Field(min_length=2, max_length=20)
    risk_level: int = Field(ge=1, le=5)

class VS(TypedDict):
    demand: str
    raw: str
    report: dict
    attempts: int

def gen(state: VS):
    hint = "" if state["attempts"] == 0 else "上次输出不合法，请严格遵守 schema。"
    schema = _json.dumps({"title": "2-20字标题", "risk_level": "1到5的整数"},
                         ensure_ascii=False)
    r = model.invoke(
        f"严格按此 JSON 格式输出项目风险报告 {schema}。"
        f"需求：{state['demand']}。{hint}")
    return {"raw": r.content}

def validate(state: VS):
    """哨兵节点：只做校验，不做生成"""
    m = re.search(r"\\{.*\\}", state["raw"], re.S)
    try:
        obj = _json.loads(m.group()) if m else {}
        Report(**obj)          # Pydantic 校验失败会抛 ValidationError
        return {"report": obj, "attempts": state["attempts"] + 1}
    except ValidationError as e:
        return {"attempts": state["attempts"] + 1,
                "raw": f"ERROR: {str(e)[:200]}"}

def route(state: VS):
    if "ERROR" not in state["raw"] or state["attempts"] >= 3:
        return "done"
    return "retry"

b = StateGraph(VS)
b.add_node("gen", gen)
b.add_node("validate", validate)
b.add_edge(START, "gen")
b.add_edge("gen", "validate")
b.add_conditional_edges("validate", route, {"retry": "gen", "done": END})
validator = b.compile()
r = validator.invoke({"demand": "数据库迁移到云上", "raw": "", "report": {}, "attempts": 0})
print("最终报告:", r.get("report") or r["raw"][:150], "| 尝试次数:", r["attempts"])'''),
        md('''## 5. 反模式清单

| 反模式 | 症状 | 正解 |
|---|---|---|
| 巨型节点 | 一个函数又检索又生成又路由 | 单一职责，节点间用状态协作 |
| 隐式全局状态 | 节点读写外部变量 | 一切经过 State（可检查、可回放） |
| 无安全阀循环 | rewrite↔retrieve 死循环烧额度 | 计数器 + recursion_limit 双保险 |
| 状态字段未声明 | TypedDict 没写的字段被静默丢弃 | 补充 schema 声明（常见 KeyError 根因） |
| 并行写同一字段 | InvalidUpdateError | 各分支写独立字段，join 节点汇总 |'''),
        md('''## 6. 常见问题（FAQ）

| 问题 | 原因 | 解决 |
|---|---|---|
| `Send` 的 payload 字段在目标节点读不到 | worker 状态与父状态不同构 | 为 worker 定义独立 TypedDict |
| Command 路由后图死循环 | 忘记 goto=END 的退出条件 | 循环体内必须有可达的出口 |
| 子图状态没传回父图 | 子图 update 的 key 不在父图 schema | 节点函数里做显式字段映射 |
| 归并结果乱序 | 并行节点写同一 list | 用 reducer（operator.add），接受非确定顺序或加序号 |'''),
    ]
    dump("02_LangGraph设计模式.ipynb", cells)


# ========== 03 Runtime 状态更新与回溯 ==========
def build_03():
    cells = [
        md('''# 进阶教程（三）：Runtime 状态更新与时间旅行

> LangGraph 把每一步状态都存成 checkpoint。本讲用好这份"存档"：
> 人工干预（update_state）、历史回放（get_state_history）、分叉重跑。

## 本讲内容
1. 观察状态：`get_state` / `get_state_history`
2. 人工干预：`update_state` + 续跑
3. 时间旅行：从任意历史检查点分叉重放'''),
        md(HEADER_MD),
        code(HEADER_CODE),
        md('''## 1. 观察状态：快照与历史

先建一个带 checkpointer 的多步图，跑完后"翻监控"：'''),
        code('''
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class S(TypedDict):
    messages: Annotated[list, lambda o, n: o + n]
    draft: str

def step_a(state: S):
    return {"draft": "初稿 v1"}

def step_b(state: S):
    r = model.invoke(f"把这句话润色得更优雅：{state['draft']}")
    return {"draft": r.content}

g = StateGraph(S)
g.add_node("step_a", step_a)
g.add_node("step_b", step_b)
g.add_edge(START, "step_a")
g.add_edge("step_a", "step_b")
g.add_edge("step_b", END)
app = g.compile(checkpointer=InMemorySaver())

cfg = {"configurable": {"thread_id": "demo-1"}}
app.invoke({"messages": [], "draft": ""}, config=cfg)

# 当前快照
snap = app.get_state(cfg)
print("当前 draft:", snap.values["draft"][:50])
print("下一节点:", snap.next)'''),
        code('''
# 完整历史：每次节点执行都是一条 checkpoint
for i, h in enumerate(app.get_state_history(cfg)):
    print(f"#{i} next={h.next} draft={str(h.values.get('draft', ''))[:40]!r}")'''),
        md('''**预期输出**（大意）：历史按时间倒序排列（最新在前），
能看到每一步之后的状态切片——这就是"存档"。'''),
        md('''## 2. 人工干预：update_state

运行中/运行后直接改状态。`as_node` 声明"以谁的身份写入"，
决定了下一步从哪里继续——**这是把人类插入工作流的标准姿势**：'''),
        code('''
# 模拟场景：step_a 产出的初稿被人工推翻
app.update_state(
    cfg,
    {"draft": "人工指定的初稿：让 LangGraph 成为 Agent 时代的操作系统"},
    as_node="step_a",   # 冒充 step_a 写入 → 下一步将从 step_b 继续
)
snap = app.get_state(cfg)
print("干预后 draft:", snap.values["draft"][:40])
print("下一步将执行:", snap.next)   # ('step_b',)

# 传 None 续跑：从当前检查点继续执行剩余节点
result = app.invoke(None, config=cfg)
print("续跑结果:", result["draft"][:60])'''),
        md('''**要点**：
- `update_state` 不会执行任何节点，只写状态并追加 checkpoint
- `invoke(None, config)` = "从上次停的地方继续跑"，配合 update_state 即"改档再玩"'''),
        md('''## 3. 时间旅行：从历史检查点分叉

拿到任意历史 checkpoint_id，指定新的 thread_id 重放——
原线程不受影响，天然支持"如果当时……"的 A/B 实验：'''),
        code('''
# 找到"step_a 刚执行完、step_b 还没跑"的历史检查点
history = list(app.get_state_history(cfg))
old = [h for h in history if h.next == ("step_b",)][0]
print("回放点 draft:", old.values["draft"][:30], "| next:", old.next)

# ---- 方式一：同线程重放（直接用历史快照的 config）----
replay = app.invoke(None, old.config)
print("重放结果:", replay["draft"][:50])

# ---- 方式二：新线程分叉（原线程不受影响，可做 A/B 实验）----
fork_cfg = {"configurable": {
    "thread_id": "fork-1",                                        # 新线程
    "checkpoint_id": old.config["configurable"]["checkpoint_id"], # 从历史点开始
}}
# 先用 update_state 把该检查点落进新线程（可顺带修改状态）
app.update_state(fork_cfg, {"draft": "分叉修改稿：换个写法"}, as_node="step_a")
# 再在新线程续跑
fork_result = app.invoke(None, {"configurable": {"thread_id": "fork-1"}})
print("分叉结果:", fork_result["draft"][:50])
print("原线程 draft 不变:", app.get_state(cfg).values["draft"][:40])'''),
        md('''**应用场景**：
| 场景 | 做法 |
|---|---|
| 调试回放 | 从出错前的检查点重跑，定位是哪一步引入的问题 |
| 人工修正 | update_state 改正某个字段后续跑，无需从头再来 |
| A/B 实验 | 同一检查点分叉多个线程，分别换参数/模型对比 |
| 生产审计 | get_state_history 即完整操作留痕 |'''),
        md('''## 4. 常见问题（FAQ）

| 问题 | 原因 | 解决 |
|---|---|---|
| update_state 后状态没变 | 字段不在 State schema | TypedDict 里声明该字段 |
| `invoke(None)` 抛 GraphRecursionError | 图已到 END，None 触发重跑空转 | 先确认 `snap.next` 非空 |
| 时间旅行后消息重复累加 | messages 有 reducer | 分叉前理解 reducer 语义；必要时用 Remove 删除消息 |
| checkpoint_id 从哪来 | 不是手写的 | 从 `state.config["configurable"]["checkpoint_id"]` 读 |
| InMemorySaver 重启就丢 | 内存实现 | 生产换 SqliteSaver / PostgresSaver，API 完全一致 |'''),
    ]
    dump("03_Runtime状态更新与回溯.ipynb", cells)


# ========== 04 流式输出与回调机制 ==========
def build_04():
    cells = [
        md('''# 进阶教程（四）：流式输出与回调机制

> 流式 = 体验（用户等待时间从"总耗时"降到"首 token 时间"）；
> 回调 = 可观测性（token 统计、耗时分析、审计日志）。

## 本讲内容
1. LLM / Chain 的 token 级流式
2. LangGraph 四种 stream_mode 与组合
3. create_agent 的流式
4. 自定义回调 Handler
5. LangSmith 追踪'''),
        md(HEADER_MD),
        code(HEADER_CODE),
        md('''## 1. LLM token 级流式

`stream()` 返回生成器，逐块产出 AIMessageChunk：'''),
        code('''
print("打字机效果: ", end="")
for chunk in model.stream("用 50 字解释什么是向量检索"):
    print(chunk.content, end="", flush=True)
print()'''),
        md('''## 2. LCEL 链的流式

链的 `stream()` 会**穿透**到内部的最小流式单元（这里是模型），
其他环节（prompt 组装、解析）逐条传递，无需改一行代码。'''),
        code('''
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = ChatPromptTemplate.from_template("一句话解释：{x}") | model | StrOutputParser()

for token in chain.stream({"x": "checkpointer"}):
    print(token, end="", flush=True)
print()'''),
        md('''## 3. LangGraph 的 stream_mode

图执行 = 多节点接力，"流什么"有四种选择：

| mode | 流出内容 | 适用 |
|---|---|---|
| `values` | 每步后的**完整状态** | 状态审计 |
| `updates` | 每步的**状态增量**（按节点分组） | 调试、进度展示 |
| `messages` | LLM 的 **token 块**（含元数据） | 前端打字机 |
| `debug` | 全部事件（含任务调度细节） | 深度排障 |'''),
        code('''
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class WS(TypedDict):
    topic: str
    steps: list

def n1(state: WS):
    return {"steps": [f"大纲：{state['topic']}三部分"]}

def n2(state: WS):
    r = model.invoke(f"按此大纲写一句话简介：{state['steps'][0]}")
    return {"steps": state["steps"] + [f"简介：{r.content}"]}

b = StateGraph(WS)
b.add_node("n1", n1); b.add_node("n2", n2)
b.add_edge(START, "n1"); b.add_edge("n1", "n2"); b.add_edge("n2", END)
demo = b.compile()

print("=== updates 模式：只看增量 ===")
for upd in demo.stream({"topic": "RAG", "steps": []}, stream_mode="updates"):
    for node, delta in upd.items():
        print(f"[{node}] 新增: {list(delta.keys())}")

print("\\n=== values 模式：每步完整状态 ===")
for i, v in enumerate(demo.stream({"topic": "RAG", "steps": []}, stream_mode="values")):
    print(f"step{i}: steps={len(v['steps'])}")'''),
        md('''### 多模式组合：进度 + 打字机同时要

传入 mode 列表，每个 chunk 是 `(mode, payload)` 元组——
rag_qa_project 的 `run.py --stream` 正是这种用法：'''),
        code('''
print("回答: ", end="")
for mode, chunk in demo.stream({"topic": "Agent", "steps": []},
                               stream_mode=["updates", "messages"]):
    if mode == "messages":
        msg, meta = chunk
        content = getattr(msg, "content", "")
        if content and meta.get("langgraph_node"):
            print(content, end="", flush=True)
    elif mode == "updates":
        pass  # 需要时在这里展示节点进度
print()'''),
        md('''## 4. create_agent 的流式

Agent 内部有模型循环，`messages` 模式天然只流出**最终回答的 token**
（中间的工具调用消息 content 为空，被过滤掉）：'''),
        code('''
from langchain_core.tools import tool
from langchain.agents import create_agent

@tool
def get_weather(city: str) -> str:
    """查询城市天气。Args: city: 城市名"""
    return {"北京": "晴 25C"}.get(city, "未收录")

agent = create_agent(model=model, tools=[get_weather],
                     system_prompt="简洁回答，必须查工具。")

print("Agent 回答: ", end="")
for mode, chunk in agent.stream(
        {"messages": [{"role": "user", "content": "北京天气？"}]},
        stream_mode=["updates", "messages"]):
    if mode == "messages":
        msg, meta = chunk
        c = getattr(msg, "content", "")
        if c and meta.get("langgraph_node") == "model" and not getattr(msg, "tool_calls", None):
            print(c, end="", flush=True)
print()'''),
        md('''## 5. 自定义回调：token 与耗时统计

继承 `BaseCallbackHandler`，挂到任意 Runnable 上：
`on_llm_new_token` 逐 token 触发，`on_llm_end` 汇总——
自己写一个极简"计费表"：'''),
        code('''
import time
from langchain_core.callbacks import BaseCallbackHandler

class Meter(BaseCallbackHandler):
    """统计 LLM 调用次数、token 数与总耗时"""
    def __init__(self):
        self.calls, self.tokens, self.t0 = 0, 0, time.time()

    def on_chat_model_start(self, serialized, messages, **kw):
        self.calls += 1

    def on_llm_new_token(self, token, **kw):
        self.tokens += 1

    def report(self):
        return (f"调用 {self.calls} 次 | ~{self.tokens} tokens | "
                f"耗时 {time.time() - self.t0:.1f}s")

meter = Meter()
r = model.invoke("用 30 字解释回调机制", config={"callbacks": [meter]})
print(r.content)
print("账单:", meter.report())'''),
        md('''## 6. LangSmith 追踪（已配置）

项目根 `.env` 中 `LANGSMITH_TRACING=true` 已开启，**无需改代码**：
每次运行自动上报到 https://smith.langchain.com （项目名 `langchain_demo`）。

能看到：完整调用树、每步输入输出、token/费用、延迟瀑布图。
本地调试用 `stream_mode="debug"`，线上排障用 LangSmith，互为补充。'''),
        md('''## 7. 常见问题（FAQ）

| 问题 | 原因 | 解决 |
|---|---|---|
| messages 模式流不出 token | 模型未真流式 / 过滤条件太严 | 确认走 stream()；meta 里看 langgraph_node |
| 输出夹杂空 content 块 | 工具调用消息 content 为空 | 加 `if c` 与 `not tool_calls` 过滤 |
| 回调没触发 | 挂错位置 | config={"callbacks": [...]} 传给 invoke/stream |
| updates 里出现 `__end__` | 正常现象 | 遍历时跳过该键 |
| LangSmith 看不到数据 | tracing 未开/网络不通 | 检查 .env 的 LANGSMITH_TRACING 与网络 |'''),
    ]
    dump("04_流式输出与回调机制.ipynb", cells)


# ========== 05 自定义工具与错误处理 ==========
def build_05():
    cells = [
        md('''# 进阶教程（五）：自定义工具与错误处理

> Agent 的下限由工具质量决定，上限由容错能力决定。

## 本讲内容
1. `@tool` 进阶：Pydantic 参数校验
2. 工具异常：错误回流（ToolException + ToolErrorMiddleware）
3. 模型层重试：with_retry
4. 级联兜底：with_fallbacks
5. middleware：审计与错误兜底'''),
        md(HEADER_MD),
        code(HEADER_CODE),
        md('''## 1. @tool 进阶：参数校验与规范

默认 `@tool` 从 docstring/type hints 推断 schema；
复杂参数用 `args_schema` 显式声明——**校验发生在工具执行前**，
模型传错参数会直接报错而不是带病执行：'''),
        code('''
from pydantic import BaseModel, Field
from langchain_core.tools import tool

class TransferArgs(BaseModel):
    """转账工具的参数规范（描述会进入模型上下文）"""
    to: str = Field(description="收款人姓名")
    amount: float = Field(gt=0, le=1_000_000, description="金额，0-100万")

@tool(args_schema=TransferArgs)
def transfer(to: str, amount: float) -> str:
    """向指定用户转账。"""
    return f"已向 {to} 转账 ¥{amount:,.2f}"

# schema 被 Pydantic 把关
print(transfer.invoke({"to": "Alice", "amount": 100}))
try:
    transfer.invoke({"to": "Bob", "amount": -5})   # 违反 gt=0
except Exception as e:
    print("校验拦截:", type(e).__name__)'''),
        md('''## 2. 工具异常：错误回流（ToolException + ToolErrorMiddleware）

> **版本坑（已实测）**：`create_agent` 在 langchain 1.3.x 已移除 `handle_tool_errors`
> 参数（传了直接 `TypeError`）。而且实测发现：工具内抛 `ToolException` 时，
> `create_agent` 默认**直接上抛崩溃**——它内部 `ToolNode` 的默认错误 handler 只吞
> "模型参数校验错误"（`ToolInvocationError`），不吞工具执行期异常。

官方正解：**挂内置中间件 `ToolErrorMiddleware`**（实现 `wrap_tool_call` 钩子），
把工具执行期异常转成 `ToolMessage(status="error")` 喂回模型，让模型自纠：
不用手写图，一行挂载即可：'''),
        code('''
from langchain.agents import create_agent
from langchain.agents.middleware import ToolErrorMiddleware
from langchain_core.tools import ToolException, tool

@tool
def query_order(order_id: str) -> str:
    """查询订单状态。Args: order_id: 订单号，形如 ORD-123"""
    if not order_id.startswith("ORD-"):
        raise ToolException(f"订单号格式错误：{order_id}，应以 ORD- 开头")
    return "已发货"

# 错误 handler：只处理 ToolException -> 可读文案（喂回模型，让模型自纠）
def on_error(exc, request) -> str | None:
    if isinstance(exc, ToolException):
        return f"工具执行出错：{exc}。请检查参数格式后重试。"
    return None   # 其他异常原样上抛，不暴露给模型

agent = create_agent(
    model=model,
    tools=[query_order],
    middleware=[ToolErrorMiddleware(on_error)],   # 错误回流
)

# system 强制走工具且不改参数格式 -> 必然触发 ToolException
r = agent.invoke({"messages": [
    {"role": "system", "content": "查询订单必须用工具。用户给什么订单号就直接查什么，不要修改格式。"},
    {"role": "user", "content": "帮我查订单 12345 的状态"}]})
for m in r["messages"]:
    tag = "  <- error 回流" if getattr(m, "status", None) == "error" else ""
    print(f"[{type(m).__name__}] {str(m.content)[:90]}{tag}")'''),
        md('''**要点**：
- `on_error(exc, request)` 的**返回值决定命运**：返回字符串 → 转成
  `ToolMessage(status="error")` 回流模型（模型自纠）；返回 `None` → 原样上抛
- **处理是 opt-in 的**：只处理你在 `on_error` 里返回了内容的异常；漏参、类型错这类
  参数校验错误由 `ToolNode` 上游拦截，到不了 `on_error`——它只处理工具**执行期**异常
- 输出里 `status=error` 的 `ToolMessage` 就是"错误已回流给模型"的信号
- 可选参数：`aon_error`（异步 handler）、`tools=[...]`（只对指定工具生效）
- **经验法则**：参数类错误用 ToolException（模型能自纠）；系统类错误（DB 挂了）
  直接抛（重试也没用，走 fallbacks）
- 需要手写自定义图时，等价写法是 `ToolNode(handle_tool_errors=handler)`（见 FAQ）'''),
        md('''## 3. 模型层重试：with_retry

网络抖动、限流是常态。`with_retry` 基于 tenacity，
指数退避自动重试：'''),
        code('''
import time

# 模拟一个 30% 概率限流的服务（前两次必失败用计数器演示）
class FlakyModel:
    calls = 0
    def invoke(self, x):
        FlakyModel.calls += 1
        if FlakyModel.calls <= 2:
            raise ConnectionError("模拟 429 限流")
        return f"第 {FlakyModel.calls} 次调用成功"

from langchain_core.runnables import RunnableLambda
flaky = RunnableLambda(FlakyModel().invoke)

# with_retry：指数退避重试（此处不真 sleep，仅演示结构）
retryable = flaky.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=False,
    retry_if_exception_type=(ConnectionError,),
)
print(retryable.invoke("test"))
print("总调用次数:", FlakyModel.calls)   # 3 次调用 = 2 次失败 + 1 次成功'''),
        md('''真实模型直接：
```python
robust_model = model.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)
```
配合 `retry_if_exception_type=(RateLimitError,)` 可只对限流重试。'''),
        md('''## 4. 级联兜底：with_fallbacks

重试解决"偶发失败"，fallbacks 解决"持续不可用"。
首选失败 N 次后自动切换备用模型，对上层完全透明：'''),
        code('''
from langchain_deepseek import ChatDeepSeek

# 备用通道：同厂商不同档位（生产可换成不同厂商实现真正容灾）
backup = ChatDeepSeek(model="deepseek-chat", temperature=0.5)

def always_fail(x):
    raise ConnectionError("主通道持续故障")

from langchain_core.runnables import RunnableLambda
primary = RunnableLambda(always_fail)

robust = primary.with_fallbacks([backup])
r = robust.invoke("用 20 字说明什么是容灾")
print("兜底回答:", r.content)'''),
        md('''## 5. middleware：审计与错误兜底

`create_agent` 的 `middleware` 参数提供 Agent 级切面。
用类中间件实现"模型调用失败 → 兜底文案"（洋葱模型包裹模型调用）：'''),
        code('''
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.messages import AIMessage

class SafeGuardMiddleware(AgentMiddleware):
    """审计 + 兜底：包裹每一次模型调用"""
    calls = 0

    async def wrap_model_call(self, request: ModelRequest, call_next):
        SafeGuardMiddleware.calls += 1
        print(f"  [audit] 第 {SafeGuardMiddleware.calls} 次模型调用")
        try:
            return await call_next(request)
        except Exception as e:
            print(f"  [safe] 模型调用失败，兜底: {type(e).__name__}")
            return AIMessage(content=f"服务暂时不可用，请稍后重试（{type(e).__name__}）")

safe_agent = create_agent(
    model=model,
    tools=[],
    system_prompt="你是简洁的助手。",
    middleware=[SafeGuardMiddleware()],
)
r = safe_agent.invoke({"messages": [{"role": "user", "content": "你好"}]})
print("回答:", r["messages"][-1].content[:80])'''),
        md('''## 6. 错误处理策略速查

| 层级 | 机制 | 处理什么 |
|---|---|---|
| 工具层 | ToolException + ToolErrorMiddleware | 参数错误（模型可自纠） |
| 模型层 | with_retry | 网络抖动、限流 |
| 通道层 | with_fallbacks | 主服务持续不可用 |
| Agent 层 | middleware wrap_model_call | 统一审计、兜底、降级 |
| 图层 | 哨兵节点 + 兜底分支（见教程二） | 输出质量不合格 |'''),
        md('''## 7. 常见问题（FAQ）

| 问题 | 原因 | 解决 |
|---|---|---|
| 工具内抛异常 Agent 直接崩 | create_agent 默认只吞参数校验错（`handle_tool_errors` 参数已移除，工具执行期异常会上抛） | 挂 `ToolErrorMiddleware(on_error)`；自定义图则用 `ToolNode(handle_tool_errors=handler)` |
| with_retry 越重试越慢 | 指数退避生效（预期行为） | 合理设置 stop_after_attempt（3 次为宜） |
| fallbacks 备用也失败 | 级联深度不够 | 再挂一个静态 Runnable 兜底作最底层 |
| middleware 不生效 | 用了装饰器却没传参 | 类中间件必须传 middleware=[...] |
| wrap_model_call 是 async | 钩子设计为协程 | def 改 async def，用 await call_next(request) |'''),
    ]
    dump("05_自定义工具与错误处理.ipynb", cells)


# ========== 06 最佳实践 ==========
def build_06():
    cells = [
        md('''# 进阶教程（六）：最佳实践——模块化、配置、测试与性能

> 以本教程的 `rag_qa_project/` 为参照系，讲清 LLM 应用的工程化四件套。

## 本讲内容
1. 模块化代码结构（分层 + 依赖注入）
2. 配置管理（pydantic-settings）
3. 测试策略（三层金字塔，零额度回归）
4. 性能优化（并行 / batch / 上下文裁剪）'''),
        md(HEADER_MD),
        code(HEADER_CODE),
        md('''## 1. 模块化代码结构

LLM 应用容易写成"一坨 notebook"，生产化的第一步是分层：

```
rag_qa_project/
├── config.py    # 配置层：所有魔法数字的唯一来源
├── ingest.py    # 数据层：知识库 -> 向量库（离线任务）
├── graph.py     # 编排层：LangGraph 工作流（核心逻辑）
├── run.py       # 接口层：CLI/API 薄壳（只做 IO 与参数解析）
└── tests/       # 测试层：Fake 注入，零额度回归
```

**三条纪律：**
1. **依赖注入**：`build_graph(model=None, vectorstore=None)` 把基础设施作为参数，
   测试塞 Fake、生产塞真货——`graph.py` 对"用什么模型"零假设
   （P3 起注入的是 **vectorstore** 而非 retriever：`retrieve` 节点要在每次调用时
   传 `filter` 做按用户隔离，而 retriever 的检索参数在构造期就冻结了）
2. **节点纯函数**：输入 state → 输出增量 update，不碰全局变量（可回放、可测试）
3. **接口层最薄**：run.py 不写业务逻辑，换 Web API 时图和节点原封不动'''),
        md('''## 2. 配置管理

规则：**代码里不允许出现裸常量**，一切进 `config.py`。
`pydantic-settings` 提供带类型校验的环境变量覆盖：'''),
        code('''
import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", extra="ignore")

    model_name: str = "deepseek-chat"
    temperature: float = 0.1
    top_k: int = 4
    max_retries: int = 3

s = AppSettings()
print("默认配置:", s.model_name, s.temperature, s.top_k)

# 环境变量覆盖（生产发布不改代码）
os.environ["APP_TOP_K"] = "10"
os.environ["APP_TEMPERATURE"] = "0.5"
s2 = AppSettings()
print("覆盖后:", s2.temperature, s2.top_k)

# 类型错误会被拦截，而不是带病运行
os.environ["APP_TOP_K"] = "abc"
try:
    AppSettings()
except Exception as e:
    print("校验拦截:", type(e).__name__)'''),
        md('''## 3. 测试策略：三层金字塔

| 层级 | 对象 | 手段 | 额度消耗 |
|---|---|---|---|
| 单元测试 | 节点函数、路由逻辑 | FakeChatModel 注入 | 0 |
| 集成测试 | 整图执行 | Fake 模型 + Fake 检索器跑全图 | 0 |
| 冒烟测试 | 端到端质量 | 真实 LLM 小样本 | 少量 |

`rag_qa_project/tests/test_graph.py` 已完整实现前两层。核心技巧：
**让 Fake 模型实现 `with_structured_output`**（真模型有、Fake 默认没有）：'''),
        code('''
# 单元测试示范：验证路由逻辑（不碰 LLM）
from typing import TypedDict

class RAGState(TypedDict):
    question: str
    documents: list
    rewrites: int

def decide(documents: list, rewrites: int, max_rewrites: int = 2):
    """rag_qa_project 的路由函数（简化版，纯逻辑可直接测）"""
    if documents:
        return "generate"
    if rewrites < max_rewrites:
        return "rewrite"
    return "generate"

# 三条路径全覆盖
assert decide(["d1"], 0) == "generate"       # 有文档 -> 生成
assert decide([], 0) == "rewrite"            # 无文档且可重写
assert decide([], 2) == "generate"           # 重写耗尽 -> 兜底
print("路由逻辑测试全部通过（0 额度）")'''),
        code('''
# 集成测试示范：在 notebook 里直接跑 rag_qa_project 的离线测试套件
import subprocess, sys
from pathlib import Path

proj = ROOT / "advanced_tutorial" / "rag_qa_project"
r = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=line"],
    cwd=proj, capture_output=True, text=True, encoding="utf-8", errors="replace")
print(r.stdout[-1500:])'''),
        md('''## 4. 性能优化

### 4.1 并行是第一杠杆

| 手段 | 语义 | 收益 |
|---|---|---|
| `RunnableParallel` | 结构性并行 | 分支耗时取 max 而非 sum |
| `.batch([...])` | 数据级并行 | 逐条→并发，内置节流 |
| `Send`（教程二） | 图内动态并行 | Map-Reduce 场景 |'''),
        code('''
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

qa = ChatPromptTemplate.from_template("一句话回答：{q}") | model | StrOutputParser()
questions = [{"q": q} for q in ["什么是RAG", "什么是Agent", "什么是Embedding"]]

# 串行
t0 = time.time()
serial = [qa.invoke(q) for q in questions]
t_serial = time.time() - t0

# batch 并发（内部线程池，自动节流）
t0 = time.time()
batched = qa.batch(questions)
t_batch = time.time() - t0

print(f"串行 {t_serial:.1f}s | batch {t_batch:.1f}s | 加速比 {t_serial/t_batch:.1f}x")'''),
        md('''### 4.2 上下文裁剪：trim_messages

多轮对话越聊越长，token 成本线性膨胀。
`trim_messages` 按 token 上限裁剪历史（优先保留 system + 最新消息）：'''),
        code('''
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages

history = [SystemMessage(content="你是助手")] + [
    m
    for i in range(1, 8)
    for m in (HumanMessage(content=f"第{i}个问题：{i}+{i}=?"),
              AIMessage(content=f"等于 {i*2}"))
]

trimmed = trim_messages(
    history,
    max_tokens=120,
    strategy="last",                        # 保留最近的消息
    token_counter=len,                      # 演示用字符数当 token 数
    include_system=True,                    # system 永不裁
    start_on="human",                       # 从 HumanMessage 开始（成对裁剪）
)
print(f"裁剪前 {len(history)} 条 -> 裁剪后 {len(trimmed)} 条")
print("首条:", trimmed[0].content[:20], "| 末条:", trimmed[-1].content)'''),
        md('''### 4.3 优化清单（按性价比排序）

1. **并行化**（batch / Parallel / Send）——不改质量纯提速
2. **上下文裁剪**（trim_messages）——直接省 token 钱
3. **小模型分工**——路由/评分用轻量档，只有生成用旗舰（rag_qa_project 的 grade 就适合）
4. **缓存**——`set_llm_cache(SQLiteCache(...))` 缓存重复问题；嵌入用 CacheBackedEmbeddings
5. **温度调低**——RAG 类任务 0.1 既省心又减少重试
6. **recursion_limit 设防线**——循环图失控 = 烧钱事故，务必显式设置'''),
        md('''## 5. 常见问题（FAQ）

| 问题 | 原因 | 解决 |
|---|---|---|
| 测试里 with_structured_output 报错 | Fake 模型未实现 | 参考 rag_qa_project/tests 的覆写写法 |
| subprocess 跑 pytest 中文乱码 | Windows GBK 控制台 | encoding="utf-8", errors="replace" |
| batch 没提速 | 目标服务限流 | 并发是本地开销的并行；服务端限流时收益有限 |
| trim 后消息不成对 | 未设 start_on | start_on="human" 保持对话完整性 |
| 改了 config 不生效 | .env 缓存/pydantic 实例早创建 | 重新实例化 Settings；检查 env_prefix |
| notebook 一切正常，脚本跑不了 | 工作目录/路径假设不同 | 用绝对路径 + Path(__file__) 锚定（见 run.py） |'''),
    ]
    dump("06_最佳实践_模块化配置测试与性能.ipynb", cells)


# ========== README ==========
def build_readme():
    content = '''# advanced_tutorial — LangChain / LangGraph 进阶教程

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

## 学习路径建议

```
01 -> 02 -> 03 -> rag_qa_project（动手跑通） -> 04 -> 05 -> 06
```

## 环境要求

```bash
cd my_langchain_demo
uv add langchain langgraph langchain-deepseek langchain-classic \\
      langchain-community langchain-huggingface langchain-chroma \\
      pydantic-settings python-dotenv pytest
```

- 根目录 `.env` 需配置 `DEEPSEEK_API_KEY`（真实 LLM，无本地降级）
- 检索章节使用本地 `BAAI/bge-small-zh-v1.5` 嵌入（首次自动经 hf-mirror 下载）
- `rag_qa_project` 完整运行指引见其目录下 `README.md`

## 已验证事项

- 所有 notebook 代码基于当前 `.venv`（langchain 1.3.14 / langgraph 1.2.10）实测 API 编写
- `rag_qa_project`：4 个离线测试全过；入库、结构图、真实问答、重写循环均已端到端验证
- 经典检索器（Ensemble/MultiQuery）从 `langchain_classic.retrievers` 导入（community 包已移除）
- `create_agent` 从 `langchain.agents` 导入（当前 langgraph 版本的 prebuilt 未导出）

## 重新生成 notebook

```bash
cd advanced_tutorial
python _build_advanced.py
```
'''
    (OUT / "README.md").write_text(content, encoding="utf-8")
    print("written: README.md")


if __name__ == "__main__":
    build_01()
    build_02()
    build_03()
    build_04()
    build_05()
    build_06()
    build_readme()
    print("\\n全部生成完毕")

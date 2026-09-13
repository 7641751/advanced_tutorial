# -*- coding: utf-8 -*-
"""生成 09_Human_in_the_loop.ipynb（基于官方 interrupts / event-streaming 指南重写）。

运行方式（在 advanced_tutorial 目录下）：
    python _build_nb09.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _build_advanced import md, code, dump, HEADER_CODE, HEADER_MD

OUT = Path(__file__).resolve().parent

# v3 流式协议在 langgraph 1.2.10 仍是实验性 API，忽略 Beta 警告
HEADER_CODE_09 = HEADER_CODE + '''
# ---- 09 讲附加：v3 流式协议在 langgraph 1.2.10 仍是实验性 API，忽略 Beta 警告 ----
warnings.filterwarnings("ignore", message="The v3 streaming protocol.*")
'''


def build_09():
    cells = [
        md('''# 进阶教程（九）：Human-in-the-loop 人工介入

> `interrupt()` 暂停 + `Command(resume=...)` 恢复——LangGraph 的 HIL 统一原语。
> 本讲按官方 [interrupts 指南](https://docs.langchain.com/oss/python/langgraph/interrupts) 重写，
> 覆盖 5 大官方模式：审批、编辑状态、工具内中断、多中断 ID 映射、验证循环。

## 本讲内容
1. 概念：interrupt 与 resume（动态暂停、checkpointer 前提、resume 4 要点）
2. 最小示例（v3 流式：`stream.interrupts` / `stream.interrupted` / `stream.output`）
3. HITL 交互循环（官方 while-loop 模式）
4. 审批流：批准 / 驳回（Approval workflows）
5. 人工编辑状态（Review and edit state + `update_state`）
6. 工具内 interrupt（Interrupts in tools）
7. 多 interrupt 与 ID 映射（Handling multiple interrupts）
8. 验证循环（Validating human input：条件边 vs `while` 反模式）
9. 规则与陷阱（Rules of interrupts）
10. 静态断点（调试/测试专用）
11. 常见问题（FAQ）'''),
        md(HEADER_MD),
        code(HEADER_CODE_09),
        md('''## 1. 概念：interrupt 与 resume

- **`interrupt(value)`**：在节点内**任意位置**调用，立即暂停整条工作流，
  把 `value`（任意 JSON 可序列化值）抛给调用方。配合 v3 流式时出现在
  `stream.interrupts`（用 `invoke` 则在结果的 `__interrupt__` 键里）
- **`Command(resume=...)`**：用**同一个 thread_id** 重新调用图，把人工决定注入到
  `interrupt()` 的返回值位置，工作流从暂停点继续

> **关键前提**：图必须用 **checkpointer** 编译——`interrupt` 依赖状态持久化，
> 否则报错、无法恢复。`thread_id` 就是你的"持久化指针"：复用=从同一 checkpoint 续跑，
> 换新值=全新线程。

### 官方 resume 四要点

1. 恢复时必须用**同一个 thread_id**
2. `Command(resume=...)` 的值成为 `interrupt()` 的返回值
3. 恢复时节点**从开头重新执行**——`interrupt()` 之前的代码会再跑一遍（见第 9 节幂等性）
4. resume 值可以是**任意 JSON 可序列化值**

> ⚠️ 官方警告：`Command(resume=...)` 是**唯一**允许作为输入传给
> `invoke()` / `stream()` / `stream_events()` 的 Command 形态；`Command(update=...)`
> 等是节点函数**返回**用的，别拿它当输入继续多轮对话。'''),
        md('''## 2. 最小示例：暂停与恢复（v3 流式）

官方推荐用 `stream_events(..., version="v3")` 拿到**类型化投影**：
- `stream.output`：最终输出（访问它也会驱动流跑完）
- `stream.interrupted`：本次运行是否被中断
- `stream.interrupts`：暂停点载荷（含 `value` 与 `id`）

一个"审批"节点，先暂停、再批准：'''),
        code('''
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

class S(TypedDict):
    question: str
    approved: bool
    final: str

def review(state: S):
    # 暂停：把待审内容抛给外部；返回值 = 外部 resume 注入的值
    decision = interrupt({"question": state["question"], "hint": "请人工审批"})
    return {"approved": decision["approve"]}

def publish(state: S):
    return {"final": "已发布" if state["approved"] else "已驳回"}

b = StateGraph(S)
b.add_node("review", review)
b.add_node("publish", publish)
b.add_edge(START, "review")
b.add_edge("review", "publish")
b.add_edge("publish", END)
app = b.compile(checkpointer=InMemorySaver())   # 必须！

cfg = {"configurable": {"thread_id": "t1"}}
s = app.stream_events({"question": "上线新功能", "approved": False, "final": ""},
                      cfg, version="v3")
_ = s.output   # 驱动流跑完
print("interrupted:", s.interrupted)
print("暂停载荷:", s.interrupts[0].value)
print("interrupt id:", s.interrupts[0].id)'''),
        md('''**预期输出**：
```
interrupted: True
暂停载荷: {'question': '上线新功能', 'hint': '请人工审批'}
interrupt id: <32 位十六进制>
```

`interrupts` 是 `Interrupt` 对象元组，`value` 就是 `interrupt()` 传进去的内容，
`id` 用于多中断时精准配对（见第 7 节）。'''),
        code('''
# ---- 第二次调用：注入人工决定（批准） ----
s2 = app.stream_events(Command(resume={"approve": True}), cfg, version="v3")
print("interrupted:", s2.interrupted)
print("批准结果:", s2.output["final"])'''),
        md('''**要点**：`Command(resume={"approve": True})` 的值成为 `review` 里
`interrupt()` 的返回值 `decision`，节点据此继续。两次调用**同一个 thread_id**。

> 老式 `invoke` 也能用（暂停信息在 `result["__interrupt__"]`），但官方建议
> 需要流式投影时统一用 `stream_events(..., version="v3")`。'''),
        md('''## 3. HITL 交互循环（官方 while-loop 模式）

真实应用是**循环**：驱动流 → 若 `interrupted` 则取载荷、请求人工、`Command(resume)`
再跑 → 直到不中断。官方姿势：

```python
stream_input = initial_input
while True:
    stream = graph.stream_events(stream_input, config, version="v3")
    if not stream.interrupted:
        return stream.output
    payload = stream.interrupts[0].value
    stream_input = Command(resume=user_responder(payload))   # 人工介入
```

封装成可复用函数，并用第 2 节的图演示：'''),
        code('''
def run_until_done(graph, user_responder, initial_input, config):
    """官方 HITL 循环：驱动流 → 检查中断 → 人工应答 → resume，直到跑完。"""
    stream_input = initial_input
    while True:
        stream = graph.stream_events(stream_input, config, version="v3")
        if not stream.interrupted:
            return stream.output
        payload = stream.interrupts[0].value
        stream_input = Command(resume=user_responder(payload))

def human_approve(payload):
    print(f">>> 人工收到: {payload}")
    return {"approve": True}

cfg3 = {"configurable": {"thread_id": "t3"}}
final = run_until_done(app, human_approve,
                       {"question": "发布 v2.0", "approved": False, "final": ""}, cfg3)
print("最终:", final["final"])'''),
        md('''这个循环把"暂停-交互-恢复"封装成一行调用，是搭建真实 HIL 服务
（如 FastAPI + WebSocket）的骨架：`user_responder` 换成向前端发消息、
等用户回复即可。'''),
        md('''## 4. 审批流：批准 / 驳回（Approval workflows）

官方 *Approve or reject* 模式：`interrupt()` 返回布尔决定，
用**条件边**路由到不同分支。这里用真实 LLM 起草文案，人工可多次驳回回炉（最多 3 次）——
**驳回循环用条件边实现**，不要在节点里写 `while True + interrupt`（原因见第 8/9 节）：'''),
        code('''
class Flow(TypedDict):
    draft: str
    attempts: int
    outcome: str

def draft(state: Flow):
    tip = "" if state["attempts"] == 0 else "（注意：上一版被驳回，请重新措辞）"
    r = model.invoke(f"写一句不超过 20 字的产品文案，主题：智能台灯{tip}")
    return {"draft": r.content}

def approve(state: Flow):
    verdict = interrupt({"draft": state["draft"], "ask": "批准还是驳回？"})
    return {"attempts": state["attempts"] + 1,
            "outcome": "approved" if verdict["ok"] else "rejected"}

def route(state: Flow):
    if state["outcome"] == "approved":
        return "publish"
    if state["attempts"] >= 3:
        return "giveup"
    return "draft"          # 驳回 → 回炉重写（条件边循环）

def publish(state: Flow):
    return {"outcome": "PUBLISHED"}

def giveup(state: Flow):
    return {"outcome": "GAVE_UP"}

g = StateGraph(Flow)
for name, fn in [("draft", draft), ("approve", approve),
                 ("publish", publish), ("giveup", giveup)]:
    g.add_node(name, fn)
g.add_edge(START, "draft")
g.add_edge("draft", "approve")
g.add_conditional_edges("approve", route,
                        {"publish": "publish", "draft": "draft", "giveup": "giveup"})
g.add_edge("publish", END)
g.add_edge("giveup", END)
flow = g.compile(checkpointer=InMemorySaver())

cfg4 = {"configurable": {"thread_id": "flow-1"}}

def ask_human(payload):
    print(f">>> 草稿待审: {payload['draft'][:40]}...")
    return {"ok": True}    # 模拟人工：直接批准

final = run_until_done(flow, ask_human,
                       {"draft": "", "attempts": 0, "outcome": ""}, cfg4)
print("最终结果:", final["outcome"])'''),
        md('''**执行过程**：`draft → approve(暂停) → resume → 批准 → publish`。
把 `ask_human` 里的 `{"ok": True}` 换成 `{"ok": False}` 并连续运行，
就能看到"驳回 → 回炉重写 → 再审批"的循环（见第 8 节验证循环如何规范多次交互）。

**与第 2 节最小示例的区别**：这里 resume 值决定**走向**（批准/驳回 → 不同分支），
而不是决定单节点的返回值。这是审批流的标准形态。'''),
        md('''## 5. 人工编辑状态（Review and edit state）

官方 *Review and edit state* 模式：`interrupt()` 把待改内容抛给人工，
resume 回来的就是**修改后的内容**，直接写回状态——适合纠错 LLM 输出、
补全缺失信息：'''),
        code('''
class EditState(TypedDict):
    draft: str
    final_text: str

def gen(state: EditState):
    r = model.invoke("写一句产品 slogan：AI 降噪耳机")
    return {"draft": r.content}

def review_node(state: EditState):
    # 暂停并展示内容；resume 值 = 人工编辑后的文本
    edited = interrupt({"instruction": "审阅并编辑以下文案",
                        "draft": state["draft"]})
    return {"final_text": edited}

g3 = StateGraph(EditState)
g3.add_node("gen", gen)
g3.add_node("review_node", review_node)
g3.add_edge(START, "gen")
g3.add_edge("gen", "review_node")
g3.add_edge("review_node", END)
edit_flow = g3.compile(checkpointer=InMemorySaver())

cfg5 = {"configurable": {"thread_id": "edit-1"}}

def reviewer(payload):
    print(f">>> 原文: {payload['draft']}")
    return "极简降噪，静在耳边"   # 人工编辑后的文本

final = run_until_done(edit_flow, reviewer,
                       {"draft": "", "final_text": ""}, cfg5)
print("定稿:", final["final_text"])'''),
        md('''### 5.1 变体：先 `update_state` 再 resume

有时人工要改的是**状态里其他字段**（不经过 interrupt 的返回值）——
官方做法是 `graph.update_state(config, {...}, as_node=...)` 直接改写 checkpoint
里的状态，再 resume。演示：先跑到暂停点，人工修正草稿，再恢复：'''),
        code('''
# 新线程：先跑到暂停点
cfg5b = {"configurable": {"thread_id": "edit-2"}}
s = edit_flow.stream_events({"draft": "", "final_text": ""}, cfg5b, version="v3")
_ = s.output
print("暂停在 review_node:", s.interrupted)

# 人工直接改写状态字段（模拟人工修正草稿，绕过模型）
edit_flow.update_state(cfg5b, {"draft": "人工修正后的草稿（未走模型）"},
                       as_node="review_node")

def reviewer2(payload):
    print(">>> 重跑时看到的草稿:", payload["draft"])
    return payload["draft"] + "（已确认）"

final2 = run_until_done(edit_flow, reviewer2, Command(resume="继续"), cfg5b)
print("二次定稿:", final2["final_text"])'''),
        md('''`update_state(config, values, as_node=...)` 是"时间旅行/状态编辑"的入口，
与 03 讲的回溯一脉相承：先改状态、再恢复执行，适合人工介入时**顺手纠错**。'''),
        md('''## 6. 工具内 interrupt（Interrupts in tools）

官方 *Interrupts in tools* 模式：把 `interrupt()` 直接放进**工具函数**——
工具被调用时暂停，等待人工审阅/修改这次调用。配合 agent（`create_agent`）即可实现
"高危操作（发邮件、转账、删库）必须人工批准"：'''),
        code('''
from langchain.tools import tool
from langchain.agents import create_agent

@tool
def send_email(to: str, subject: str, body: str):
    """发送邮件给收件人。"""
    interrupt({"action": "发送邮件", "to": to,
               "subject": subject, "body": body})   # 工具内暂停，等待批准
    return "邮件已发送"

agent = create_agent(
    model=model, tools=[send_email],
    system_prompt="你是邮件助手。用户要求发邮件时，必须调用 send_email 工具，发完即结束。",
    checkpointer=InMemorySaver(), name="mail_agent",
)

cfg6 = {"configurable": {"thread_id": "mail-1"}}
s = agent.stream_events({"messages": [{"role": "user", "content": "给 bob 发邮件，主题：周报，内容：已完成"}]},
                        cfg6, version="v3")
_ = s.output
print("interrupted:", s.interrupted)
print("工具待批:", s.interrupts[0].value)'''),
        code('''
# 人工批准 → 工具继续执行，agent 收尾
s2 = agent.stream_events(Command(resume=True), cfg6, version="v3")
print("interrupted:", s2.interrupted)
print("agent 最终回复:", s2.output["messages"][-1].content)'''),
        md('''**要点**：
- 工具内 `interrupt()` 的暂停载荷就是"这次工具调用的参数"，
  人工可以批准（resume=True）或修改参数后继续（resume=修改后的 dict）
- agent 必须配 checkpointer（`create_agent(..., checkpointer=...)`），
  否则工具内 interrupt 无法暂停
- 与"节点内 interrupt"完全同构：同一套 `stream_events` + `Command(resume=...)` 交互'''),
        md('''## 7. 多 interrupt 与 ID 映射（Handling multiple interrupts）

**并行分支同时暂停**时，一次运行会收集多个暂停点。resume 时用
`{interrupt_id: resume_value}` 字典一次恢复全部，ID 保证配对不串味
（官方 *Handling multiple interrupts* 模式；08 讲已有完整演示，这里给最小复现）：'''),
        code('''
from typing import Annotated
import operator

class ReviewState(TypedDict):
    vals: Annotated[list, operator.add]

def node_a(state: ReviewState):
    ans = interrupt("question_a")
    return {"vals": [f"a:{ans}"]}

def node_b(state: ReviewState):
    ans = interrupt("question_b")
    return {"vals": [f"b:{ans}"]}

g4 = StateGraph(ReviewState)
g4.add_node("a", node_a).add_node("b", node_b)
g4.add_edge(START, "a").add_edge(START, "b")
g4.add_edge("a", END).add_edge("b", END)
wf = g4.compile(checkpointer=InMemorySaver())

cfg7 = {"configurable": {"thread_id": "multi"}}
s = wf.stream_events({"vals": []}, cfg7, version="v3")
_ = s.output
print("并行暂停点数:", len(s.interrupts))

resume_map = {i.id: f"ans-{i.value}" for i in s.interrupts}   # ID -> resume 值
s2 = wf.stream_events(Command(resume=resume_map), cfg7, version="v3")
print("最终状态:", s2.output)'''),
        md('''**注意**：多中断**必须靠并行分支同时触发**。顺序流程里连写两个 `interrupt()`，
第一个就暂停了，第二个要等下次 resume 才会轮到（顺序暂停，不是并行）。'''),
        md('''## 8. 验证循环（Validating human input）

需要"输入不合法就重新问"时，官方推荐姿势是：

1. 把提示词（含错误提示）存进状态字段
2. 节点内 **恰好调用一次** `interrupt()`
3. 输入非法 → 更新 `pending` 字段，用**条件边**路由回本节点重新问
4. 合法 → 走向 END

> ⚠️ 官方明令**禁止**在节点里写 `while True + interrupt()`：
> 每次 resume 节点都从开头重放，第一次 resume 重放 1 轮、第二次重放 2 轮……
> 循环体内代码指数级重复执行。条件边才是正确写法。'''),
        code('''
class FormState(TypedDict):
    age: int | None
    pending: str | None

def get_age(state: FormState):
    ans = interrupt(state["pending"] or "你多大了？")
    try:
        return {"age": int(ans), "pending": None}
    except ValueError:
        return {"pending": f"'{ans}' 不是有效年龄，请重新输入"}

def route(state: FormState):
    return END if state["age"] is not None else "get_age"

g5 = StateGraph(FormState)
g5.add_node("get_age", get_age)
g5.add_edge(START, "get_age")
g5.add_conditional_edges("get_age", route, {"get_age": "get_age", END: END})
form = g5.compile(checkpointer=InMemorySaver())

cfg8 = {"configurable": {"thread_id": "form-1"}}
answers = iter(["三十", "abc", 30])   # 两次非法，一次合法

def responder(payload):
    print(">>> 表单提问:", payload)
    return next(answers)

final = run_until_done(form, responder, {"age": None, "pending": None}, cfg8)
print("最终 age:", final["age"])'''),
        md('''每次 resume 只进一次 `get_age`、只调一次 `interrupt()`，非法输入通过条件边
回到自己（re-prompt 提示会更新），合法后退出——**没有重复执行**。'''),
        md('''## 9. 规则与陷阱（Rules of interrupts）

官方 *Rules of interrupts* 的核心纪律（08 讲已实测前两条，这里汇总全量）：

| 规则 | 说明 | 正确做法 |
|---|---|---|
| **不要 try/except 包住 interrupt** | 被捕获后不暂停、静默继续，人工介入点丢失 | interrupt 放 try 外，或捕获后 re-raise |
| **节点内多 interrupt 按索引匹配** | resume 值按**调用顺序**严格配对 | interrupt 调用顺序保持稳定，不可条件性跳过 |
| **interrupt 前的副作用必须幂等** | resume 时节点从头重跑，之前的代码会再执行 | 副作用放 interrupt 之后，或拆到独立节点 |
| **禁止 `while True + interrupt` 循环** | 每次 resume 重放历史迭代 → 指数级重执行 | 条件边回环（见第 8 节） |
| **resume 前别改节点结构** | 打乱 checkpoint 与 resume 值配对 → 重放失败 | 代码结构保持稳定 |

> 之所以有这些规则，都源于同一条机制：**resume 时节点从开头重新执行**，
> 已完成的结果靠 checkpoint 复用——所以 interrupt 之前的代码会再跑，必须幂等。'''),
        md('''## 10. 静态断点（调试/测试专用）

`interrupt_before` / `interrupt_after` 在**编译时**或**运行时**指定，
在节点执行前/后强制暂停——无需改动业务代码。官方定位：**主要用于调试和测试**，
不推荐用于生产 HIL（那是动态 `interrupt()` 的职责）：'''),
        code('''
# 编译时：approve 之前必停
g6 = StateGraph(Flow)
for name, fn in [("draft", draft), ("approve", approve), ("publish", publish)]:
    g6.add_node(name, fn)
g6.add_edge(START, "draft").add_edge("draft", "approve")
g6.add_edge("approve", "publish").add_edge("publish", END)
static = g6.compile(checkpointer=InMemorySaver(), interrupt_before=["approve"])

cfg9 = {"configurable": {"thread_id": "static-1"}}
r = static.invoke({"draft": "", "attempts": 0, "outcome": ""}, cfg9)
print("暂停在 approve 前，next =", r.get("next"))

# 运行时也可以临时指定（每次调用可改），resume 用 None 继续
r2 = static.invoke(None, cfg9)
print("resume 后:", r2["outcome"])

# 运行时配置方式（可与编译时混用，逐次生效）
r3 = static.invoke({"draft": "", "attempts": 0, "outcome": ""},
                   interrupt_before=["draft"],
                   config={"configurable": {"thread_id": "static-2"}})
print("运行时断点停在 draft 前，next =", r3.get("next"))'''),
        md('''**动态 vs 静态断点**：

| 维度 | 静态断点 | 动态 `interrupt()` |
|---|---|---|
| 粒度 | 节点级（before/after） | 值级（代码任意位置） |
| 条件化 | 固定节点必停 | 可依据运行时逻辑 |
| 载荷 | 无（只暂停） | 可携带 JSON 载荷 |
| 官方定位 | **调试/测试** | HIL 首选 |

> 调试单步走图用静态断点；生产"人工介入业务逻辑"用 `interrupt()`。'''),
        md('''## 11. 常见问题（FAQ）

| 问题 | 原因 | 解决 |
|---|---|---|
| resume 报 `thread not found` | 两次调用 thread_id 不一致 | 用同一个 `thread_id` |
| 不加 checkpointer 报错 | interrupt 依赖持久化 | `compile(checkpointer=...)` |
| resume 值怎么取 | 成为 `interrupt()` 的返回值 | `decision = interrupt(...)` 处接收 |
| 想人工改状态再继续 | resume 只注入一个值 | `update_state(config, {...})` 后再 resume |
| 工具里 interrupt 不生效 | agent 未配 checkpointer | `create_agent(..., checkpointer=...)` |
| 多轮"重新输入"越跑越慢 | `while True + interrupt` 指数重放 | 条件边回环（第 8 节） |
| resume 前改了节点代码 | 重放与 checkpoint 失配 | 保持节点结构稳定 |
| 生产环境持久化 | `InMemorySaver` 重启丢 | `SqliteSaver` / `PostgresSaver`（见 11 讲） |'''),
    ]
    dump("09_Human_in_the_loop.ipynb", cells)


if __name__ == "__main__":
    build_09()

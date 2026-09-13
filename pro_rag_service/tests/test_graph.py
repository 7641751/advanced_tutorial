# -*- coding: utf-8 -*-
"""pro_rag_service 离线测试（零 API 额度）。

核心思想：FakeChatModel + 内存检索器替换真实依赖，
验证 Functional API 工作流的检索/生成/审批流程。

运行方式（在 pro_rag_service 目录下）：
    python -m pytest tests/ -v
"""
import sys
from pathlib import Path

# 让 tests/ 能导入项目模块
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.documents import Document
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeModel(FakeMessagesListChatModel):
    """固定返回"模拟回答"的假模型。"""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[
            ChatGeneration(message=AIMessage(content="（模拟回答）年假满 1 年 10 天。"))
        ])


class FakeRetriever:
    """固定返回两段文档的假检索器。"""

    def invoke(self, query: str, **kwargs):
        return [
            Document(page_content="年假制度：满 1 年 10 天，满 3 年 15 天。",
                     metadata={"title": "年假制度"}),
            Document(page_content="报销需在 30 天内提交。",
                     metadata={"title": "报销制度"}),
        ]


def test_app_compiles():
    """工作流能成功构建。"""
    from backend.app.agent.graph import build_app
    app = build_app(model=FakeModel(responses=[]), retriever=FakeRetriever())
    assert app is not None


def test_rag_without_approval():
    """非审批流：直接返回检索 + 生成的回答。"""
    from backend.app.agent.graph import build_app
    app = build_app(model=FakeModel(responses=[]), retriever=FakeRetriever())
    result = app.invoke("年假有几天？")
    assert "模拟回答" in result["answer"]


def test_rag_with_approval():
    """审批流：先暂停（interrupt），resume 批准后返回 approved=True。"""
    from backend.app.agent.graph import build_app
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    app = build_app(model=FakeModel(responses=[]), retriever=FakeRetriever(),
                    with_approval=True, checkpointer=InMemorySaver())

    cfg = {"configurable": {"thread_id": "t-approve"}}

    # 第一次：停在审批
    r1 = app.invoke("年假有几天？", config=cfg)
    assert "__interrupt__" in r1

    # 第二次：批准
    r2 = app.invoke(Command(resume={"ok": True}), config=cfg)
    assert r2["approved"] is True
    assert "模拟回答" in r2["answer"]


def test_rag_with_rejection():
    """审批流：驳回后 approved=False。"""
    from backend.app.agent.graph import build_app
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    app = build_app(model=FakeModel(responses=[]), retriever=FakeRetriever(),
                    with_approval=True, checkpointer=InMemorySaver())

    cfg = {"configurable": {"thread_id": "t-reject"}}
    app.invoke("年假有几天？", config=cfg)

    r2 = app.invoke(Command(resume={"ok": False}), config=cfg)
    assert r2["approved"] is False

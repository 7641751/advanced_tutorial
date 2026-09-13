# -*- coding: utf-8 -*-
"""核心工作流：Functional API 编排（教程 08）+ 人工审批（教程 09）。

流程（with_approval=True 时）：
    retrieve(检索) -> generate(生成) -> interrupt(人工审批)

设计要点：
- 用 @entrypoint / @task（Functional API）而非 StateGraph，串联教程 08
- interrupt 审批串联教程 09（必须配 checkpointer）
- build_app(...) 依赖注入，测试可传 Fake model / retriever，零 API 额度
"""
from typing import Optional

from config import settings


def get_model():
    from langchain_deepseek import ChatDeepSeek
    return ChatDeepSeek(model=settings.deepseek_model,
                        temperature=settings.temperature)


def get_retriever():
    from ingest import build_embeddings
    from langchain_chroma import Chroma

    embeddings = build_embeddings()
    vectorstore = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )
    return vectorstore.as_retriever(search_kwargs={"k": settings.top_k})


def build_app(model=None, retriever=None, checkpointer=None, with_approval=False):
    """构建 RAG 工作流（Functional API）。

    Args:
        model: 可注入 Fake 模型用于测试
        retriever: 可注入内存检索器用于测试
        checkpointer: 审批流必须（interrupt 依赖持久化）
        with_approval: True 时在生成后插入人工审批节点
    """
    from langgraph.func import entrypoint, task
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import interrupt

    model = model or get_model()
    retriever = retriever or get_retriever()
    checkpointer = checkpointer or InMemorySaver()

    @task
    def retrieve(question: str):
        """检索相关文档（task 可并行，这里单 task 演示）。"""
        docs = retriever.invoke(question)
        print(f"    [retrieve] 命中 {len(docs)} 段")
        return docs

    @task
    def generate(question: str, docs):
        """基于检索文档生成回答。"""
        if not docs:
            return "知识库中未找到相关信息，请换个问法或联系管理员。"
        context = "\n\n".join(
            f"【{d.metadata.get('title', '片段')}】{d.page_content}" for d in docs)
        r = model.invoke(
            f"你是公司制度问答助手，仅依据以下资料回答，资料没有的明确说明无法回答，"
            f"禁止编造。\n\n资料：\n{context}\n\n问题：{question}")
        return r.content

    if with_approval:
        @entrypoint(checkpointer=checkpointer)
        def app(question: str):
            docs = retrieve(question).result()
            answer = generate(question, docs).result()
            # 人工审批：暂停，等外部 resume 传入 {"ok": bool}
            verdict = interrupt({"question": question, "answer": answer,
                                 "ask": "批准还是驳回？"})
            return {"question": question, "answer": answer,
                    "approved": verdict.get("ok", False)}
    else:
        @entrypoint()
        def app(question: str):
            docs = retrieve(question).result()
            answer = generate(question, docs).result()
            return {"question": question, "answer": answer}

    return app


if __name__ == "__main__":
    # 打印功能说明（不执行真实检索，避免触发依赖）
    print("pro_rag_service.graph —— 用法见 README.md")

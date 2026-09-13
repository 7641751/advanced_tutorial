# -*- coding: utf-8 -*-
"""FastAPI 部署（教程 11）：把 RAG 链部署成 REST API（含 SSE 流式）。

langserve 已进入维护模式，本项目用 FastAPI 手写（更可控）。

运行方式（在 pro_rag_service 目录下，先 python ingest.py）：
    python server.py                # 默认 127.0.0.1:8000
    python server.py --port 9000
访问：
    http://127.0.0.1:8000/docs                          # Swagger UI
    http://127.0.0.1:8000/ask?q=年假有几天              # 非流式
    http://127.0.0.1:8000/ask/stream?q=年假有几天        # SSE 流式
"""
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 关键：HF 镜像 + .env 必须在导入业务模块前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
ROOT = Path(__file__).resolve().parents[2]
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from config import settings
from cache import enable_cache

# 启用 LLM 缓存（教程 07）
if settings.cache_enabled:
    print("[cache]", enable_cache(settings.cache_persistent, settings.cache_path))

import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from graph import get_model, get_retriever


# ---- 组装 RAG LCEL 链（检索 + 生成）----
retriever = get_retriever()
model = get_model()


def format_docs(docs):
    return "\n\n".join(
        f"【{d.metadata.get('title', '片段')}】{d.page_content}" for d in docs)


prompt = ChatPromptTemplate.from_template(
    "你是公司制度问答助手，仅依据以下资料回答，资料没有的明确说明无法回答，"
    "禁止编造。\n\n资料：\n{context}\n\n问题：{question}")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | model | StrOutputParser()
)

app = FastAPI(title="生产级 RAG 服务（带缓存 + 流式）")


@app.get("/ask")
def ask(q: str):
    """非流式：返回完整回答。"""
    return {"question": q, "answer": rag_chain.invoke(q)}


@app.get("/ask/stream")
def ask_stream(q: str):
    """SSE 流式：逐 token 推送。"""
    def gen():
        for token in rag_chain.stream(q):
            yield f"data: {json.dumps({'t': token}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="RAG 服务")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"服务启动: http://127.0.0.1:{args.port}/docs")
    uvicorn.run(app, host="127.0.0.1", port=args.port)

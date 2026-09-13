# -*- coding: utf-8 -*-
"""知识库入库脚本：加载 -> 切分 -> 嵌入 -> Chroma 持久化。

文档带 year/topic 元数据（教程 10 的 SelfQuery 依赖元数据过滤）。

运行方式（在 pro_rag_service 目录下）：
    python ingest.py            # 幂等：已有库则跳过
    python ingest.py --force    # 强制重建
"""
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 关键：HF 镜像必须在任何 langchain/chromadb 导入之前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from config import settings


def build_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.embed_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_documents():
    """加载知识库，把 year/topic 放进 metadata（供 SelfQuery 过滤）。"""
    raw = json.loads((settings.data_dir / "knowledge.json").read_text(encoding="utf-8"))
    from langchain_core.documents import Document

    docs = []
    for p in raw["policies"]:
        docs.append(Document(
            page_content=p["content"],
            metadata={
                "title": p["title"],
                "year": p.get("year"),
                "topic": p.get("topic", "通用"),
                "source": "knowledge.json",
            },
        ))
    return docs


def ingest(force: bool = False):
    if settings.chroma_dir.exists() and any(settings.chroma_dir.iterdir()) and not force:
        print(f"[skip] 向量库已存在: {settings.chroma_dir}（--force 重建）")
        return

    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs = load_documents()
    print(f"[1/3] 加载文档: {len(docs)} 篇")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[2/3] 切分 chunks: {len(chunks)} 段")

    embeddings = build_embeddings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_dir),
    )
    print(f"[3/3] 入库完成: {vectorstore._collection.count()} 条向量 -> {settings.chroma_dir}")


if __name__ == "__main__":
    ingest(force="--force" in sys.argv)

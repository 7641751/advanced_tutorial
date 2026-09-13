# -*- coding: utf-8 -*-
"""pro_rag_service 配置中心。

所有可调参数集中在此，通过环境变量覆盖（前缀 PRORAG_），
例如：PRORAG_TOP_K=6 python server.py
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """项目配置。默认值面向本地开发，生产用环境变量覆盖。"""

    model_config = SettingsConfigDict(
        env_prefix="PRORAG_",                       # 环境变量前缀
        env_file=str(PROJECT_DIR.parent.parent / ".env"),  # 项目根 .env
        extra="ignore",
    )

    # ---- LLM（DeepSeek）----
    deepseek_model: str = "deepseek-chat"
    temperature: float = 0.1                       # RAG 用低温度，减少编造

    # ---- 嵌入（本地 bge，离线可用）----
    embed_model: str = "BAAI/bge-small-zh-v1.5"
    hf_endpoint: str = "https://hf-mirror.com"     # 国内镜像

    # ---- 检索 ----
    top_k: int = 4                                 # 每次检索返回的文档数
    chunk_size: int = 300
    chunk_overlap: int = 50

    # ---- 缓存（教程 07）----
    cache_enabled: bool = True                     # 是否启用 LLM 缓存
    cache_persistent: bool = False                 # True 用 SQLite 持久化，False 用内存

    # ---- 路径 ----
    data_dir: Path = PROJECT_DIR / "data"
    chroma_dir: Path = PROJECT_DIR / "data" / "chroma_db"
    collection_name: str = "prorag_docs"
    cache_path: str = str(PROJECT_DIR / ".pro_rag_cache.db")


settings = Settings()

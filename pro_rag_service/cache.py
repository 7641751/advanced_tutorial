# -*- coding: utf-8 -*-
"""LLM 缓存封装（教程 07）。

- 内存缓存：进程内命中，重启失效
- SQLite 缓存：持久化到磁盘，跨重启复用

实测导入路径：
- InMemoryCache 在 langchain_core.caches
- SQLiteCache   在 langchain_community.cache（不在 core！）
"""
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache


def enable_cache(persistent: bool = False, path: str = ".pro_rag_cache.db"):
    """启用全局 LLM 缓存。

    Args:
        persistent: True 用 SQLite 持久化，False 用内存
        path: SQLite 缓存文件路径
    """
    if persistent:
        from langchain_community.cache import SQLiteCache
        set_llm_cache(SQLiteCache(database_path=path))
        return f"SQLiteCache(persistent) -> {path}"
    set_llm_cache(InMemoryCache())
    return "InMemoryCache(memory)"


def disable_cache():
    """关闭缓存（默认状态）。"""
    set_llm_cache(None)
    return "cache disabled"

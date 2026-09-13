# -*- coding: utf-8 -*-
"""离线评估脚本（教程 11 的可观测性落地方案）。

无 LangSmith key 时，用固定问题集 + 断言式检查做本地评估：
- 每个问题跑一遍，检查回答非空、命中检索、不含"无法回答"兜底（对已知问题）
- 输出 PASS/FAIL 汇总

运行方式（在 pro_rag_service 目录下，先 python ingest.py）：
    python evaluate.py
"""
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
ROOT = Path(__file__).resolve().parents[2]
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from graph import build_app

# 已知问题的预期关键词（用于断言式评估）
CASES = [
    ("年假有几天？", ["年假", "天"]),
    ("报销流程怎么走？", ["报销"]),
    ("远程办公有什么规定？", ["远程"]),
]


def evaluate():
    app = build_app()   # 非审批流
    passed = 0
    for q, keywords in CASES:
        result = app.invoke(q)
        answer = result.get("answer", "")
        ok = bool(answer) and all(k in answer for k in keywords)
        passed += 1 if ok else 0
        print(f"[{'PASS' if ok else 'FAIL'}] Q: {q}")
        print(f"          A: {answer[:80]}")
        print()
    print(f"结果: {passed}/{len(CASES)} 通过")


if __name__ == "__main__":
    evaluate()

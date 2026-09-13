# -*- coding: utf-8 -*-
"""验证并回填 09_Human_in_the_loop.ipynb：逐 code cell 顺序 exec（共享命名空间），
捕获 stdout/stderr 后写回 notebook 的 outputs 与 execution_count。

用法（在 advanced_tutorial 目录下）：
    python _run_nb09_check.py
"""
import io
import json
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE / "09_Human_in_the_loop.ipynb"


def main():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    ns = {"__name__": "__main__"}
    n_exec = 0
    failed = []

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        n_exec += 1
        out_buf, err_buf = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(out_buf), redirect_stderr(err_buf):
                exec(compile(src, f"<cell-{idx}>", "exec"), ns)
            outputs = []
            if out_buf.getvalue():
                outputs.append({"name": "stdout", "output_type": "stream",
                                "text": out_buf.getvalue().splitlines(keepends=True)})
            if err_buf.getvalue():
                outputs.append({"name": "stderr", "output_type": "stream",
                                "text": err_buf.getvalue().splitlines(keepends=True)})
            cell["outputs"] = outputs
            cell["execution_count"] = n_exec
            print(f"[OK] cell {idx}: {len(outputs)} output(s)")
        except Exception as e:
            tb = traceback.format_exc()
            cell["outputs"] = [{
                "ename": type(e).__name__, "evalue": str(e),
                "output_type": "error",
                "traceback": tb.splitlines(),
            }]
            cell["execution_count"] = n_exec
            failed.append((idx, type(e).__name__, str(e)))
            print(f"[FAIL] cell {idx}: {type(e).__name__}: {e}")

    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n== {n_exec} code cells, {len(failed)} failed ==")
    if failed:
        for idx, name, msg in failed:
            print(f"  cell {idx}: {name}: {msg}")
        sys.exit(1)
    print("ALL PASSED -> outputs written back to", NB.name)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

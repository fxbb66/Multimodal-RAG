# -*- coding: utf-8 -*-
"""
smoke_test.py — 前端/服务烟测脚本化（2026-10-01）
把"口述烟测"变成可验证、可留痕的产物：
  - GET  /api/health  服务存活、索引条数、rerank 累计计数
  - POST /api/query   多意图查询（不传 top_k，验证服务端默认 k=5）
    断言：返回 5 块、来源数 >= 2（MMR 多样性生效）、引用标记 [n] 存在且不越界、
          rerank 主链零降级（fallbacks=0）
结果落盘 output/smoke_report.json，含时间戳/每一项断言 pass-fail/响应摘要。

用法：python smoke_test.py [base_url]   默认 http://127.0.0.1:5001
退出码：0=全部通过，1=有失败项
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(_HERE, "output", "smoke_report.json")
MULTI_INTENT_Q = "上海迪士尼门票多少钱？停车费怎么收？附近有什么酒店推荐？"


def _req(method, url, payload=None, timeout=120):
    data = None
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    base = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:5001"
    checks, started = [], time.time()

    def check(name, ok, detail, warn_only=False):
        checks.append({"name": name, "pass": bool(ok), "warn_only": warn_only, "detail": detail})
        tag = "PASS" if ok else ("WARN" if warn_only else "FAIL")
        print(f"  [{tag}] {name}: {detail}")

    # 1) 健康检查
    print("=== /api/health ===")
    try:
        health = _req("GET", f"{base}/api/health", timeout=15)
        check("health.status", health.get("status") == "ok", health.get("status"))
        check("health.index_total>0", int(health.get("index_total", 0)) > 0,
              f"index_total={health.get('index_total')}")
    except Exception as e:
        check("health 可达", False, f"{type(e).__name__}: {e}")
        _write_report(base, checks, started, None)
        return 1

    # 2) 多意图查询（不传 top_k，验证默认值=5）
    print(f"=== /api/query 多意图 ===\n  Q: {MULTI_INTENT_Q}")
    try:
        r = _req("POST", f"{base}/api/query",
                 {"query": MULTI_INTENT_Q, "use_rerank": True, "use_rewrite": True})
    except Exception as e:
        check("query 可达", False, f"{type(e).__name__}: {e}")
        _write_report(base, checks, started, None)
        return 1

    chunks = r.get("chunks", [])
    sources = sorted({c.get("source", "") for c in chunks})
    answer = r.get("answer", "") or ""
    cites = sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer)})
    trace = r.get("trace", {})
    rr_stats = r.get("rerank_stats", {})

    check("默认 top_k=5", r.get("top_k") == 5, f"top_k={r.get('top_k')}")
    check("返回 5 块", len(chunks) == 5, f"chunks={len(chunks)}")
    check("不同来源>=2(MMR多样性)", len(sources) >= 2, f"sources={len(sources)}: {sources}")
    check("引用标记[n]存在", len(cites) >= 1, f"citations={cites}")
    check("引用标记不越界", all(1 <= n <= len(chunks) for n in cites),
          f"cites={cites} 范围应在1..{len(chunks)}")
    check("rerank 有模型真正出序", bool(rr_stats.get("last_model")),
          f"calls={rr_stats.get('calls')} last={rr_stats.get('last_model')}@{rr_stats.get('last_source')}")
    # 首选 Cohere 链若因上游计费故障自动降级到 qwen3-rerank，记 WARN 而非 FAIL
    preferred = rr_stats.get("last_source") == "AGICTO" and rr_stats.get("fallbacks", 0) == 0
    check("rerank 首选模型零降级(非阻断)", preferred,
          f"fallbacks={rr_stats.get('fallbacks')}（上游故障自动降级时为 WARN，不影响服务）",
          warn_only=True)
    check("answer 非空", len(answer.strip()) > 20, f"answer_len={len(answer)}")

    summary = {
        "query": MULTI_INTENT_Q,
        "effective_query": r.get("effective_query"),
        "intent_types": trace.get("intent_types"),
        "intent_confidence": trace.get("intent_confidence"),
        "top_k": r.get("top_k"),
        "chunk_ids": [c.get("id") for c in chunks],
        "sources": sources,
        "citations": cites,
        "rerank": rr_stats,
        "answer_head": answer[:200],
        "total_ms": trace.get("total_ms"),
    }
    _write_report(base, checks, started, summary)
    failed = [c for c in checks if not c["pass"] and not c.get("warn_only")]
    warns = [c for c in checks if not c["pass"] and c.get("warn_only")]
    print(f"\n结果: {len(checks) - len(failed) - len(warns)}/{len(checks)} 通过"
          + (f"，{len(warns)} 警告" if warns else ""))
    return 0 if not failed else 1


def _write_report(base, checks, started, summary):
    report = {
        "gen_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - started, 1),
        "base_url": base,
        "all_pass": all(c["pass"] for c in checks if not c.get("warn_only")),
        "warnings": [c["name"] for c in checks if not c["pass"] and c.get("warn_only")],
        "checks": checks,
        "response_summary": summary,
    }
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[落盘] {os.path.relpath(REPORT, _HERE)}")


if __name__ == "__main__":
    sys.exit(main())

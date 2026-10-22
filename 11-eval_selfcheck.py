# -*- coding: utf-8 -*-
"""
S3 证据充分性自检：183 金标 A/B 评估（2026-10-20）
================================================
对 183 对 QA 金标分别跑"自检关 / 自检开"两档检索（线上口径 k=5+rerank+MMR0.72+cap3），
量化自检闭环的净收益与代价：

  指标                    含义
  coverage_off/on         严格证据覆盖率（gold 在最终 top-5）
  trigger_rate            重检索触发率（retried>=1，即首轮自检判不充分）
  false_trigger_rate      误触发率：gold 本在首轮 top-5 却被判不充分（白跑一次重检索）
  retry_rescued           重检索后 gold 进入 top-5 的题数（净收益）
  retry_lost              首轮命中但重检索后丢失的题数（净损失，替换策略的风险面）
  still_insufficient_after_retry   重试后自检仍判不充分的题数（check 口径）
  not_rescued_after_retry          重试后 gold 仍未进 top-5 的题数（gold 口径）
  check_calls             自检 LLM 调用总数（成本）
  check_degraded          自检 LLM 失败次数（按"充分"兜底，不阻塞）

通过线（建议，待机时与准确率权衡后由人拍板默认开关）：
  coverage_on > coverage_off 且 retry_lost 占比 < 3% 且 false_trigger_rate < 25%

用法：
    python 11-eval_selfcheck.py [workers]
输出：output/eval_selfcheck_183.json
"""
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))

_spec = importlib.util.spec_from_file_location("ev", os.path.join(HERE, "4-evaluate.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

import faiss  # noqa: E402
import pickle  # noqa: E402

from retriever import HybridRetriever, ascii_faiss_path  # noqa: E402

OUT = os.path.join(HERE, "output", "eval_selfcheck_183.json")
K = 5


def _one_ab(qa_item, index, meta, bm25):
    """同一题跑两档：off 档记录首轮位置，on 档记录自检全链路。

    口径注意：search_with_evidence_check 在每次重检索后会再自检一次并用
    后值覆盖 info["sufficient"]，因此"重试后仍判不充分"= retried>=1 且
    最终 sufficient=False；"重检索触发"统一以 retried>=1 判定（首轮自检
    不充分才会重试），不能用最终 sufficient 反推首轮结论。
    """
    q = qa_item["query"]
    gold = qa_item["gold_idx"]
    vec = ev.get_embedding(ev.expand_query_terms(q))
    retr = HybridRetriever(index, meta, bm25=bm25, use_bm25=True, use_rerank=True)
    # off：纯 search（无自检）
    ranked_off = retr.search(vec, q, k=K, use_rerank=True, rerank_query=q,
                             max_per_source=ev.MAX_PER_SOURCE, use_mmr=ev.USE_MMR,
                             mmr_lambda=ev.MMR_LAMBDA, bm25_weight=ev.BM25_WEIGHT)
    off_pos = next((r for r, (i, _) in enumerate(ranked_off) if i == gold), None)
    # on：带自检 + 一次重检索
    ranked_on, info = retr.search_with_evidence_check(
        vec, q, k=K, use_rerank=True, rerank_query=q,
        max_per_source=ev.MAX_PER_SOURCE, use_mmr=ev.USE_MMR,
        mmr_lambda=ev.MMR_LAMBDA, bm25_weight=ev.BM25_WEIGHT,
        self_check=True, max_retry=1)
    on_pos = next((r for r, (i, _) in enumerate(ranked_on) if i == gold), None)
    return {"query": q, "gold_idx": gold,
            "off_pos": off_pos, "on_pos": on_pos,
            "triggered": info["triggered"], "sufficient": info["sufficient"],
            "retried": info["retried"], "degraded": info["degraded"],
            "check_ms": info["check_ms"]}


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    ev._load_emb_cache()
    qa = ev.load_or_build_qa(ev.get_index_md5())
    print(f"S3 自检 A/B：{len(qa)} 对，workers={workers}")

    index = faiss.read_index(ascii_faiss_path(ev.INDEX_FILE))
    meta = ev._load_metadata()
    with open(ev.BM25_FILE, "rb") as f:
        bm25 = pickle.load(f)

    outs = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one_ab, it, index, meta, bm25): it for it in qa}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                outs.append(fut.result())
            except Exception as e:
                it = futs[fut]
                outs.append({"query": it["query"], "gold_idx": it["gold_idx"],
                             "error": str(e)[:120]})
            if n % 30 == 0:
                print(f"  进度 {n}/{len(qa)}（{time.time()-t0:.0f}s）")

    ok = [o for o in outs if "error" not in o]
    n = max(len(ok), 1)
    cov_off = sum(1 for o in ok if o["off_pos"] is not None) / n
    cov_on = sum(1 for o in ok if o["on_pos"] is not None) / n
    # 重检索触发 = retried>=1（首轮自检不充分才会重试；sufficient 已被重试后的
    # 复检覆盖，不能用它反推首轮结论，见 _one_ab 注释）
    triggered = [o for o in ok if o["retried"] >= 1]
    # 误触发：首轮 gold 已在 top-5 却被判不充分（白跑一次重检索）
    false_trig = [o for o in triggered if o["off_pos"] is not None]
    # 净收益/损失：触发重检索的题中，off 未中而 on 中 / off 中而 on 未中
    rescued = [o for o in triggered if o["off_pos"] is None and o["on_pos"] is not None]
    lost = [o for o in triggered if o["off_pos"] is not None and o["on_pos"] is None]
    # 重试后自检仍判不充分（check 口径）/ 重试后 gold 仍未进 top-5（gold 口径）
    still_ins = [o for o in triggered if o["sufficient"] is False]
    not_rescued = [o for o in triggered if o["on_pos"] is None]
    degraded = [o for o in ok if o["degraded"]]

    summary = {
        "n": len(ok), "errors": len(outs) - len(ok),
        "coverage_off": round(cov_off, 4), "coverage_on": round(cov_on, 4),
        "coverage_delta": round(cov_on - cov_off, 4),
        "trigger_rate": round(len(triggered) / n, 4),
        "false_trigger_rate": round(len(false_trig) / max(len(triggered), 1), 4),
        "retry_rescued": len(rescued), "retry_lost": len(lost),
        "still_insufficient_after_retry": len(still_ins),
        "not_rescued_after_retry": len(not_rescued),
        "check_degraded": len(degraded),
        "check_ms_median": int(np.median([o["check_ms"] for o in ok if o["triggered"]]))
        if any(o["triggered"] for o in ok) else 0,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    print("\n=== S3 证据自检 A/B（183 对，k=5 线上口径） ===")
    for k2, v in summary.items():
        print(f"  {k2:32s} {v}")
    print("\n  净收益题（off 丢 on 中）:")
    for o in rescued[:10]:
        print(f"    + {o['query'][:36]!r}")
    print("  净损失题（off 中 on 丢）:")
    for o in lost[:10]:
        print(f"    - {o['query'][:36]!r}")
    print(f"  误触发样例（首轮已中被判不充分）:")
    for o in false_trig[:5]:
        print(f"    ~ {o['query'][:36]!r} (off_pos={o['off_pos']})")

    out = {"__meta__": {"ts": int(time.time()), "k": K, "n": len(qa),
                        "index_md5": ev.get_index_md5(), "workers": workers,
                        "口径": "k=5+rerank+MMR0.72+cap3；on 档=自检+词典扩展重检索1次(替换)"},
           "summary": summary, "details": outs}
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUT)
    print(f"\n已落盘: {os.path.relpath(OUT, HERE)}")


if __name__ == "__main__":
    main()

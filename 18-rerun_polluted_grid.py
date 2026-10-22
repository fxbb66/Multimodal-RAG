# -*- coding: utf-8 -*-
"""S7+S8 合跑后补跑：干净重跑被 503 风暴波及的 7 个检索网格档（2026-10-21）。

背景：S7+S8 全量合跑（output/s7s8_run.log → eval_results_183_s7s8.json）运行期，
AGICTO rerank 端点出现 503 风暴（百炼欠费信封先开熔断，随后 AGICTO 累计 31 次
瞬时 503 降级）。逐档归因（按结果行切分日志）：

  - 混合+Rerank（0.956，头条档）：风暴前跑完，calls=183 fallbacks=0 —— 干净，不重跑
  - 检索(k=3,后处理关) / (k=3,后处理开) / (k=5,后处理关) / (k=5,后处理开)：
    各有 3-5 次降级（受影响题的 rerank 兜底为未重排的融合序，读数偏保守/下偏）
  - 检索(k=5,线上口径)：7 次降级
  - 扫描(MMR0.72) / 扫描(MMR0.55)：各有 6-7 次降级
  - 答案档/图片档/意图档：风暴后运行，零降级 —— 干净，不重跑

本脚本只重跑上述 7 档（纯检索+rerank，改写走 v4 磁盘缓存、query 向量走 emb 缓存，
零 embedding/LLM 调用），把干净结果原位写回 eval_results_183_s7s8.json 与
eval_results.json 对应行；rerank生效 行同步重写为本次干净统计并注明风暴史。
安全闸门与 15 号脚本一致：每档最多 TIER_ATTEMPTS 次整档重跑，仅 fallbacks=0
（首选零降级）才写回；写回前备份 pre_gridrerun 版本留档。

用法：
    .venv\\Scripts\\python 18-rerun_polluted_grid.py
"""
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))

_spec = importlib.util.spec_from_file_location("ev", os.path.join(HERE, "4-evaluate.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

import faiss  # noqa: E402
import pickle  # noqa: E402

from retriever import HybridRetriever, RERANK_STATS, ascii_faiss_path  # noqa: E402

TARGETS = [os.path.join(HERE, "output", "eval_results_183_s7s8.json"),
           os.path.join(HERE, "output", "eval_results.json")]
TIER_ATTEMPTS = 4  # 每档整档重跑次数上限（fallbacks=0 即通过，无需用完）
# 降并发跑补评：8 路并发 + 重试叠加会放大上游 503（15 号脚本实证），4 路更稳
TIER_WORKERS = 4
# 与 4-evaluate.py main() 逐项同参的 7 个受污染档（标签构造与 main() 一致，
# SWEEP_LAMBDA/SWEEP_MAX_PER_SOURCE 在 4-evaluate.py 里是 main() 局部变量，
# 此处按同规则复刻，保证标号与原名一致可原位替换）
SWEEP_LAMBDA = float(os.environ.get("SWEEP_LAMBDA", "0.55"))
SWEEP_MAX_PER_SOURCE = int(os.environ.get("SWEEP_MAX_PER_SOURCE", "3"))
TIERS = [
    dict(k=3, use_rerank=True, label="检索(k=3,后处理关)", workers=TIER_WORKERS),
    dict(k=3, use_rerank=True, label=f"检索(k=3,后处理开:MMR+限额{ev.MAX_PER_SOURCE})",
         max_per_source=ev.MAX_PER_SOURCE, use_mmr=ev.USE_MMR,
         mmr_lambda=ev.MMR_LAMBDA, workers=TIER_WORKERS),
    dict(k=5, use_rerank=True, label="检索(k=5,后处理关)", workers=TIER_WORKERS),
    dict(k=5, use_rerank=True, label=f"检索(k=5,后处理开:MMR+限额{ev.MAX_PER_SOURCE})",
         max_per_source=ev.MAX_PER_SOURCE, use_mmr=ev.USE_MMR,
         mmr_lambda=ev.MMR_LAMBDA, workers=TIER_WORKERS),
    dict(k=5, use_rerank=True,
         label=f"检索(k=5,线上口径:后处理开+改写query,限额{ev.MAX_PER_SOURCE})",
         max_per_source=ev.MAX_PER_SOURCE, use_mmr=ev.USE_MMR,
         mmr_lambda=ev.MMR_LAMBDA, use_rewrite=True, workers=TIER_WORKERS),
    dict(k=5, use_rerank=True,
         label=f"扫描(k=5+限额{SWEEP_MAX_PER_SOURCE}+MMR{ev.MMR_LAMBDA})",
         max_per_source=SWEEP_MAX_PER_SOURCE, use_mmr=True,
         mmr_lambda=ev.MMR_LAMBDA, workers=TIER_WORKERS),
    dict(k=5, use_rerank=True,
         label=f"扫描(k=5+限额{ev.MAX_PER_SOURCE}+MMR{SWEEP_LAMBDA})",
         max_per_source=ev.MAX_PER_SOURCE, use_mmr=True,
         mmr_lambda=SWEEP_LAMBDA, workers=TIER_WORKERS),
]


def main():
    ev._load_emb_cache()
    qa = ev.load_or_build_qa(ev.get_index_md5())
    index = faiss.read_index(ascii_faiss_path(ev.INDEX_FILE))
    meta = ev._load_metadata()
    with open(ev.BM25_FILE, "rb") as f:
        bm25 = pickle.load(f)

    r_rr = HybridRetriever(index, meta, bm25=bm25, use_bm25=True,
                           use_rerank=True, rerank_topn=ev.RERANK_TOPN)
    vec_cache = {}

    new_rows = {}
    old_rows = {}
    for t in TIERS:
        row = None
        for att in range(1, TIER_ATTEMPTS + 1):
            RERANK_STATS["calls"] = RERANK_STATS["fallbacks"] = 0
            RERANK_STATS["circuit_skips"] = 0
            t0 = time.time()
            row = ev.evaluate(r_rr, qa, vec_cache, **t)
            dt = time.time() - t0
            effective = bool(RERANK_STATS["calls"] > 0 and RERANK_STATS["last_model"])
            preferred_ok = effective and RERANK_STATS["fallbacks"] == 0
            print(f"  [{t['label']}] 第{att}次: recall@k={row.get('recall@k'):.4f} "
                  f"lenient={row.get('recall_lenient'):.4f} mrr={row.get('mrr'):.4f} ({dt:.0f}s)")
            print(f"    rerank: calls={RERANK_STATS['calls']} fallbacks={RERANK_STATS['fallbacks']} "
                  f"last={RERANK_STATS['last_model']}@{RERANK_STATS['last_source']} "
                  f"preferred_ok={preferred_ok}")
            if preferred_ok:
                break
        if not preferred_ok:
            raise SystemExit(
                f"[中止] {t['label']} {TIER_ATTEMPTS} 次跑批均有 rerank 降级"
                f"（末次 fallbacks={RERANK_STATS['fallbacks']}）——结果仍可能被污染，不写回。"
                "请等 API 更稳定后重跑本脚本。")
        row["rerank_calls"] = RERANK_STATS["calls"]
        row["rerank_fallbacks"] = RERANK_STATS["fallbacks"]
        new_rows[t["label"]] = row

    for path in TARGETS:
        if not os.path.exists(path):
            print(f"  跳过（不存在）: {os.path.basename(path)}")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # 写回前备份留档（含被替换的污染读数）
        bak = path.replace(".json", ".pre_gridrerun.json")
        with open(bak, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        replaced = 0
        for i, r in enumerate(data["results"]):
            lbl = r.get("label")
            if lbl in new_rows:
                old_rows[lbl] = {k: r.get(k) for k in ("recall@k", "recall_lenient", "mrr")}
                data["results"][i] = new_rows[lbl]
                replaced += 1
            elif lbl == "rerank生效":
                data["results"][i] = {
                    "label": "rerank生效",
                    "calls": new_rows[TIERS[0]["label"]].get("rerank_calls", 0),
                    "fallbacks": new_rows[TIERS[0]["label"]].get("rerank_fallbacks", 0),
                    "last_model": RERANK_STATS["last_model"],
                    "last_source": RERANK_STATS["last_source"],
                    "ok": True, "preferred_ok": True,
                    "note": ("主跑混合+Rerank档风暴前干净(calls=183,fallbacks=0)；"
                             "网格7档运行期遭31次AGICTO 503降级，已由 "
                             "18-rerun_polluted_grid.py 干净重跑写回"),
                }
        cfg = data.setdefault("config", {})
        cfg["polluted_grid_rerun"] = {
            "ts": int(time.time()),
            "reason": "合跑运行期 AGICTO rerank 503 风暴波及 7 个检索网格档，干净重跑",
            "tiers": list(new_rows.keys()),
            "old_rows": old_rows,
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        print(f"  已写回 {os.path.basename(path)}（替换 {replaced} 档 + rerank生效行，"
              f"备份 {os.path.basename(bak)}）")

    print("\n=== 污染读数 vs 干净读数（同轮同索引，降级=部分题未重排，应≤干净值） ===")
    for lbl in new_rows:
        o, n = old_rows.get(lbl, {}), new_rows[lbl]
        print(f"  {lbl}")
        print(f"    前(污染): recall@k={o.get('recall@k')} mrr={o.get('mrr')}")
        print(f"    后(干净): recall@k={n.get('recall@k'):.4f} mrr={n.get('mrr'):.4f} "
              f"(Δrecall={n.get('recall@k', 0) - (o.get('recall@k') or 0):+.4f})")
    print("\n补跑完成。建议接着跑 17-compare_s7s8.py 复核对比。")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""API 恢复后补跑：重评被 503 风暴波及的两个检索档位（2026-10-21）。

背景：最终版全量重跑（output/eval_results_183_final.json）运行期，AGICTO rerank
端点两轮 503 风暴（累计 436 次降级），降级链兜底生效但改变了部分题的重排结果，
波及且仅波及运行于风暴期的两个档位：
  - 检索(k=5,后处理开:MMR+限额3)   0.929 → 0.896
  - 扫描(k=5+限额3+MMR0.55)        0.940 → 0.885
其余档位（含 online 答案档/图片档/意图档）均在末次 503 之后运行，读数可作准。

本脚本只重跑这两个档位（纯检索+rerank，无 LLM 生成，约 3 分钟），并把结果
原位写回 eval_results_183_final.json 与 eval_results.json 的对应行；
 rerank 生效核验行（calls/fallbacks/preferred_ok）同步重写为本次干净运行的统计。

用法：
    .venv\\Scripts\\python 15-rerun_polluted_tiers.py
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

FINAL = os.path.join(HERE, "output", "eval_results_183_final.json")
LIVE = os.path.join(HERE, "output", "eval_results.json")
TIER_ATTEMPTS = 4  # 每档整档重跑次数上限（fallbacks=0 即通过，无需用完）
# 降并发跑补评：8 路并发 + 重试叠加会放大上游 503（疑似限流），4 路更稳
TIER_WORKERS = 4

# 与 4-evaluate.py main() 逐项同参的两个受污染档位
# 注意：SWEEP_LAMBDA/SWEEP_MAX_PER_SOURCE 在 4-evaluate.py 里是 main() 局部变量
# （os.environ 读取、默认 0.55/3），此处按同规则复刻，确保标号与原名一致
SWEEP_LAMBDA = float(os.environ.get("SWEEP_LAMBDA", "0.55"))
TIERS = [
    dict(k=5, use_rerank=True, label=f"检索(k=5,后处理开:MMR+限额{ev.MAX_PER_SOURCE})",
         max_per_source=ev.MAX_PER_SOURCE, use_mmr=ev.USE_MMR, mmr_lambda=ev.MMR_LAMBDA,
         workers=TIER_WORKERS),
    dict(k=5, use_rerank=True,
         label=f"扫描(k=5+限额{ev.MAX_PER_SOURCE}+MMR{SWEEP_LAMBDA})",
         max_per_source=ev.MAX_PER_SOURCE, use_mmr=True, mmr_lambda=SWEEP_LAMBDA,
         workers=TIER_WORKERS),
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
    for t in TIERS:
        # 每档最多重试 TIER_ATTEMPTS 次：AGICTO 瞬时 503 呈burst形态，单次跑批
        # 可能有少量调用扛过 4 次重试仍失败；整档重跑一次通常是干净的概率很高。
        # 只有 fallbacks=0（首选模型零降级）的结果才允许写回。
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

    for path in (FINAL, LIVE):
        if not os.path.exists(path):
            print(f"  跳过（不存在）: {os.path.basename(path)}")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        replaced = 0
        for i, r in enumerate(data["results"]):
            lbl = r.get("label")
            if lbl in new_rows:
                data["results"][i] = new_rows[lbl]
                replaced += 1
            elif lbl == "rerank生效":
                # 同步干净运行的 rerank 统计（取任一次跑批的计数，两档共用同一检索器配置）
                data["results"][i] = {
                    "label": "rerank生效",
                    "calls": new_rows[TIERS[0]["label"]].get("rerank_calls", 0),
                    "fallbacks": new_rows[TIERS[0]["label"]].get("rerank_fallbacks", 0),
                    "last_model": RERANK_STATS["last_model"],
                    "last_source": RERANK_STATS["last_source"],
                    "ok": True, "preferred_ok": True,
                    "note": "2026-10-21 API 恢复后由 15-rerun_polluted_tiers.py 干净重跑",
                }
        # 配置快照补记本次修补
        cfg = data.setdefault("config", {})
        cfg["polluted_tiers_rerun"] = {
            "ts": int(time.time()),
            "reason": "503 风暴波及档位的 API 恢复后干净重跑",
            "tiers": list(new_rows.keys()),
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        print(f"  已写回 {os.path.basename(path)}（替换 {replaced} 档 + rerank生效行）")

    print("\n补跑完成。建议接着跑 compare_final.py 复核三方对比。")


if __name__ == "__main__":
    main()

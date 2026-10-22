# -*- coding: utf-8 -*-
"""
S6 图片金标检索评估（2026-10-20）
================================
对 12-build_image_gold.py 生成的图片 QA 金标，走线上同款检索口径
（k=5 + rerank + MMR0.72 + cap3），回答两个问题：

  ① 图片块能否被文字 query 检索到（image_recall@5）——caption 进索引后的基线；
  ② 检索不到时丢在哪一环（分阶段追踪：向量路/BM25路/融合/rerank/MMR）。

图片块进 top-k 的意义：用户问"邮轮内舱多少钱"时，价格表图片应作为证据出现，
由 build_context 并入上下文（[n] 编号），生成侧才可能引用"详见图表"。

口径与 4-evaluate.py 主评估逐项同源；可被 4-evaluate.py 以模块方式调用
（evaluate_image_gold），结果写入 eval_results 的 results[]。

用法：
    python 13-eval_image_gold.py [workers]
输出：output/eval_image_gold_result.json
"""
import importlib.util
import json
import os
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))

_spec = importlib.util.spec_from_file_location("ev", os.path.join(HERE, "4-evaluate.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

import faiss  # noqa: E402
import pickle  # noqa: E402

from retriever import (HybridRetriever, rrf_fuse, expand_query_terms,  # noqa: E402
                       qwen_rerank, ascii_faiss_path)

GOLD_FILE = os.path.join(HERE, "output", "eval_image_gold.json")
OUT = os.path.join(HERE, "output", "eval_image_gold_result.json")
K = 5


def _one(qa_item, index, meta, bm25):
    q = qa_item["query"]
    gold = qa_item["gold_chunk_hash"]
    h2i = {m.get("chunk_hash"): i for i, m in enumerate(meta)}
    gi = h2i.get(gold)
    if gi is None:
        return {"query": q, "category": "gold_missing"}
    # 双向量：图片在索引里占主行(gi) + 文本影子行(shadow_of=gi)；
    # 任一行的"金标图片"都映射回 gi，诊断才如实反映图片是否被检索到
    idx2gold = {i: (m["shadow_of"] if m.get("is_shadow") else i)
                for i, m in enumerate(meta)}
    vec = ev.get_embedding(expand_query_terms(q))
    retr = HybridRetriever(index, meta, bm25=bm25, use_bm25=True, use_rerank=True)
    ranked = retr.search(vec, q, k=K, use_rerank=True, rerank_query=q,
                         max_per_source=ev.MAX_PER_SOURCE, use_mmr=ev.USE_MMR,
                         mmr_lambda=ev.MMR_LAMBDA, bm25_weight=ev.BM25_WEIGHT)
    pos = next((r for r, (i, _) in enumerate(ranked) if i == gi), None)

    # 分阶段追踪（诊断用，口径同 10-diagnose_evidence；S7/S8 后与线上逐项同源：
    # 向量路先 group 归并，融合走 _fuse_with_path_union（含路径级并集保底））
    D, I = index.search(np.array([vec], dtype="float32"), 20)
    vec_ranks = retr._dedup_vec_ranks(
        [(int(i2), float(s)) for i2, s in zip(I[0], D[0]) if i2 >= 0])
    vec_pos = next((r for r, (i2, _) in enumerate(vec_ranks)
                    if idx2gold.get(i2) == gi), None)
    bm = bm25.search(expand_query_terms(q), topk=20)
    bm_pos = next((r for r, (i2, _) in enumerate(bm) if idx2gold.get(int(i2)) == gi), None)
    fused = retr._fuse_with_path_union([vec_ranks, bm], [0.5, 0.5], 12, True)
    fuse_pos = next((r for r, (i2, _) in enumerate(fused) if idx2gold.get(i2) == gi), None)
    return {"query": q, "gold_chunk_hash": gold, "final_pos": pos,
            "vec_pos": vec_pos, "bm_pos": bm_pos, "fuse_pos": fuse_pos}


def evaluate_image_gold(workers=4):
    """供 4-evaluate.py 调用的入口：返回 results[] 单行（无金标时返回 None）。"""
    if not os.path.exists(GOLD_FILE):
        return None
    gold = json.load(open(GOLD_FILE, encoding="utf-8"))
    items = gold.get("items", [])
    if not items:
        return None
    index = faiss.read_index(ascii_faiss_path(ev.INDEX_FILE))
    meta = ev._load_metadata()
    with open(ev.BM25_FILE, "rb") as f:
        bm25 = pickle.load(f)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    outs = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, it, index, meta, bm25): it for it in items}
        for fut in as_completed(futs):
            try:
                outs.append(fut.result())
            except Exception as e:
                outs.append({"query": futs[fut]["query"], "category": "error",
                             "error": str(e)[:100]})
    ok = [o for o in outs if o.get("category") != "gold_missing"]
    n = max(len(ok), 1)
    hits = sum(1 for o in ok if o.get("final_pos") is not None)
    cnt = Counter()
    for o in ok:
        if o.get("final_pos") is not None:
            cnt["hit"] += 1
        elif o.get("fuse_pos") is not None:
            cnt["lost_at_rerank_or_mmr"] += 1
        elif o.get("vec_pos") is not None or o.get("bm_pos") is not None:
            cnt["lost_at_fusion"] += 1
        else:
            cnt["lost_both_paths"] += 1
    return {"label": f"图片金标检索(k={K},后处理开:MMR+限额{ev.MAX_PER_SOURCE})",
            "n": len(ok), "image_recall_at_k": round(hits / n, 4),
            "lost_breakdown": dict(cnt),
            "index_md5": ev.get_index_md5(),
            "details": outs}


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    ev._load_emb_cache()
    row = evaluate_image_gold(workers)
    if row is None:
        raise SystemExit("图片金标缺失，请先跑 12-build_image_gold.py")
    print("\n=== 图片金标检索评估（线上口径） ===")
    print(f"  n={row['n']}  image_recall@{K}={row['image_recall_at_k']}")
    print(f"  丢失分布: {row['lost_breakdown']}")
    for o in row["details"]:
        if o.get("final_pos") is None:
            print(f"    丢: vec={o.get('vec_pos')} bm={o.get('bm_pos')} "
                  f"fuse={o.get('fuse_pos')} | {o['query'][:30]!r}")
    out = {"__meta__": {"ts": int(time.time()), "k": K,
                        "index_md5": row["index_md5"],
                        "口径": "k=5+rerank+MMR0.72+cap3，与主评估同源"},
           "result": row}
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUT)
    print(f"\n已落盘: {os.path.relpath(OUT, HERE)}")


if __name__ == "__main__":
    main()

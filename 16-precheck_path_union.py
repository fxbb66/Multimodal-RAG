# -*- coding: utf-8 -*-
"""
S7 预分析：融合前路径级 top-k 并集的参数选择（本地零 API）
=========================================================
用 eval_emb_cache.npz 已缓存的 183 条查询向量 + 本地 FAISS/BM25，复刻线上
RRF(k=60, w=0.5/0.5, top12) 融合，回答三个问题：

  ① 不同 PATH_UNION_TOPK（6/8/12/16）下，并集比 RRF top-12 多引入多少候选
     （rerank 池增长 → 成本）；
  ② S2 诊断的 7 例 lost_at_fusion 金标，在哪些 TOPK 下能全部进入候选池；
  ③ 全量 183 题的金标进池率变化（gold in union）与池规模分布。

只用缓存向量，不发任何 API 请求；查询向量缺失的条目跳过并计数。
"""
import importlib.util
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))

_spec = importlib.util.spec_from_file_location("ev", os.path.join(HERE, "4-evaluate.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

import faiss  # noqa: E402
import pickle  # noqa: E402

from retriever import rrf_fuse, expand_query_terms, ascii_faiss_path  # noqa: E402

VEC_TOPK = 20
FUSE_TOPK = 12
TOPK_CANDIDATES = [8, 9, 10, 12]


def main():
    ev._load_emb_cache()
    cache = ev._emb_disk
    print(f"embedding 缓存 {len(cache)} 条")

    qa_all = json.load(open(os.path.join(HERE, "output", "eval_qa_v2.json"), encoding="utf-8"))
    items = qa_all["items"] if isinstance(qa_all, dict) else qa_all
    meta = ev._load_metadata()
    h2i = {m.get("chunk_hash"): i for i, m in enumerate(meta)}
    for it in items:
        it["gold_idx"] = h2i.get(it.get("chunk_hash"))
    items = [it for it in items if it["gold_idx"] is not None]
    print(f"QA 金标 {len(items)} 对（按 chunk_hash 对齐当前索引）")

    index = faiss.read_index(ascii_faiss_path(ev.INDEX_FILE))
    with open(ev.BM25_FILE, "rb") as f:
        bm25 = pickle.load(f)

    # S2 诊断的 7 例 lost_at_fusion 查询（gold 在单路 top-20 但融不进 top-12）
    diag = json.load(open(os.path.join(HERE, "output", "diag_evidence_183.json"), encoding="utf-8"))
    lost_fusion_queries = {o["query"] for o in diag["details"] if o["category"] == "lost_at_fusion"}

    stats = {t: {"extra_sum": 0, "n": 0, "pool_hist": Counter(), "gold_in_union": 0,
                 "gold_in_fused12": 0} for t in TOPK_CANDIDATES}
    missing_vec = 0
    lf_detail = {t: [] for t in TOPK_CANDIDATES}

    for it in items:
        q = it["query"]
        gold = it["gold_idx"]
        exp = expand_query_terms(q)
        vec = cache.get(exp)
        if vec is None:
            missing_vec += 1
            continue
        D, I = index.search(np.array([vec], dtype="float32"), VEC_TOPK)
        vec_ranks = [(int(i), float(s)) for i, s in zip(I[0], D[0]) if i >= 0]
        bm_ranks = bm25.search(exp, topk=VEC_TOPK)
        fused12 = rrf_fuse(vec_ranks, bm_ranks, topk=FUSE_TOPK, weights=[0.5, 0.5])
        fused12_idx = {i for i, _ in fused12}

        for t in TOPK_CANDIDATES:
            union_idx = {i for i, _ in vec_ranks[:t]} | {i for i, _ in bm_ranks[:t]}
            extras = union_idx - fused12_idx
            st = stats[t]
            st["n"] += 1
            st["extra_sum"] += len(extras)
            st["pool_hist"][FUSE_TOPK + len(extras)] += 1
            if gold in union_idx:
                st["gold_in_union"] += 1
            if gold in fused12_idx:
                st["gold_in_fused12"] += 1
            if q in lost_fusion_queries:
                lf_detail[t].append((q, gold in union_idx))

    print(f"\n查询向量缺失 {missing_vec} 条（跳过）")
    print("\n=== 并集参数对比（n=183，RRF top12 基线）===")
    print(f"{'TOPK':>5} {'平均额外候选':>12} {'池规模中位':>10} {'池规模P95':>10} "
          f"{'gold进池率':>10} {'(基线)':>8}")
    for t in TOPK_CANDIDATES:
        st = stats[t]
        n = max(st["n"], 1)
        sizes = sorted(st["pool_hist"].elements())
        med = sizes[len(sizes) // 2] if sizes else 0
        p95 = sizes[int(len(sizes) * 0.95)] if sizes else 0
        print(f"{t:>5} {st['extra_sum']/n:>12.2f} {med:>10} {p95:>10} "
              f"{st['gold_in_union']/n:>10.3f} {st['gold_in_fused12']/n:>8.3f}")

    print("\n=== 7 例 lost_at_fusion 金标进池情况（S2 诊断集）===")
    for t in TOPK_CANDIDATES:
        ok = sum(1 for _, hit in lf_detail[t] if hit)
        print(f"  TOPK={t:>2}: {ok}/7 进池")
        for q, hit in lf_detail[t]:
            if not hit:
                print(f"      未进: {q!r}")

    # 候选池增长对 rerank 负载的影响：总候选文档数（183 题 × 池规模）
    print("\n=== rerank 候选文档总量（183 题，越大越贵）===")
    for t in TOPK_CANDIDATES:
        st = stats[t]
        total = sum(size * cnt for size, cnt in st["pool_hist"].items())
        print(f"  TOPK={t:>2}: {total} docs（基线 {FUSE_TOPK * st['n']}，"
              f" +{(total - FUSE_TOPK * st['n']) / max(FUSE_TOPK * st['n'], 1):.1%}）")


if __name__ == "__main__":
    main()

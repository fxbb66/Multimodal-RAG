# -*- coding: utf-8 -*-
"""
S2 证据覆盖瓶颈诊断（2026-10-20）
================================
对 183 对 QA 金标逐题复刻线上检索管线各阶段，追踪金标块（gold_idx）在哪一环丢失，
把"证据未覆盖"从一坨数字拆成可行动的三类根因：

  ① lost_at_rerank   金标在 RRF 候选池（rerank 前 top-12）里，但被 rerank 排歪
                      （掉出 top-k）或被 MMR 挤出 —— 重排阶段问题
  ② lost_at_fusion   金标在向量路或 BM25 路任一路的 top-20 里，但 RRF 融合后
                      没进候选池 —— 融合权重/位次问题
  ③ lost_both        两路 top-20 都没有金标，附 BM25 最高分、向量最高余弦、
                      查询-金标块字符二元组 jaccard、块字数，供判读：
                      词面不匹配 / 语义不匹配 / chunk 粒度不当

口径与线上完全同源：k=5, rerank 开, MMR λ=0.72, 同源限额 cap=3, BM25_WEIGHT=0.5。
不写任何共享缓存（embedding 只读内存副本），workers 默认 3，避免与主评估抢限流。

用法：
    python 10-diagnose_evidence.py [workers]
输出：output/diag_evidence_183.json + 控制台三类汇总
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

OUT = os.path.join(HERE, "output", "diag_evidence_183.json")
K = 5
VEC_TOPK = 20          # 线上 allowed=None 时 vec_k = max(k*4, 20)
FUSE_TOPK = 12         # rerank 开时 RRF 取 rerank_topn


def _bigrams(s):
    s = "".join(ch for ch in s if ch.isalnum())
    return set(s[i:i + 2] for i in range(len(s) - 1)) or {s} if s else set()


def _one(qa_item, index, meta, bm25):
    q = qa_item["query"]
    gold = qa_item["gold_idx"]
    vec = ev.get_embedding(expand_query_terms(q))
    # S8 双向量：槽位→主条目映射（影子槽命中即主条目图片被检索到），
    # 与 13-eval_image_gold.py 同口径；向量路先按 group 归并再进融合，与线上一致
    idx2gold = {i: (m["shadow_of"] if m.get("is_shadow") else i)
                for i, m in enumerate(meta)}

    # 向量路（归并后与线上 _vec_search 输出同构）
    D, I = index.search(np.array([vec], dtype="float32"), VEC_TOPK)
    vec_ranks = [(int(i), float(sc)) for sc, i in zip(D[0], I[0]) if i >= 0]
    vec_ranks = _RETR._dedup_vec_ranks(vec_ranks)
    vec_pos = next((r for r, (i, _) in enumerate(vec_ranks)
                    if idx2gold.get(i) == gold), None)

    # BM25 路
    exp = expand_query_terms(q)
    bm_ranks = bm25.search(exp, topk=VEC_TOPK)
    bm_pos = next((r for r, (i, _) in enumerate(bm_ranks)
                    if idx2gold.get(i) == gold), None)
    bm_top_score = bm_ranks[0][1] if bm_ranks else 0.0
    gold_bm_score = next((s for i, s in bm_ranks if idx2gold.get(i) == gold), 0.0)

    # RRF 融合（线上同权重）
    fused = rrf_fuse(vec_ranks, bm_ranks, topk=FUSE_TOPK, weights=[0.5, 0.5])
    fused = [(i, s) for i, s in fused if _valid(meta[i])]
    fuse_pos = next((r for r, (i, _) in enumerate(fused)
                    if idx2gold.get(i) == gold), None)

    # rerank（候选池整条喂，top_n=K，与线上一致）
    rerank_pos = None
    if len(fused) > 1:
        cands = [meta[i].get("content", "") for i, _ in fused]
        try:
            rr = qwen_rerank(q, cands, topk=K)
            order = [fused[oi][0] for oi, _ in rr]
            for i, _ in fused:
                if i not in set(order):
                    order.append(i)
        except Exception:
            order = [i for i, _ in fused]
    else:
        order = [i for i, _ in fused]
    rerank_pos = next((r for r, i in enumerate(order)
                      if idx2gold.get(i) == gold), None)

    # MMR + 同源限额（线上参数）
    retr = _RETR
    final = retr._mmr_select(vec, order, K, lam=0.72)
    seen_src, capped = set(), []
    for i in final:
        src = meta[i].get("source", "")
        if src in seen_src:
            continue
        seen_src.add(src)
        capped.append(i)
        if len(capped) >= K:
            break
    if len(capped) < K:  # 限额未排满时按序补齐（线上同逻辑）
        for i in final:
            if i not in capped:
                capped.append(i)
                if len(capped) >= K:
                    break
    final_pos = next((r for r, i in enumerate(capped)
                     if idx2gold.get(i) == gold), None)

    gold_content = meta[gold].get("content", "")
    gb_q, gb_g = _bigrams(q), _bigrams(gold_content)
    jaccard = len(gb_q & gb_g) / max(1, len(gb_q | gb_g))
    cos_top = vec_ranks[0][1] if vec_ranks else 0.0

    if final_pos is not None:
        cat = "hit"
    elif rerank_pos is not None:
        cat = "lost_at_rerank"       # 候选池里有，rerank/MMR/限额环节丢
    elif fuse_pos is not None:
        cat = "lost_at_rerank"       # 融合后有，rerank 丢
    elif vec_pos is not None or bm_pos is not None:
        cat = "lost_at_fusion"
    else:
        cat = "lost_both"

    return {"query": q, "gold_idx": gold, "category": cat,
            "source": meta[gold].get("source", ""),
            "vec_pos": vec_pos, "bm_pos": bm_pos, "fuse_pos": fuse_pos,
            "rerank_pos": rerank_pos, "final_pos": final_pos,
            "bm_gold_score": round(gold_bm_score, 2), "bm_top_score": round(bm_top_score, 2),
            "cos_top": round(cos_top, 4), "bigram_jaccard": round(jaccard, 4),
            "gold_len": len(gold_content),
            "gold_head": gold_content[:60]}


def _valid(m):
    """R4 过滤同款：丢弃过短且非表格块。"""
    c = m.get("content", "")
    return len(c) >= 40 or m.get("type") == "table" or "<table" in c.lower()


_RETR = None


def main():
    global _RETR
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    ev._load_emb_cache()
    qa = ev.load_or_build_qa(ev.get_index_md5())
    print(f"QA 金标 {len(qa)} 对， workers={workers}")

    index = faiss.read_index(ascii_faiss_path(ev.INDEX_FILE))
    meta = ev._load_metadata()
    with open(ev.BM25_FILE, "rb") as f:
        bm25 = pickle.load(f)
    _RETR = HybridRetriever(index, meta, bm25=bm25, use_bm25=True, use_rerank=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    outs = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, it, index, meta, bm25): it for it in qa}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                outs.append(fut.result())
            except Exception as e:
                it = futs[fut]
                outs.append({"query": it["query"], "gold_idx": it["gold_idx"],
                             "category": "error", "error": str(e)[:120]})
            if n % 30 == 0:
                print(f"  进度 {n}/{len(qa)}（{time.time()-t0:.0f}s）")

    cnt = Counter(o["category"] for o in outs)
    n = len(outs)
    print("\n=== 证据覆盖诊断（183 对，线上口径 k=5+rerank+MMR0.72+cap3） ===")
    for c in ("hit", "lost_at_rerank", "lost_at_fusion", "lost_both", "error"):
        if cnt.get(c):
            print(f"  {c:16s} {cnt[c]:3d} 条  ({cnt[c]/n:.1%})")

    strict = cnt.get("hit", 0) / max(n - cnt.get("error", 0), 1)
    print(f"  严格证据覆盖率（gold 在最终 top-5）: {strict:.3f}")

    for cat in ("lost_at_rerank", "lost_at_fusion", "lost_both"):
        sub = [o for o in outs if o["category"] == cat]
        if not sub:
            continue
        print(f"\n--- {cat} 明细 ---")
        if cat == "lost_at_rerank":
            for o in sub:
                print(f"  rerank_pos={o['rerank_pos']} | {o['query'][:30]!r} "
                      f"| gold: {o['source'][:30]}")
        elif cat == "lost_at_fusion":
            for o in sub:
                print(f"  vec_pos={o['vec_pos']} bm_pos={o['bm_pos']} | "
                      f"{o['query'][:30]!r} | gold: {o['source'][:30]}")
        else:
            for o in sorted(sub, key=lambda x: x["bigram_jaccard"]):
                print(f"  jac={o['bigram_jaccard']:.3f} cos={o['cos_top']:.3f} "
                      f"bm={o['bm_gold_score']}/{o['bm_top_score']} len={o['gold_len']} | "
                      f"{o['query'][:26]!r} | {o['gold_head'][:40]!r}")

    out = {"__diag_meta__": {"ts": int(time.time()), "n": n, "k": K,
                             "index_md5": ev.get_index_md5(),
                             "口径": "vec20+bm20+RRF12(0.5/0.5)+rerank5+MMR0.72+cap3",
                             "workers": workers},
           "summary": dict(cnt), "strict_coverage": round(strict, 4),
           "details": outs}
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUT)
    print(f"\n已落盘: {os.path.relpath(OUT, HERE)}（{time.time()-t0:.0f}s）")


if __name__ == "__main__":
    main()

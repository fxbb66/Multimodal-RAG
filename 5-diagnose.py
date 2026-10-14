# -*- coding: utf-8 -*-
"""
5-diagnose.py — 诊断档固化（2026-10-01）
把此前仅以临时脚本存在、不可审计的三组诊断入库：
  ① output/diag_pool.json      候选池流水 + 单因素消融（A/B1/B2/C）
  ② output/diag_mmr_fix.json   MMR 修复前后口径（F1/F2/F3，修复后版本）
  ③ output/diag_answer_k.json  答案侧 2×2（{k=3,k=5} × {MMR+限额2 开, 关}）
  ④ output/diag_calib.json     阈值标定网格（cap∈{1,2,3,不限} × λ∈{0.40,0.55,0.72,0.85}）

每个产物均带头信息：
  __diag_meta__ = {gen_ts, gen_model, embedding_model, rerank_topn,
                   index_md5, index_total, pipeline_ver}
每行结果自带 {k, use_mmr, mmr_lambda, max_per_source} 生效口径。
口径与主评估 4-evaluate.py 完全同源（复用其 evaluate/evaluate_answers/缓存/QA 加载）。

用法：
  python 5-diagnose.py            # 跑全部四节
  python 5-diagnose.py pool       # 只跑候选池诊断（检索侧，约 2 分钟）
  python 5-diagnose.py mmr        # 只跑 MMR 验证（检索侧，约 1.5 分钟）
  python 5-diagnose.py answer     # 只跑答案 2×2（含 LLM 生成，按缓存命中量计费）
  python 5-diagnose.py calib      # 只跑阈值标定网格（检索侧，约 10 分钟）
"""
import importlib.util
import json
import os
import pickle
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "lib"))

# 以模块方式加载 4-evaluate.py（其 main() 有 __main__ 守卫，加载不触发评估）
_spec = importlib.util.spec_from_file_location("ev", os.path.join(_HERE, "4-evaluate.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

import faiss  # noqa: E402
from retriever import (HybridRetriever, ascii_faiss_path,  # noqa: E402
                       RERANK_STATS, POOL_STATS, _STATS_LOCK)

OUT_DIR = os.path.join(_HERE, "output")


def _diag_meta():
    return {
        "gen_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gen_model": ev.GEN_MODEL,
        "embedding_model": ev.EMBEDDING_MODEL,
        "embedding_dim": ev.EMBEDDING_DIM,
        "rerank_topn": ev.RERANK_TOPN,
        "max_per_source_setting": ev.MAX_PER_SOURCE,
        "mmr_lambda_setting": ev.MMR_LAMBDA,
        "index_md5": ev.get_index_md5(),
        "pipeline_ver": ev.PIPELINE_VER,
        "script": "5-diagnose.py",
    }


def _write_diag(name, rows):
    payload = {"__diag_meta__": _diag_meta(), "results": rows}
    dst = os.path.join(OUT_DIR, name)
    tmp = dst + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, dst)
    print(f"[落盘] {os.path.relpath(dst, _HERE)}（{len(rows)} 行，index={payload['__diag_meta__']['index_md5']}）")


def _reset_pool():
    with _STATS_LOCK:
        for key in POOL_STATS:
            POOL_STATS[key] = 0


def _pool_snapshot():
    with _STATS_LOCK:
        snap = dict(POOL_STATS)
    n = snap.pop("queries") or 1
    return {k: round(v / n, 2) for k, v in snap.items()}


def _load_world():
    """加载索引/BM25/QA/缓存，返回 (r_rr, r_hyb, qa, vec_cache)。"""
    ev._load_emb_cache()
    index = faiss.read_index(ascii_faiss_path(ev.INDEX_FILE))
    if index.d != ev.EMBEDDING_DIM:
        raise SystemExit(f"[维度不匹配] {index.d} != {ev.EMBEDDING_DIM}")
    with open(ev.METADATA_FILE, encoding="utf-8") as f:
        meta = json.load(f)
    with open(ev.BM25_FILE, "rb") as f:
        bm25 = pickle.load(f)
    qa = ev.load_or_build_qa(ev.get_index_md5())
    r_rr = HybridRetriever(index, meta, bm25=bm25, use_bm25=True,
                           use_rerank=True, rerank_topn=ev.RERANK_TOPN)
    r_hyb = HybridRetriever(index, meta, bm25=bm25, use_bm25=True, use_rerank=False)
    return r_rr, r_hyb, qa, {}


def _run_retrieval(r_rr, qa, vec_cache, label, **kw):
    _reset_pool()
    RERANK_STATS["calls"] = RERANK_STATS["fallbacks"] = 0
    res = ev.evaluate(r_rr, qa, vec_cache, k=5, use_rerank=True, label=label, **kw)
    res["pool_avg"] = _pool_snapshot()
    res["rerank_calls"] = RERANK_STATS["calls"]
    res["rerank_fallbacks"] = RERANK_STATS["fallbacks"]
    return res


def diag_pool(r_rr, qa, vec_cache):
    """① 候选池流水 + 单因素消融（全部 k=5；定位损失发生在哪一段）。"""
    print("\n=== ① 候选池流水 / 单因素消融（k=5） ===")
    rows = [
        _run_retrieval(r_rr, qa, vec_cache, "A 无后处理(k=5)"),
        _run_retrieval(r_rr, qa, vec_cache, "B1 仅MMR(k=5,lam0.72)",
                       use_mmr=True, mmr_lambda=0.72),
        _run_retrieval(r_rr, qa, vec_cache, "B2 仅限额(k=5,cap2)",
                       max_per_source=2),
        _run_retrieval(r_rr, qa, vec_cache, "C MMR+限额(k=5,线上同参)",
                       use_mmr=True, mmr_lambda=ev.MMR_LAMBDA,
                       max_per_source=ev.MAX_PER_SOURCE),
    ]
    _write_diag("diag_pool.json", rows)


def diag_mmr(r_rr, qa, vec_cache):
    """② MMR 修复验证（位次相关性）：λ 两档 + 叠加限额。"""
    print("\n=== ② MMR 修复验证（位次相关性，k=5） ===")
    rows = [
        _run_retrieval(r_rr, qa, vec_cache, "F1 修复MMR(k=5,lam0.72)",
                       use_mmr=True, mmr_lambda=0.72),
        _run_retrieval(r_rr, qa, vec_cache, "F2 修复MMR(k=5,lam0.55)",
                       use_mmr=True, mmr_lambda=0.55),
        _run_retrieval(r_rr, qa, vec_cache, "F3 修复MMR+限额2(k=5,lam0.72)",
                       use_mmr=True, mmr_lambda=0.72, max_per_source=2),
    ]
    _write_diag("diag_mmr_fix.json", rows)


def diag_answer(r_rr, qa, vec_cache):
    """③ 答案侧 2×2：{k=3,k=5} × {MMR+限额2 开, 关}。
    与主评估同网格；口径经 _ANS_SCOPE 写入缓存 key，不同格之间不串缓存。"""
    print("\n=== ③ 答案侧 2×2（k × 后处理） ===")
    ev._load_ans_cache()
    grid = [
        ("答案(k=3,后处理关)", dict(k=3)),
        ("答案(k=3,后处理开:MMR+限额2)", dict(k=3, max_per_source=ev.MAX_PER_SOURCE,
                                              use_mmr=True, mmr_lambda=ev.MMR_LAMBDA)),
        ("答案(k=5,后处理关)", dict(k=5)),
        ("答案(k=5,后处理开:MMR+限额2)", dict(k=5, max_per_source=ev.MAX_PER_SOURCE,
                                              use_mmr=True, mmr_lambda=ev.MMR_LAMBDA)),
    ]
    rows = []
    for label, kw in grid:
        res = ev.evaluate_answers(r_rr, qa, vec_cache, label, use_rerank=True, **kw)
        rows.append({"label": label, **res})
        ev._save_ans_cache()
    _write_diag("diag_answer_k.json", rows)
    ev.log.info("答案缓存 hit=%s miss=%s", ev.ANS_CACHE_HIT["hit"], ev.ANS_CACHE_HIT["miss"])


def diag_calib(r_rr, qa, vec_cache):
    """④ 阈值标定网格（k=5，检索侧）：限额 cap 与 MMR λ 二维扫描。
    新向量空间（1152 维）重标定 settings.MAX_PER_SOURCE / MMR_LAMBDA 的数据依据；
    cap=0 表示不限额。每行带完整生效口径，落盘 diag_calib.json 供回写决策审计。"""
    print("\n=== ④ 阈值标定网格（k=5：cap 扫描 + λ 扫描） ===")
    rows = []
    # cap 扫描：固定 λ=0.72（MMR 开），cap ∈ {1,2,3,0(不限)}
    for cap in (1, 2, 3, 0):
        rows.append(_run_retrieval(
            r_rr, qa, vec_cache, f"标定 cap={cap or '不限'}/λ0.72",
            use_mmr=True, mmr_lambda=0.72, max_per_source=cap or None))
    # λ 扫描：固定 cap=2（线上当前限额），λ ∈ {0.40,0.55,0.72,0.85}
    for lam in (0.40, 0.55, 0.72, 0.85):
        rows.append(_run_retrieval(
            r_rr, qa, vec_cache, f"标定 cap=2/λ{lam:.2f}",
            use_mmr=True, mmr_lambda=lam, max_per_source=2))
    _write_diag("diag_calib.json", rows)


def main():
    section = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    ev.log.info("=== 5-diagnose 开始 section=%s index=%s pipeline=%s ===",
                section, ev.get_index_md5(), ev.PIPELINE_VER)
    need_world = section in ("all", "pool", "mmr", "answer", "calib")
    if not need_world:
        raise SystemExit(f"未知小节: {section}（可选 pool/mmr/answer/calib/all）")
    r_rr, _r_hyb, qa, vec_cache = _load_world()
    if section in ("all", "pool"):
        diag_pool(r_rr, qa, vec_cache)
    if section in ("all", "mmr"):
        diag_mmr(r_rr, qa, vec_cache)
    if section in ("all", "answer"):
        diag_answer(r_rr, qa, vec_cache)
    if section in ("all", "calib"):
        diag_calib(r_rr, qa, vec_cache)
    print(f"\n诊断完成，总耗时 {time.time() - t0:.0f}s")
    ev.log.info("=== 5-diagnose 结束 %.0fs ===", time.time() - t0)


if __name__ == "__main__":
    main()

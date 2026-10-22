# -*- coding: utf-8 -*-
"""S6 收尾：82 旧基线 vs 183 S1 基线 vs 183 最终版（S4 门修复后）三方对比。

最终版与 S1 基线的唯一差异 = S4 指代门放宽 + 改写缓存 v2→v4 + 金标第三轮修订，
用于隔离"链路修复"对检索/答案指标的净效应（预期：仅指代类题目流向 clarify，
其余题目读数不变；若主指标波动超 2pt 需人工复核）。

用法：python compare_final.py
"""
import json
import os

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
FILES = [("82基线", "eval_results_82_baseline.json"),
         ("183-S1", "eval_results_183.json"),
         ("183-最终", "eval_results_183_final.json")]


def load(name):
    with open(f"{HERE}/{name}", encoding="utf-8") as f:
        return json.load(f)


def rowkey(r):
    return r.get("label") or r.get("query_mode") or "?"


datas = []
for tag, fn in FILES:
    try:
        d = load(fn)
        datas.append((tag, {rowkey(r): r for r in d["results"]}, d.get("config", {})))
    except FileNotFoundError:
        datas.append((tag, {}, {}))

_, new_rows, new_cfg = datas[2]
_, s1_rows, s1_cfg = datas[1]
_, old_rows, _ = datas[0]

if not new_rows:
    raise SystemExit("[缺失] eval_results_183_final.json 不存在，先跑最终版 4-evaluate.py")

METRICS = ("recall@k", "recall@3", "mrr", "evidence_coverage", "answer_hit",
           "answer_hit_given_evidence", "action_acc", "clarify_recall",
           "clarify_f1", "type_hit_rate", "multi_acc", "term_hit_rate",
           "image_recall_at_k", "n")

print("=== 三方对比（82 基线 → 183 S1 → 183 最终版） ===")
print(f"{'档位':<40} {'指标':<24} {'82':>8} {'S1':>8} {'最终':>8} {'最终-S1':>9}")
attention = []
for label in old_rows:
    orow, s1row, nrow = old_rows.get(label), s1_rows.get(label), new_rows.get(label)
    if not nrow:
        continue
    for k in METRICS:
        ov = orow.get(k) if orow else None
        sv = s1row.get(k) if s1row else None
        nv = nrow.get(k)
        if not all(isinstance(v, (int, float)) for v in (ov, sv, nv) if v is not None) \
                or nv is None:
            continue
        d = nv - sv if isinstance(sv, (int, float)) else None
        flag = ""
        if d is not None and k in ("recall@k", "answer_hit", "action_acc",
                                   "type_hit_rate", "multi_acc") and d < -0.02:
            flag = "  ← 关注"
            attention.append((label, k, sv, nv, d))
        ds = f"{d:>+9.3f}" if d is not None else f"{'—':>9}"
        print(f"{label[:38]:<40} {k:<24} {ov if ov is not None else '—':>8} "
              f"{sv if sv is not None else '—':>8} {nv:>8.3f} {ds}{flag}")

print("\n=== 最终版新增档位（S1 基线没有） ===")
for label in new_rows:
    if label not in s1_rows:
        r = new_rows[label]
        keys = [k for k in ("recall@k", "image_recall_at_k", "n") if k in r]
        print(f"  {label[:60]}  " + "  ".join(f"{k}={r[k]}" for k in keys))

print("\n=== 最终版 vs S1：主指标回退判定（>2pt 记关注） ===")
if attention:
    for label, k, s, n, d in attention:
        print(f"  [关注] {label} {k}: {s:.3f} → {n:.3f} ({d:+.3f})")
else:
    print("  无主指标回退超 2pt 的档位")

rr = new_rows.get("混合+Rerank") or {}
r5 = rr.get("recall@k")
print(f"\n  §8 触发器（recall@5 跌破 0.90 才触发框架化修复）:")
print(f"    82={old_rows.get('混合+Rerank', {}).get('recall@k')} "
      f"S1={s1_rows.get('混合+Rerank', {}).get('recall@k')} "
      f"最终={r5} → {'触发' if (r5 or 1) < 0.90 else '未触发'}")

print("\n=== 配置指纹核对（S1 vs 最终，应仅缓存版本/时间不同） ===")
for k in ("index_md5", "gen_model", "embedding_model", "rerank_topn",
          "max_per_source", "mmr_lambda", "online_system_prompt_sha1",
          "rewrite_cache_ver", "answer_eval_n", "qa_count"):
    a, b = s1_cfg.get(k), new_cfg.get(k)
    mark = "" if a == b else "  ← 变化"
    print(f"  {k:<26} S1={a}  最终={b}{mark}")

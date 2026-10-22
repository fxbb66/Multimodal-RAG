# -*- coding: utf-8 -*-
"""S7+S8 合跑对比：183 最终版（旧索引 5c879a20569d，S7/S8 前）vs 183-S7S8
（新索引 d59376571cd8 + 路径级并集 + 前端/诊断适配）。

合跑口径说明：本轮为用户拍板的"合起来跑"——S7（融合并集）与 S8（双向量索引）
同轮生效，无法从单轮读数中分离两者净效应；若某档回归超 2pt，用
FUSION_PATH_UNION_TOPK=0（关 S7）或回滚 backup_pre_dualvec 索引（关 S8）
补跑隔离轮定位。

用法：python 17-compare_s7s8.py（需先跑完 4-evaluate.py 并复制产物）
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
FILES = [("183-最终", "eval_results_183_final.json"),
         ("183-S7S8", "eval_results_183_s7s8.json")]


def load(fn):
    with open(os.path.join(OUT, fn), encoding="utf-8") as f:
        return json.load(f)


def rowkey(r):
    return r.get("label") or r.get("query_mode") or "?"


datas = []
for tag, fn in FILES:
    if not os.path.exists(os.path.join(OUT, fn)):
        print(f"[缺失] {fn}，先完成对应轮次的全量评估")
        sys.exit(1)
    d = load(fn)
    datas.append((tag, {rowkey(r): r for r in d["results"]}, d.get("config", {})))

_, old_rows, old_cfg = datas[0]
_, new_rows, new_cfg = datas[1]

METRICS = ("recall@k", "recall@3", "mrr", "evidence_coverage", "answer_hit",
           "answer_hit_given_evidence", "action_acc", "clarify_recall",
           "clarify_f1", "type_hit_rate", "multi_acc", "term_hit_rate",
           "image_recall_at_k", "n")

print("=== S7+S8 合跑对比（旧索引+S7/S8前 → 新索引+S7/S8） ===")
print(f"{'档位':<44} {'指标':<22} {'前':>8} {'后':>8} {'Δ':>9}")
attention = []
for label in old_rows:
    orow, nrow = old_rows.get(label), new_rows.get(label)
    if not nrow:
        print(f"{label[:42]:<44} （新轮缺失该档）")
        continue
    for k in METRICS:
        ov, nv = orow.get(k), nrow.get(k)
        if not isinstance(ov, (int, float)) or not isinstance(nv, (int, float)):
            continue
        d = nv - ov
        flag = ""
        if k in ("recall@k", "answer_hit", "answer_hit_given_evidence",
                 "action_acc", "type_hit_rate", "multi_acc",
                 "image_recall_at_k") and d < -0.02:
            flag = "  ← 关注"
            attention.append((label, k, ov, nv, d))
        print(f"{label[:42]:<44} {k:<22} {ov:>8.3f} {nv:>8.3f} {d:>+9.3f}{flag}")

print("\n=== 新增/消失档位 ===")
for label in new_rows:
    if label not in old_rows:
        r = new_rows[label]
        keys = [k for k in ("recall@k", "image_recall_at_k", "n") if k in r]
        print(f"  + {label[:60]}  " + "  ".join(f"{k}={r[k]}" for k in keys))
for label in old_rows:
    if label not in new_rows:
        print(f"  - {label[:60]}（新轮消失）")

print("\n=== 主指标回退判定（>2pt 记关注） ===")
if attention:
    for label, k, o, n, d in attention:
        print(f"  [关注] {label} {k}: {o:.3f} → {n:.3f} ({d:+.3f})")
else:
    print("  无主指标回退超 2pt 的档位")

rr = new_rows.get("混合+Rerank") or {}
r5 = rr.get("recall@k")
old_r5 = old_rows.get("混合+Rerank", {}).get("recall@k")
print(f"\n  §8 触发器（recall@5 跌破 0.90 才触发框架化修复）:")
print(f"    前={old_r5} 后={r5} → {'触发' if (r5 or 1) < 0.90 else '未触发'}")

print("\n=== 配置指纹核对（应仅索引/缓存计数/耗时不同） ===")
for k in ("index_md5", "index_total", "gen_model", "embedding_model",
          "rerank_topn", "rerank_chain", "max_per_source", "use_mmr",
          "mmr_lambda", "online_system_prompt_sha1", "rewrite_cache_ver",
          "answer_eval_n", "qa_count", "ans_cache_hit", "ans_cache_miss",
          "intent_confidence_min"):
    a, b = old_cfg.get(k), new_cfg.get(k)
    mark = "" if a == b else "  ← 变化"
    print(f"  {k:<26} 前={a}  后={b}{mark}")

print("\n=== 图片金标档（S8 直接受益档） ===")
for label in new_rows:
    if "图片" in label:
        r = new_rows[label]
        print(f"  {label}: image_recall@5={r.get('image_recall_at_k')} "
              f"n={r.get('n')} lost={r.get('lost_breakdown')}")
        o = old_rows.get(label, {})
        if o:
            print(f"    前值: image_recall@5={o.get('image_recall_at_k')} "
                  f"lost={o.get('lost_breakdown')}")

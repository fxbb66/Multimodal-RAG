# -*- coding: utf-8 -*-
"""S1 收尾：183 新基线 vs 82 旧基线逐档对比（一次性工具，跑完可留档）。"""
import json
import os
import sys

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def load(name):
    with open(f"{HERE}/{name}", encoding="utf-8") as f:
        return json.load(f)


def rowkey(r):
    return r.get("label") or r.get("query_mode") or "?"


new = load("eval_results_183.json")
old = load("eval_results_82_baseline.json")
new_rows = {rowkey(r): r for r in new["results"]}
old_rows = {rowkey(r): r for r in old["results"]}

print("=== 逐档对比（183 新基线 vs 82 旧基线） ===")
print(f"{'档位':<44} {'指标':<22} {'82':>8} {'183':>8} {'Δ':>8}")
regressions = []
for label, orow in old_rows.items():
    nrow = new_rows.get(label)
    if not nrow:
        print(f"{label[:42]:<44} {'(183 无此档)':<22}")
        continue
    for k in ("recall@5", "recall@3", "strict_recall", "mrr", "evidence_coverage",
              "answer_hit", "answer_hit_given_evidence", "gen_loss_when_evidence",
              "action_acc", "clarify_rate", "fallback_rate", "multi_intent_rate",
              "type_hit_rate", "multi_acc", "term_hit_rate"):
        if k in orow and k in nrow and isinstance(orow[k], (int, float)) \
                and isinstance(nrow[k], (int, float)):
            d = nrow[k] - orow[k]
            flag = ""
            # 回退判定：主指标下降超 2pt 记关注
            if k in ("recall@5", "recall@3", "answer_hit", "action_acc",
                     "type_hit_rate", "multi_acc") and d < -0.02:
                flag = "  ← 关注"
                regressions.append((label, k, orow[k], nrow[k], d))
            print(f"{label[:42]:<44} {k:<22} {orow[k]:>8.3f} {nrow[k]:>8.3f} {d:>+8.3f}{flag}")

print("\n=== 新增档位（82 基线没有） ===")
for label in new_rows:
    if label not in old_rows:
        r = new_rows[label]
        keys = [k for k in ("recall@5", "image_recall_at_k", "n") if k in r]
        print(f"  {label[:60]}  " + "  ".join(f"{k}={r[k]}" for k in keys))

print("\n=== 触发回退判据 ===")
if regressions:
    for label, k, o, n, d in regressions:
        print(f"  [关注] {label} {k}: {o:.3f} → {n:.3f} ({d:+.3f})")
else:
    print("  无主指标回退超 2pt 的档位")
main_rr = new_rows.get("混合+Rerank") or {}
r5 = main_rr.get("recall@5")
print(f"  §8 触发器（recall@5 跌破 0.90 才触发框架化修复）: "
      f"183 主线 recall@5={r5} → {'触发' if (r5 or 1) < 0.90 else '未触发'}")

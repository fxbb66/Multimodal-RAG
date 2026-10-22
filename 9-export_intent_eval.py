# -*- coding: utf-8 -*-
"""
意图金标评估导出脚本（8-金标扩充版本）
=====================================
只跑意图金标评估一项（不跑检索/答案网格），把结果连同降级审计与时延统计落盘，
供报告引用。复用 output/eval_rewrite_cache.json 磁盘缓存：query 未变时严格复现
上一次真实 LLM 运行的结果（含 degraded/degrade_reason/elapsed_ms），零 LLM 调用。

落盘口径（全部由 details 现算，保证文件自洽，不依赖进程内计数器）：
  need_rewrite          pred_types 非空——LLM 判定需要改写/拆分/消解的样本数
  true_degraded         degraded=True——LLM 调用失败/低置信/JSON 抢救失败，走了兜底
  true_degrade_rate     true_degraded / need_rewrite
  degraded_with_impact  degraded=True 且最终结果受影响（action/type/term/multi 任一未过）
  llm_clear_not_fault   pred_types 为空且未降级且 action=retrieve——LLM 判 clear，
                        本就无需改写，不是故障（口径修正的关键：不能再计入降级率）
  step2_ms_*            intent_ms 即 Step2 阶段耗时（Step1 完成后重置计时，
                        clarify/多意图路径不产 elapsed_ms，不参与统计）

用法：
    python 9-export_intent_eval.py [输出文件名]
默认输出 output/intent_gold_eval_after_gold_revision.json
"""
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "eval_mod", os.path.join(HERE, "4-evaluate.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)  # __main__ 保护确保不会触发完整评估

OUT = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(HERE, "output", "intent_gold_eval_after_gold_revision.json")

t0 = time.time()
ev._load_rewrite_cache()          # 装载改写缓存（query 未变 -> 全部命中，零 LLM 调用）
res = ev.evaluate_intent_gold()   # 线上同款 intent_rewrite 流水线
if res is None:
    raise SystemExit("意图金标集缺失，退出")

det = res["details"]
need_rewrite = sum(1 for x in det if x["pred_types"])
true_degraded = sum(1 for x in det if x["degraded"])
with_impact = sum(1 for x in det if x["degraded"] and not (
    x["action_ok"] and x["type_hit"] is not False and x["multi_ok"]
    and x["term_hit"] is not False))
clear_not_fault = sum(1 for x in det if not x["pred_types"] and not x["degraded"]
                      and x["pred_action"] == "retrieve")
ms = sorted(x["intent_ms"] for x in det if x["intent_ms"])

audit = {
    "need_rewrite": need_rewrite,
    "true_degraded": true_degraded,
    "true_degrade_rate": round(true_degraded / need_rewrite, 4) if need_rewrite else 0.0,
    "degraded_with_impact": with_impact,
    "llm_clear_not_fault": clear_not_fault,
    "degraded_samples": [
        {"query": x["query"], "reason": x["degrade_reason"],
         "action_ok": x["action_ok"]} for x in det if x["degraded"]],
    "def": "true_degraded=degraded=True；llm_clear_not_fault=types 空且未降级且 retrieve"
           "（LLM 判 clear 非故障，不计入降级率）",
}
latency = {
    "step2_ms_median": int(statistics.median(ms)) if ms else None,
    "step2_ms_max": max(ms) if ms else None,
    "step2_ms_n": len(ms),
    "def": "intent_ms = Step2 阶段耗时（Step1 完成后 _budget.t0 重置）；"
           "clarify/多意图/规则门路径不产该字段",
}

out = {
    "__baseline_meta__": {
        "ts": int(time.time()),
        "n": len(det),
        "variant": "gold_revised",
        "elapsed_sec": round(time.time() - t0, 1),
        "index_md5": ev.get_index_md5(),
        "rw_cache_hits": ev.RW_CACHE_HIT["hit"],
        "rw_cache_misses": ev.RW_CACHE_HIT["miss"],
        "note": "金标第二轮口径修订后的复测：query 集合未变，命中改写缓存严格复现上次真实"
                "LLM 运行结果；尺子（gold 标注）已按正确行为修订",
    },
    "summary": {k: v for k, v in res.items() if k != "details"},
    "intent_stats": "缓存复跑，进程内计数器不反映本次（见 __baseline_meta__.rw_cache_hits）",
    "degrade_audit": audit,
    "latency": latency,
    "details": det,
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
tmp = OUT + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
os.replace(tmp, OUT)
# 持久化改写缓存（2026-10-21 修复）：本脚本此前只加载不保存，缓存一旦被清，
# 每次导出都全量重调 LLM——与"query 未变则零调用复测"的设计相悖。
# 缓存带版本指纹（REWRITE_CACHE_VER），规则变更时旧缓存自动作废，不会串味。
ev._save_rewrite_cache()
print(f"\n已导出: {os.path.relpath(OUT, HERE)}")
print(f"缓存命中 {ev.RW_CACHE_HIT['hit']} / 未命中 {ev.RW_CACHE_HIT['miss']}")
print(f"降级审计: {json.dumps(audit, ensure_ascii=False)}")
print(f"时延: {json.dumps(latency, ensure_ascii=False)}")

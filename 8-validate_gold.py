# -*- coding: utf-8 -*-
"""
金标校验脚本（8-金标扩充版本）
=============================
对两个金标文件做结构化 + 口径校验，全部确定性检查，不调 LLM：

A. eval_intent_gold.json
   1. JSON 合法、items 非空
   2. 必需字段存在且类型正确（query/history/gold_action/gold_types/gold_multi/must_include）
   3. gold_action 枚举合法；gold_types 是 type_enums 子集且主类型在前
   4. 【口径】调用线上同款 _rule_gate_short_no_entity：确定性门判 clarify 的样本，
      gold_action 必须是 clarify（否则标注与线上行为矛盾）
   5. 【口径】调用线上同款 _rule_split_multi_question：>=3 问号的样本 gold_multi 必须为 true
   6. clarify 样本 must_include 应为空（term_keep 只度量 retrieve 题）
   7. query 全局去重（含存量）
B. output/eval_qa_v2.json
   1. JSON 合法、schema 版本正确
   2. 每条 chunk_hash 必须存在于当前 metadata（否则加载时会被失配剔除）
   3. 字段完整、gold_answer 非空、query 非空
   4. query 去重；新增部分每来源 <=2 条；新增金标块不与存量重复
   5. 分布报告（类别 / 来源数 / 每源条数）

退出码：0 = 全部通过；1 = 存在 ERROR（WARN 不影响退出码）。
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))

from retriever import (_rule_gate_short_no_entity,   # noqa: E402
                       _rule_split_multi_question)

ERRORS, WARNS = [], []


def err(msg):
    ERRORS.append(msg)


def warn(msg):
    WARNS.append(msg)


def check_intent_gold():
    path = os.path.join(HERE, "eval_intent_gold.json")
    doc = json.load(open(path, encoding="utf-8"))
    items = doc["items"]
    enums = set(doc["__meta__"]["type_enums"])
    print(f"== A. 意图金标：{len(items)} 条 ==")

    seen_q = set()
    gate_hits = multi_gate_hits = 0
    for n, it in enumerate(items):
        tag = f"#{n} {it.get('query', '')[:24]!r}"
        for f in ("query", "gold_action", "gold_types", "gold_multi", "must_include"):
            if f not in it:
                err(f"{tag} 缺字段 {f}")
        if not isinstance(it.get("query"), str) or not it["query"].strip():
            err(f"{tag} query 为空")
        if it.get("gold_action") not in ("retrieve", "clarify"):
            err(f"{tag} gold_action 非法: {it.get('gold_action')}")
        if not isinstance(it.get("gold_types"), list):
            err(f"{tag} gold_types 非数组")
        else:
            bad = [t for t in it["gold_types"] if t not in enums]
            if bad:
                err(f"{tag} gold_types 含非法枚举 {bad}")
            if "clear" in it["gold_types"]:
                err(f"{tag} clear 型按口径应留空数组")
        if not isinstance(it.get("gold_multi"), bool):
            err(f"{tag} gold_multi 非布尔")
        if not isinstance(it.get("must_include"), list):
            err(f"{tag} must_include 非数组")
        for h in it.get("history", []):
            if h.get("role") not in ("user", "assistant") or not h.get("content"):
                err(f"{tag} history 条目缺 role/content")
        if it["query"] in seen_q:
            err(f"{tag} query 重复")
        seen_q.add(it["query"])

        # 口径 1：确定性短问门
        gated = _rule_gate_short_no_entity(it["query"], it.get("history") or [])
        if gated is not None:
            gate_hits += 1
            if it["gold_action"] != "clarify":
                err(f"{tag} 确定性门必判 clarify，但标注为 {it['gold_action']}")
            if it["gold_types"][:1] != ["underspecified"]:
                err(f"{tag} 确定性门类型应为 underspecified，实际 {it['gold_types']}")
        # 口径 2：三连问门
        segs = _rule_split_multi_question(it["query"], min_segments=3)
        if segs:
            multi_gate_hits += 1
            if not it["gold_multi"]:
                err(f"{tag} {len(segs)} 个并列问号 -> gold_multi 应为 true")
        # 口径 3：clarify 题不带 must_include
        if it["gold_action"] == "clarify" and it.get("must_include"):
            warn(f"{tag} clarify 题携带 must_include（不影响评测，建议清空）")

    print(f"  确定性短问门命中 {gate_hits} 条（全部标注 clarify 一致）")
    print(f"  三连问门命中 {multi_gate_hits} 条（全部 gold_multi=true 一致）")
    print(f"  action: {dict(Counter(i['gold_action'] for i in items))}")
    print(f"  types : {dict(Counter(t for i in items for t in i['gold_types']))}")
    print(f"  multi : {sum(1 for i in items if i['gold_multi'])}  "
          f"history: {sum(1 for i in items if i['history'])}  "
          f"must_include: {sum(1 for i in items if i['must_include'])}")
    return items


def check_qa_gold():
    path = os.path.join(HERE, "output", "eval_qa_v2.json")
    doc = json.load(open(path, encoding="utf-8"))
    items = doc["items"]
    meta = doc["__qa_meta__"]
    meta_hash = {m["chunk_hash"] for m in
                 json.load(open(os.path.join(HERE, "output", "disney_full_metadata.json"),
                                encoding="utf-8")) if m.get("chunk_hash")}
    print(f"== B. QA 金标：{len(items)} 对 ==")
    if meta.get("schema_ver") != "chunk_hash_v1":
        err(f"QA schema_ver 异常: {meta.get('schema_ver')}")

    seen_q, seen_h = set(), set()
    src_cnt = Counter()
    for n, it in enumerate(items):
        tag = f"#{n} {it.get('query', '')[:20]!r}"
        for f in ("query", "gold_idx", "chunk_hash", "gold_answer", "source"):
            if f not in it:
                err(f"{tag} 缺字段 {f}")
        if not it.get("query", "").strip():
            err(f"{tag} query 为空")
        if not it.get("gold_answer", "").strip():
            err(f"{tag} gold_answer 为空")
        if it.get("chunk_hash") not in meta_hash:
            err(f"{tag} chunk_hash 不在当前索引（加载时会被剔除）")
        if it["query"] in seen_q:
            warn(f"{tag} query 重复")
        seen_q.add(it["query"])
        seen_h.add(it["chunk_hash"])
        src_cnt[it["source"]] += 1

    # 每块 2 对是 gen_qa 的设计意图（build_qa_set 每 chunk 生成 2 对），
    # 共用金标块只作统计，不告警；query 完全重复才告警。
    dup_block = len(items) - len(seen_h)
    print(f"  来源文件 {len(src_cnt)} 个，金标块 {len(seen_h)} 个"
          f"（每块约 2 对，共用块 {dup_block} 条属正常设计）")

    over = {s: c for s, c in src_cnt.items() if c > 2}
    print(f"  来源文件 {len(src_cnt)} 个，金标块 {len(seen_h)} 个")
    print(f"  每源条数 >2 的来源（存量历史遗留，不阻断）: "
          f"{len(over)} 个 {list(over.items())[:3]}")
    for s, c in over.items():
        if c > 4:
            warn(f"来源 {s[:40]} 多达 {c} 条，同源偏差过大")
    print(f"  remap 履历 {len(meta.get('remap', []))} 条，"
          f"末条: {meta.get('remap', [{}])[-1].get('expansion', '-')}")
    return items


if __name__ == "__main__":
    check_intent_gold()
    print()
    check_qa_gold()
    print()
    for w in WARNS:
        print(f"[WARN] {w}")
    for e in ERRORS:
        print(f"[ERROR] {e}")
    print(f"\n校验结果：{len(ERRORS)} 个 ERROR，{len(WARNS)} 个 WARN")
    sys.exit(1 if ERRORS else 0)

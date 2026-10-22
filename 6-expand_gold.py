# -*- coding: utf-8 -*-
"""
QA 金标扩充脚本（8-金标扩充版本专用）
=====================================
在存量 82 对 QA 基础上，用项目原生 gen_qa 管线（与 4-evaluate.py 同源、
同模型 REWRITE_MODEL、同 prompt）从更多代表性 chunk 生成新问答对。

抽样口径（沿用 build_qa_set 的初衷并收紧冲突）：
  1. 候选 = text chunk 且 100 < len(content) < 800（与 build_qa_set 一致）；
  2. 只从【存量 QA 未使用过的来源文件】或【存量使用数 < 2 的来源】中抽取，
     保证新增部分自身每来源 ≤ 2 条，且不与存量金标块重复；
  3. 类别轮询抽样，保持类别均衡；
  4. 每 chunk 生成 2 对（问题|答案），由 gen_qa 内部做长度过滤。

产出：与存量合并后原子写回 output/eval_qa_v2.json（存量条目原样保留，
remap 履历追加一条扩充记录），总量 82 + 新增。
"""
import os
import sys
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))
sys.path.insert(0, HERE)

import importlib.util
spec = importlib.util.spec_from_file_location("ev", os.path.join(HERE, "4-evaluate.py"))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

QA_FILE = os.path.join(HERE, "output", "eval_qa_v2.json")
TARGET_NEW_CHUNKS = int(os.environ.get("EXPAND_TARGET", "100"))
WORKERS = int(os.environ.get("EXPAND_WORKERS", "6"))


def main():
    meta = ev._load_metadata()
    qa_doc = json.load(open(QA_FILE, encoding="utf-8"))
    old_items = qa_doc["items"]
    old_meta = qa_doc.get("__qa_meta__", {})

    used_hashes = set(it["chunk_hash"] for it in old_items)
    src_used = {}
    for it in old_items:
        src_used[it["source"]] = src_used.get(it["source"], 0) + 1

    # 候选池：长度合格 + chunk 未被存量使用
    texts = [(i, m) for i, m in enumerate(meta)
             if m["type"] == "text" and 100 < len(m["content"]) < 800
             and m.get("chunk_hash") and m["chunk_hash"] not in used_hashes]
    # 来源过滤：存量用满 2 条的源不再抽（新增部分每源 ≤2）
    texts = [(i, m) for i, m in texts if src_used.get(m["source"], 0) < 2]

    # 类别轮询
    by_cat = {}
    for i, m in texts:
        by_cat.setdefault(m["category"], []).append((i, m))
    cat_lists = [lst[:] for lst in by_cat.values()]

    samples, src_cnt = [], {}
    pos = 0
    while len(samples) < TARGET_NEW_CHUNKS and any(cat_lists):
        cl = cat_lists[pos % len(cat_lists)]
        pos += 1
        while cl:
            i, m = cl.pop(0)
            if src_cnt.get(m["source"], 0) >= 2:
                continue
            samples.append((i, m))
            src_cnt[m["source"]] = src_cnt.get(m["source"], 0) + 1
            break
        if all(not cl for cl in cat_lists):
            break

    print(f"[抽样] 候选 chunk {len(texts)} 个 / {len(by_cat)} 类，"
          f"实抽 {len(samples)} 个（目标 {TARGET_NEW_CHUNKS}）", flush=True)

    results = [None] * len(samples)
    done = [0]
    lock = threading.Lock()

    def work(k):
        i, m = samples[k]
        for attempt in range(2):  # 空结果的外层重试（gen_qa 内部已有 3 次重试）
            pairs = ev.gen_qa(m["content"])
            if pairs:
                break
            time.sleep(1.0)
        with lock:
            done[0] += 1
            print(f"  [{done[0]}/{len(samples)}] {len(pairs)} 对 "
                  f"({m['source'][:30]})", flush=True)
        return k, pairs

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(work, k) for k in range(len(samples))]
        for fut in as_completed(futs):
            k, pairs = fut.result()
            results[k] = pairs

    new_items, n_pairs, empty_chunks = [], 0, 0
    seen_q = set(it["query"] for it in old_items)  # 与存量重复的 query 直接丢弃
    dup_q = 0
    for (i, m), pairs in zip(samples, results):
        if not pairs:
            empty_chunks += 1
            continue
        for q, a in pairs:
            if q in seen_q:
                dup_q += 1
                continue
            seen_q.add(q)
            new_items.append({"query": q, "gold_idx": i,
                              "chunk_hash": m["chunk_hash"],
                              "gold_answer": a,
                              "source": m["source"],
                              "heading": m.get("heading", "")})
            n_pairs += 1

    # 与存量合并（存量在前，保持既有评估顺序稳定）
    merged = old_items + new_items
    hist = list(old_meta.get("remap", []))
    hist.append({"index_md5": ev.get_index_md5(), "ts": int(time.time()),
                 "total": len(merged), "mapped": len(merged), "missing": 0,
                 "expansion": f"+{len(new_items)} pairs from +{len(samples) - empty_chunks} chunks"})
    ev._write_qa(merged, {"schema_ver": ev.QA_SCHEMA_VER,
                          "born_index_md5": old_meta.get("born_index_md5") or ev.get_index_md5(),
                          "n_chunks": old_meta.get("n_chunks", 0) + len(samples),
                          "ts": int(time.time()), "remap": hist})
    print(f"[完成] 新增 {n_pairs} 对（{len(samples)} chunk，{empty_chunks} 个空结果，"
          f"{dup_q} 条与存量 query 重复已剔除），"
          f"总量 {len(merged)} 对，耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

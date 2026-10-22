# -*- coding: utf-8 -*-
"""
S6+ 口语化问句混入（2026-10-21）
==============================
把 183 对正式 QA 金标改写为"口语化/生活化"变体，扩充 QA 尺子对口语输入的覆盖
（当前 183 题是自包含书面问句，意图链路回退率 96.7%——口语题能给改写/澄清更多活干）。

产物：output/eval_qa_colloquial.json
  - 每条复用父题的 chunk_hash / gold_idx / gold_answer / source / heading（答案不变，只改提问口吻）
  - 新增字段：style="colloquial"、parent_query=原书面问句
  - 由 4-evaluate.py 在 QA_INCLUDE_COLLOQUIAL=1 时并入主评估（默认关，不改既有 183 口径）

用法：
    python 16-gen_colloquial.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from settings import (AGICTO_BASE_URL, REWRITE_API_KEY, REWRITE_MODEL)  # noqa: E402
from openai import OpenAI  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
QA_FILE = os.path.join(OUT, "eval_qa_v2.json")
COLL_FILE = os.path.join(OUT, "eval_qa_colloquial.json")
BATCH = 12
MODEL = REWRITE_MODEL  # qwen3.8-flash：改写/复述性价比高

SYS = (
    "你是上海迪士尼客服问答的语料改写员。下面会给你一组【书面问句】，请为每一条改写成"
    "【真实游客会说的口语化/生活化问法】。要求：\n"
    "1. 只改口吻，不改意图与答案——保留原问题要问的那件事、要的那个信息；\n"
    "2. 口语化手段任选：加语气词（啊/呢/嘛/哎）、省略主语、用代词指代、更啰嗦随意、"
    "加感叹或困惑语气、用大白话替代书面词（如'多少钱'→'得花多少''贵不贵'）；\n"
    "3. 不要凭空加前提（如虚构日期/人数），除非原句已有；\n"
    "4. 每条只输出一个口语化版本，不要解释。\n"
    "严格按 JSON 数组返回，元素顺序与输入一一对应，格式："
    "[{\"idx\":0,\"q\":\"口语化改写\"} ...]"
)

client = OpenAI(api_key=REWRITE_API_KEY, base_url=AGICTO_BASE_URL.rstrip("/"), timeout=40)


def _parse_variants(content):
    """把模型返回解析为 {idx: q}。模型被要求返回顶层 JSON 数组，故可能是 list；
    也兼容 {"items":[...]} 包裹。解析与 API 分离，解析失败绝不白打 API。"""
    c = content.strip()
    if c.startswith("```"):                       # 剥离 ```json 围栏
        c = c.strip("`")
        c = c[c.find("[") if "[" in c else 0:]
    try:
        data = json.loads(c)
    except Exception:
        s, e = c.find("["), c.rfind("]")
        data = json.loads(c[s:e + 1]) if s != -1 and e != -1 else []
    if isinstance(data, dict):                     # 兼容 {"items": [...]} 包裹
        data = data.get("items") or data.get("result") or []
    out = {}
    if isinstance(data, list):
        for el in data:
            if isinstance(el, dict) and "idx" in el:
                out[int(el["idx"])] = str(el.get("q", "")).strip()
    return out


def gen_batch(batch):
    """batch: list[(idx, q)] -> dict[idx]->colloquial_q。
    API 异常才重试；解析失败不再重打 API（同样输入会同样解析失败）。"""
    payload = [{"idx": i, "q": q} for i, q in batch]
    text = json.dumps(payload, ensure_ascii=False)
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": text}],
                temperature=0.9,
            )
            content = r.choices[0].message.content.strip()
        except Exception as e:
            print(f"    [batch API 重试 {attempt + 1}] {str(e)[:80]}")
            time.sleep(2 ** attempt)
            continue
        out = _parse_variants(content)
        if out:
            return out
        print(f"    [batch 解析失败 {attempt + 1}] 无有效元素: {content[:60]!r}")
        time.sleep(1)
    # 整批失败：降级保留原句（保证不丢失样本）
    return {i: q for i, q in batch}


def main():
    qa = json.load(open(QA_FILE, encoding="utf-8"))["items"]
    print(f"载入 QA 金标: {len(qa)} 对")

    out_items = []
    n_batches = (len(qa) + BATCH - 1) // BATCH
    ok = 0
    for b in range(n_batches):
        seg = qa[b * BATCH:(b + 1) * BATCH]
        res = gen_batch([(i, it["query"]) for i, it in enumerate(seg, b * BATCH)])
        batch_ok = 0
        for i, it in enumerate(seg, b * BATCH):
            cq = res.get(i)
            if not cq or cq == it["query"]:
                cq = it["query"]  # 降级：原句
            else:
                ok += 1
                batch_ok += 1
            out_items.append({
                "query": cq,
                "gold_idx": it.get("gold_idx"),
                "chunk_hash": it.get("chunk_hash"),
                "gold_answer": it.get("gold_answer"),
                "source": it.get("source"),
                "heading": it.get("heading"),
                "style": "colloquial",
                "parent_query": it["query"],
            })
        print(f"  批次 {b + 1}/{n_batches} 完成（本批口语化 {batch_ok}/{len(seg)}，累计 {ok}/{len(out_items)}）")

    doc = {
        "__qa_meta__": {
            "schema_ver": "chunk_hash_v1",
            "style": "colloquial_supplement",
            "parent_file": "eval_qa_v2.json",
            "ts": int(time.time()),
            "n_parent": len(qa),
            "n_colloquial": len(out_items),
        },
        "items": out_items,
    }
    tmp = COLL_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    os.replace(tmp, COLL_FILE)
    print(f"\n已落盘: {os.path.relpath(COLL_FILE, HERE)}（{ok}/{len(out_items)} 条成功口语化，其余降级原句）")


if __name__ == "__main__":
    main()

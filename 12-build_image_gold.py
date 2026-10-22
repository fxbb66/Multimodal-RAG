# -*- coding: utf-8 -*-
"""
S6 图片金标构建（2026-10-20）
============================
知识库 9 张图片（邮轮价格表）此前零金标，图片召回质量完全不可观测。
本脚本用 VL 模型（qwen3.5-omni-flash，与索引 caption 同模型）为每张图生成
3 个"必须看图才能回答"的 QA 对，OCR 文字随 prompt 注入以保证数字事实准确。

金标 schema（image_gold_v1）：
  query            用户口语化问题（依赖图片具体内容）
  gold_chunk_hash  图片块在索引中的 chunk_hash（与文本 QA 同指纹体系）
  answer           标准答案（以 OCR 数字为准）
  source           图片所属文件
落盘：output/eval_image_gold.json（幂等：已有 query 不重复生成，--force 重建）

用法：
    python 12-build_image_gold.py [--force]
"""
import base64
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))

from settings import VL_API_KEY, AGICTO_BASE_URL, VL_CAPTION_MODEL  # noqa: E402
from openai import OpenAI  # noqa: E402

KB_ROOT = os.path.normpath(os.path.join(HERE, "..", "迪士尼RAG知识库（完整）"))
META_FILE = os.path.join(HERE, "output", "disney_full_metadata.json")
IMG_TEXTS = os.path.join(HERE, "output", "image_texts_qwen3_5_omni_flash.json")
OUT_FILE = os.path.join(HERE, "output", "eval_image_gold.json")

_PROMPT = """这是上海迪士尼度假区知识库中的一张图片，附带有 OCR 识别的图片文字。
请基于图片内容生成 3 个用户可能会提问的问题及标准答案。

要求：
1. 问题必须依赖这张图片的具体内容才能回答（不看图就答不出）；
2. 问题用中文口语，5-25 字，覆盖不同角度（具体价格、房型/档位差异、适用日期或条件）；
3. 答案必须与 OCR 文字一致，数字以 OCR 为准，25 字以内；
4. 不要生成图片回答不了的泛化问题（如"迪士尼怎么样"）。

只输出 JSON 数组，不要解释：[{"query": "...", "answer": "..."}, ...]

【图片文件】{name}
【OCR 文字】
{ocr}
"""


def _extract_json_array(text):
    import re
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.MULTILINE)
    i, j = t.find("["), t.rfind("]")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(t[i:j + 1])
    except Exception:
        return None


def gen_qa_for_image(cli, path, name, ocr):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(path)[1].lower().lstrip(".") or "jpeg"
    if ext == "jpg":
        ext = "jpeg"
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/{ext};base64,{b64}"}},
            {"type": "text",
             "text": _PROMPT.replace("{name}", name).replace("{ocr}", ocr[:1500] or "（无）")},
        ],
    }]
    last_err = None
    for attempt in range(3):
        try:
            resp = cli.chat.completions.create(
                model=VL_CAPTION_MODEL, messages=messages, max_tokens=500,
                timeout=60)
            arr = _extract_json_array(resp.choices[0].message.content or "")
            if arr:
                return arr
            last_err = "JSON 解析失败: " + (resp.choices[0].message.content or "")[:80]
        except Exception as e:
            last_err = str(e)[:120]
        time.sleep(2 ** attempt)
    raise RuntimeError(f"VL QA 生成失败 {name}: {last_err}")


def main():
    force = "--force" in sys.argv
    meta = json.load(open(META_FILE, encoding="utf-8"))
    img_texts = json.load(open(IMG_TEXTS, encoding="utf-8")) \
        if os.path.exists(IMG_TEXTS) else {}
    img_blocks = [m for m in meta if m.get("type") == "image"]
    print(f"索引图片块 {len(img_blocks)} 张")

    existing = set()
    items = []
    if os.path.exists(OUT_FILE) and not force:
        old = json.load(open(OUT_FILE, encoding="utf-8"))
        existing = {it["query"] for it in old.get("items", [])}
        items = list(old.get("items", []))  # 携带旧条目：非 force 重跑不得丢金标
        print(f"已有金标 {len(existing)} 条（--force 可重建）")

    cli = OpenAI(api_key=VL_API_KEY, base_url=AGICTO_BASE_URL, timeout=90)
    items, t0 = [], time.time()
    for m in img_blocks:
        rel = m.get("rel_path", "")
        path = os.path.join(KB_ROOT, rel)
        if not os.path.exists(path):
            print(f"  [跳过] 原图不存在: {rel}")
            continue
        rec = img_texts.get(rel, {})
        ocr = rec.get("ocr", "") if isinstance(rec, dict) else ""
        try:
            pairs = gen_qa_for_image(cli, path, os.path.basename(rel), ocr)
        except Exception as e:
            print(f"  [失败] {rel}: {e}")
            continue
        n_new = 0
        for p in pairs:
            q = (p.get("query") or "").strip()
            a = (p.get("answer") or "").strip()
            if not q or not a or q in existing:
                continue
            existing.add(q)
            items.append({"query": q, "answer": a,
                          "gold_chunk_hash": m.get("chunk_hash", ""),
                          "source": m.get("source", ""), "rel_path": rel,
                          "gen": "vlm", "model": VL_CAPTION_MODEL})
            n_new += 1
        print(f"  {os.path.basename(rel)[:30]:32s} +{n_new} 对")

    out = {"__meta__": {"schema_ver": "image_gold_v1", "ts": int(time.time()),
                        "model": VL_CAPTION_MODEL, "n_images": len(img_blocks),
                        "note": "VL 模型基于图片+OCR 生成；gold 绑图片块 chunk_hash"},
           "items": items}
    tmp = OUT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUT_FILE)
    print(f"\n图片金标 {len(items)} 对已落盘（{time.time()-t0:.0f}s）: "
          f"{os.path.relpath(OUT_FILE, HERE)}")


if __name__ == "__main__":
    main()

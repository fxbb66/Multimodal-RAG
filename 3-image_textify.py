# -*- coding: utf-8 -*-
"""
M1 图片文本化：OCR + VLM Caption
================================
对知识库中的 9 张邮轮价格图片：
  1. RapidOCR 识别文字（价格表）
  2. qwen3.5-omni-flash 生成图片描述 caption
结果按 caption 模型名分文件落盘（image_texts_<model>.json，含 __cache_meta__ 指纹），
供 build_index 合并到图片 metadata 的 content 字段，使图片可被文字检索命中（BM25 与向量均可）。
首次运行若检测到无指纹旧版 image_texts.json，自动迁移（不重复消耗 OCR/VL 调用）。
"""
import os
import re
import sys
import json
import base64
import time
import math
from collections import OrderedDict, Counter
import numpy as np
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))
from settings import (VL_API_KEY, AGICTO_BASE_URL, VL_CAPTION_MODEL,  # noqa: E402
                      IMAGE_TEXTS_FILE, LEGACY_IMAGE_TEXTS_FILE)
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location("chunk_module", os.path.join(HERE, "1-parse_and_chunk.py"))
_cm = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cm)
update_build_report = _cm.update_build_report
file_md5 = _cm.file_md5

KB_ROOT = os.path.normpath(os.path.join(HERE, "..", "迪士尼RAG知识库（完整）"))
OUT_DIR = os.path.join(HERE, "output")
OUT_FILE = IMAGE_TEXTS_FILE  # N2: 模型指纹文件名
_META_KEY = "__cache_meta__"

# N5: OCR 引擎进程内单例（原现在循环内每图新建，模型反复加载极慢）
_OCR_ENGINE = None


def get_ocr_engine():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def ocr_image(path):
    result, _ = get_ocr_engine()(path)
    if not result:
        return ""
    return "\n".join(item[1] for item in result if len(item) > 1)


def vlm_caption(path):
    """用 qwen3.5-omni-flash 生成图片中文描述（OpenAI 多模态接口）。"""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}},
            {"type": "text", "text": "请用中文简要描述这张图片的内容，重点说明它是什么（如价格表、海报等）以及包含的关键信息。100字以内。"}
        ]
    }]
    last_err = None
    _vl_cli = OpenAI(api_key=VL_API_KEY, base_url=AGICTO_BASE_URL)
    for attempt in range(3):
        try:
            resp = _vl_cli.chat.completions.create(
                model=VL_CAPTION_MODEL, messages=messages)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = str(e)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"vlm caption failed: {last_err}")


def _cache_meta():
    return {"caption_model": VL_CAPTION_MODEL, "ts": int(time.time())}


def _write_cache(cache):
    """落盘（带模型指纹；先 tmp 再 replace 防中断写花）。"""
    payload = dict(cache)
    payload[_META_KEY] = _cache_meta()
    tmp = OUT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUT_FILE)


# ---------------------------------------------------------------- Caption 质量信号（纯规则，不调模型）

def _qa_tokens(text):
    """零依赖分词：CJK 单字 + 连续字母/数字串。token_count 与 TF-IDF 共用。"""
    return re.findall(r"[一-鿿]|[A-Za-z0-9]+", text or "")


def _grounded_keyword_groups(ocr):
    """从单图 OCR 抽取"可核验词组"，caption 命中它们即说明没偏题。
    返回该图 OCR 中实际出现的组名集合。"""
    t = ocr or ""
    groups = set()
    if re.search(r"\d+\s*天\s*\d+\s*晚", t):
        groups.add("duration")                                   # 行程时长
    if re.search(r"20\d{2}\s*年\s*\d{1,2}\s*月", t):
        groups.add("date")                                     # 出发年月
    if "USD" in t or "$" in t:
        groups.add("currency")                                 # 币种
    if any(w in t for w in ("内舱", "海景", "阳台", "套房", "礼宾")):
        groups.add("room")                                     # 房型
    if "税" in t:
        groups.add("tax")                                      # 港务税费
    if "新加坡" in t:
        groups.add("location")                                 # 出发地
    return groups


def _caption_has_keywords(caption, groups):
    """caption 对 OCR 锚定词组的覆盖率。≥0.5 且至少命中 2 组 → True。
    OCR 抽不出 ≥2 组（无文字海报等）时退化为要求 caption ≥15 token。"""
    c = caption or ""
    if len(groups) < 2:
        return len(_qa_tokens(c)) >= 15
    hit = 0
    if "duration" in groups and re.search(r"\d+\s*天\s*\d+\s*晚", c):
        hit += 1
    if "date" in groups and re.search(r"20\d{2}\s*年(?:\d{1,2}\s*月)?", c):
        hit += 1
    if "currency" in groups and ("USD" in c or "美元" in c or "$" in c):
        hit += 1
    if "room" in groups and any(w in c for w in
                                ("内舱", "海景", "阳台", "套房", "礼宾", "房型")):
        hit += 1
    if "tax" in groups and "税" in c:
        hit += 1
    if "location" in groups and "新加坡" in c:
        hit += 1
    return hit / len(groups) >= 0.5 and hit >= 2


def _tfidf_cosine_matrix(texts):
    """字符二元组 TF-IDF 的 L2 归一化矩阵（纯 numpy，与 BM25 同一套 token 粒度）。
    返回行向量矩阵；行内积即 caption 间 TF-IDF 余弦。"""
    docs, df = [], Counter()
    for t in texts:
        s = re.sub(r"\s+", "", t or "")
        terms = [s[i:i + 2] for i in range(len(s) - 1)] if len(s) >= 2 else ([s] if s else [])
        docs.append(Counter(terms))
        df.update(set(terms))
    n = len(texts)
    terms = sorted(df)
    term_idx = {t: j for j, t in enumerate(terms)}
    mat = np.zeros((n, len(terms)))
    for i, c in enumerate(docs):
        for term, f in c.items():
            idf = math.log((n + 1) / (df[term] + 1)) + 1.0
            mat[i, term_idx[term]] = f * idf
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def compute_caption_quality(cache):
    """纯规则计算 caption 质量信号，返回 (per_image, summary)，零 API 调用。
    每张图：{is_empty, token_count, has_keywords, diversity_score}。
      token_count     : 区分"长度不足"
      has_keywords   : OCR 锚定关键词覆盖，区分"内容偏题"
      diversity_score: 1 - 与其他 caption 的平均 TF-IDF 余弦，低分=疑似重复 caption；
                       空 caption 为 null（不可比），仅有一条非空时为 1.0。"""
    rels = list(cache.keys())
    captions = [(cache[r].get("caption") or "").strip() for r in rels]
    nonempty = [i for i, c in enumerate(captions) if c]
    cosine = _tfidf_cosine_matrix(captions) if len(nonempty) >= 2 else None

    per = OrderedDict()
    for i, rel in enumerate(rels):
        cap, ocr = captions[i], cache[rel].get("ocr") or ""
        empty = not cap
        if empty:
            diversity = None
        elif cosine is None:
            diversity = 1.0
        else:
            others = [j for j in nonempty if j != i]
            diversity = 1.0 - float(np.mean([cosine[i][j] for j in others])) \
                if others else 1.0
        per[rel] = {
            "is_empty": empty,
            "token_count": len(_qa_tokens(cap)),
            "has_keywords": False if empty else _caption_has_keywords(
                cap, _grounded_keyword_groups(ocr)),
            "diversity_score": round(diversity, 3) if diversity is not None else None,
        }

    tokens = [v["token_count"] for v in per.values()]
    divs = [v["diversity_score"] for v in per.values()
            if v["diversity_score"] is not None]
    summary = {
        "images_n": len(per),
        "empty_n": sum(v["is_empty"] for v in per.values()),
        "token_count": {"min": min(tokens), "mean": round(sum(tokens) / len(tokens), 1),
                        "max": max(tokens)},
        "has_keywords_n": sum(v["has_keywords"] for v in per.values()),
        "diversity_score": {"min": min(divs), "mean": round(sum(divs) / len(divs), 3)}
        if divs else None,
    }
    return per, summary


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cache = {}
    migrated = False
    if os.path.exists(OUT_FILE):
        try:
            raw = json.load(open(OUT_FILE, encoding="utf-8"))
            meta = raw.get(_META_KEY, {})
            if meta.get("caption_model") != VL_CAPTION_MODEL:
                print(f"[缓存失效] caption 模型 {meta.get('caption_model')}→{VL_CAPTION_MODEL}，全量重跑")
            else:
                cache = {k: v for k, v in raw.items() if k != _META_KEY}
        except Exception:
            cache = {}
    elif os.path.exists(LEGACY_IMAGE_TEXTS_FILE):
        # N2: 无指纹旧版自动迁移（内容由同一 VL_CAPTION_MODEL 产出则直接复用）
        try:
            cache = json.load(open(LEGACY_IMAGE_TEXTS_FILE, encoding="utf-8"))
            print(f"[迁移] 检测到无指纹旧缓存，迁移 {len(cache)} 条到 {os.path.basename(OUT_FILE)}")
            migrated = True
        except Exception:
            cache = {}

    images = []
    for dp, _, files in os.walk(KB_ROOT):
        for fn in sorted(files):
            if fn.lower().endswith((".jpeg", ".jpg", ".png")):
                fp = os.path.join(dp, fn)
                images.append((os.path.relpath(fp, KB_ROOT), fp))

    for rel, fp in images:
        if rel in cache:
            print(f"跳过(已缓存): {rel}")
            continue
        print(f"处理: {rel}")
        ocr = ""
        try:
            ocr = ocr_image(fp)
            print(f"  OCR: {len(ocr)} 字")
        except Exception as e:
            print(f"  OCR失败: {e}")
        caption = ""
        try:
            caption = vlm_caption(fp)
            print(f"  Caption: {caption[:60]}")
        except Exception as e:
            print(f"  Caption失败: {e}")
        cache[rel] = {"ocr": ocr, "caption": caption}
        _write_cache(cache)

    _write_cache(cache)  # 确保指纹写入（全缓存命中/迁移场景）
    if migrated and os.path.exists(LEGACY_IMAGE_TEXTS_FILE):
        os.remove(LEGACY_IMAGE_TEXTS_FILE)
        print("已删除无指纹旧缓存 image_texts.json")

    print(f"\n完成，共 {len(cache)} 张图片，结果: {os.path.relpath(OUT_FILE, HERE)}")

    # 机器可读构建报告（stage3）：覆盖度 + OCR/caption 空结果作为质量信号
    n_ocr_empty = sum(1 for v in cache.values() if not v.get("ocr"))
    n_cap_empty = sum(1 for v in cache.values() if not v.get("caption"))
    ocr_ver = "unknown"
    try:
        from importlib.metadata import version
        ocr_ver = version("rapidocr_onnxruntime")
    except Exception:
        pass
    # caption 结构化质量信号（纯规则，零 API）：区分长度不足/偏题/重复，替代单纯空值布尔
    cap_quality, cap_quality_summary = compute_caption_quality(cache)
    update_build_report("stage3_image_textify", {
        "images_n": len(cache), "images_expected": len(images),
        "ocr_empty_n": n_ocr_empty, "caption_empty_n": n_cap_empty,
        "caption_model": VL_CAPTION_MODEL, "ocr_engine_ver": ocr_ver,
        "file_md5": file_md5(OUT_FILE),
        "caption_quality": cap_quality,
        "caption_quality_summary": cap_quality_summary,
        "ok": len(cache) == len(images) and n_cap_empty == 0,
    })


if __name__ == "__main__":
    main()

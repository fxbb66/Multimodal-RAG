# -*- coding: utf-8 -*-
"""
迪士尼完整知识库 - Embedding 与 FAISS 入库
==========================================
读取 output/chunks.jsonl（全部去重文本 chunk）+ 知识库中的 9 张图片，
调用 tongyi-embedding-vision-plus 生成多模态向量，构建 IndexFlatIP（R6：归一化后内积=余弦相似度）。

输出（FAISS 经 ASCII 临时路径写入后移动到位，规避 Windows fopen 不支持中文路径的问题）：
  output/disney_full_index.faiss
  output/disney_full_metadata.json
  output/embeddings.npy          文本向量断点缓存
特性：
- 文本批量 embedding（每批 8 条），失败自动重试并降级为逐条
- 断点续传：已完成的文本向量缓存在 embeddings.npy，重跑不重复计费
- metadata 与现有查询/前端字段兼容：id/type/source/content/path/sources
"""
import os
import sys
import json
import time
import pickle
import base64
import numpy as np
from openai import OpenAI

# 引入同项目 lib
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "lib"))
from retriever import BM25Index  # noqa: E402
from settings import (EMBEDDING_API_KEY, AGICTO_BASE_URL, EMBEDDING_MODEL,  # noqa: E402
                      IMAGE_TEXTS_FILE, LEGACY_IMAGE_TEXTS_FILE, EMBEDDING_DIM)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("chunk_module", os.path.join(_HERE, "1-parse_and_chunk.py"))
_cm = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cm)
extract_tags = _cm.extract_tags
image_chunk_hash = _cm.image_chunk_hash
update_build_report = _cm.update_build_report
file_md5 = _cm.file_md5
BUILD_REPORT_FILE = _cm.BUILD_REPORT_FILE

MODEL = EMBEDDING_MODEL
BATCH_SIZE = 8
MAX_RETRY = 3

HERE = os.path.dirname(os.path.abspath(__file__))
KB_ROOT = os.path.normpath(os.path.join(HERE, "..", "迪士尼RAG知识库（完整）"))
OUT_DIR = os.path.join(HERE, "output")
CHUNKS_FILE = os.path.join(OUT_DIR, "chunks.jsonl")
INDEX_FILE = os.path.join(OUT_DIR, "disney_full_index.faiss")
METADATA_FILE = os.path.join(OUT_DIR, "disney_full_metadata.json")
EMB_CACHE = os.path.join(OUT_DIR, "embeddings.npy")


import requests as _requests

_emb_session = _requests.Session()
_emb_headers = {"Authorization": f"Bearer {EMBEDDING_API_KEY}",
                "Content-Type": "application/json"}
_EMB_URL = f"{AGICTO_BASE_URL.rstrip('/')}/embeddings"


def call_with_retry(items, label=""):
    """调用 tongyi-embedding-vision-plus（百炼原生 contents 格式）。
    items: List[dict]，每项 {"text": ...} 或 {"image": data_url}。
    该模型为独立向量：每个 content 返回一个 1152 维向量；文本与图片同空间可直接比 cosine。
    指数退避重试；批量失败降级逐条。"""
    contents = []
    for it in items:
        if "text" in it:
            contents.append({"text": it["text"]})
        elif "image" in it:
            contents.append({"image": it["image"]})
        else:
            contents.append({"text": str(it)})
    body = {"model": MODEL, "input": {"contents": contents},
            "encoding_format": "float"}
    last_err = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            r = _emb_session.post(_EMB_URL, headers=_emb_headers,
                                  json=body, timeout=120)
            r.raise_for_status()
            data = r.json()
            embs = data.get("output", {}).get("embeddings") or data.get("data")
            if not embs:
                raise RuntimeError(f"无 embedding 数据: {str(data)[:200]}")
            return [e["embedding"] for e in embs]
        except Exception as e:
            last_err = str(e)
        wait = 2 ** attempt
        print(f"    [重试 {attempt}/{MAX_RETRY}] {label} {last_err[:80]}，{wait}s 后重试")
        time.sleep(wait)
    # 批量整体失败 -> 逐条降级
    if len(items) > 1:
        print(f"    [降级] {label} 批量失败，改为逐条调用")
        out = []
        for i, it in enumerate(items):
            out.append(call_with_retry([it], label=f"{label}#{i}")[0])
        return out
    raise RuntimeError(f"Embedding 彻底失败 {label}: {last_err}")


_EMB_META_KEY = "__cache_meta__"


def _emb_cache_meta():
    return {"model": MODEL, "dim": EMBEDDING_DIM}


def _save_emb_cache(chunks, vectors):
    """断点缓存落盘，带模型/维度指纹（换模型时加载侧自动判废，防跨模型向量污染）。"""
    cache = {"hashes": [c["hash"] for c in chunks if c["hash"] in vectors],
             "vectors": [vectors[c["hash"]] for c in chunks if c["hash"] in vectors],
             _EMB_META_KEY: _emb_cache_meta()}
    np.save(EMB_CACHE, cache)


def embed_texts(chunks):
    """对全部文本 chunk 生成向量，支持断点续传。
    缓存头 __cache_meta__={model,dim}：模型或维度不符整份丢弃重嵌入；
    历史无指纹缓存：维度与当前配置一致时信任并立即回写补指纹（零 API 成本回填）。"""
    hashes = [c["hash"] for c in chunks]
    vectors = {}
    backfill_meta = False
    if os.path.exists(EMB_CACHE):
        old = np.load(EMB_CACHE, allow_pickle=True).tolist()
        if isinstance(old, dict):
            meta = old.get(_EMB_META_KEY)
            if meta is not None:
                if meta.get("model") != MODEL or meta.get("dim") != EMBEDDING_DIM:
                    print(f"[断点缓存判废] 指纹 model={meta.get('model')}/dim={meta.get('dim')} "
                          f"!= 当前 {MODEL}/{EMBEDDING_DIM}，全部重新嵌入")
                else:
                    for h, v in zip(old.get("hashes", []), old.get("vectors", [])):
                        vectors[h] = v
            else:
                # 无指纹旧缓存：校验现存向量维度，一致则信任并标记回填
                old_vecs = old.get("vectors", [])
                old_dim = len(old_vecs[0]) if len(old_vecs) else None
                if old_dim == EMBEDDING_DIM:
                    for h, v in zip(old.get("hashes", []), old_vecs):
                        vectors[h] = v
                    backfill_meta = True
                    print(f"[断点缓存无指纹] 维度 {old_dim} 与当前一致，信任并回写模型指纹")
                else:
                    print(f"[断点缓存判废] 无指纹且维度 {old_dim} != {EMBEDDING_DIM}，重新嵌入")
        done = sum(1 for h in hashes if h in vectors)
        print(f"断点缓存命中 {done}/{len(hashes)}")

    pending = [(i, c) for i, c in enumerate(chunks) if c["hash"] not in vectors]
    if pending:
        print(f"待 embedding: {len(pending)} 条，批次大小 {BATCH_SIZE}")
        for bstart in range(0, len(pending), BATCH_SIZE):
            batch = pending[bstart:bstart + BATCH_SIZE]
            texts = [c["content"] for _, c in batch]
            embs = call_with_retry([{"text": t} for t in texts],
                                   label=f"文本{bstart + 1}-{bstart + len(batch)}")
            for (_, c), v in zip(batch, embs):
                vectors[c["hash"]] = v
            # 每批落盘一次断点（带指纹）
            _save_emb_cache(chunks, vectors)
            print(f"  进度 {min(bstart + BATCH_SIZE, len(pending))}/{len(pending)}")
    elif backfill_meta:
        _save_emb_cache(chunks, vectors)  # 全命中但缺指纹：零调用回写
    return [vectors[h] for h in hashes]


def collect_images():
    images = []
    for dirpath, _, files in os.walk(KB_ROOT):
        for fn in sorted(files):
            if fn.lower().endswith((".jpeg", ".jpg", ".png")):
                fp = os.path.join(dirpath, fn)
                images.append((os.path.relpath(fp, KB_ROOT), fp))
    return images


def embed_image(path, text=None):
    """多模态 embedding：直接用图片视觉向量（1152 维）。
    OCR+caption 文本保留在 metadata.content 供 BM25 命中，向量侧走跨模态视觉向量，
    文字查询（如"邮轮价格"）也能通过语义空间匹配到图片。"""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    return call_with_retry([{"image": f"data:image/{ext};base64,{b64}"}],
                           label=f"图片 {os.path.basename(path)}")[0]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)  # FAISS 写入走 ASCII 临时路径，无需 os.chdir

    chunks = [json.loads(line) for line in open(CHUNKS_FILE, encoding="utf-8")]
    print(f"载入文本 chunk: {len(chunks)}")

    # 长度探针：先 embed 最长 chunk，及早暴露输入长度限制
    longest = max(chunks, key=lambda c: c["body_chars"])
    print(f"长度探针: {longest['body_chars']} 字 -> 调用模型验证 ...")
    call_with_retry([{"text": longest["content"]}], label="长度探针")
    print("  探针通过")

    text_vectors = embed_texts(chunks)

    print("\n处理图片 ...")
    # M1: 加载 OCR + caption 文本
    img_texts = {}
    # N2: 优先模型指纹文件，回退无指纹旧版（迁移期兼容）
    img_texts_path = IMAGE_TEXTS_FILE if os.path.exists(IMAGE_TEXTS_FILE) else LEGACY_IMAGE_TEXTS_FILE
    if os.path.exists(img_texts_path):
        raw = json.load(open(img_texts_path, encoding="utf-8"))
        img_texts = {k: v for k, v in raw.items() if k != "__cache_meta__"}
        print(f"图片文本缓存: {os.path.basename(img_texts_path)} ({len(img_texts)} 张)")
    images = collect_images()
    image_meta, image_vectors = [], []
    for rel, fp in images:
        print(f"  - {rel}")
        info = img_texts.get(rel, {})
        ocr = info.get("ocr", "")
        cap = info.get("caption", "")
        text_content = f"{cap}\n{ocr}".strip() if (ocr or cap) else f"[图片] {os.path.basename(rel)}"
        image_vectors.append(embed_image(fp, text=text_content))
        image_meta.append({
            "type": "image",
            "chunk_hash": image_chunk_hash(rel),
            "source": f"图片: {os.path.basename(rel)}",
            "content": text_content,     # M1: OCR+caption 文本，可被 BM25/向量检索
            "path": fp,
            "rel_path": rel,
            "category": rel.split(os.sep)[0],
            "tags": extract_tags(os.path.basename(rel), rel.split(os.sep)[0], cap),
        })

    # 组装 metadata（与向量顺序严格对齐）
    metadata = []
    for c in chunks:
        src0 = c["sources"][0]
        metadata.append({
            "id": c["id"],
            "type": "text",
            "chunk_hash": c["hash"],   # 内容指纹：评估 QA 尺子按它 remap，与索引位置解耦
            "source": src0["file"],
            "content": c["content"],
            "parent_context": c.get("parent_context", c["content"]),  # I1
            "category": src0["category"],
            "heading": src0.get("heading", ""),
            "tags": c.get("tags", []),                                # M2
            "sources": c["sources"],
        })
    for i, m in enumerate(image_meta):
        m["id"] = len(metadata)
        metadata.append(m)

    # R6: 向量 L2 归一化，使用 IndexFlatIP（内积 = 余弦相似度）
    all_vectors = np.array(text_vectors + image_vectors, dtype="float32")
    dim = all_vectors.shape[1]
    # N6: 维度与 settings.EMBEDDING_DIM 强校验，防止换模型后混库
    if dim != EMBEDDING_DIM:
        raise SystemExit(
            f"[维度不匹配] 实得 {dim} 维 != settings.EMBEDDING_DIM={EMBEDDING_DIM}；"
            f"请确认 EMBEDDING_MODEL 与 EMBEDDING_DIM 一致后删除 embeddings.npy/*.faiss 重建")
    norms = np.linalg.norm(all_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    all_vectors = all_vectors / norms
    print(f"\n向量维度: {dim}, 总向量数: {len(all_vectors)} (文本 {len(text_vectors)} + 图片 {len(image_vectors)})")

    import faiss
    index = faiss.IndexFlatIP(dim)  # R6: 归一化后内积=余弦相似度
    index.add(all_vectors)
    # FAISS fopen 不支持中文路径 -> 先写 ASCII 临时路径再移动到 output/
    import shutil
    import tempfile
    _tmp_idx = os.path.join(tempfile.gettempdir(), "disney_rag_build", "disney_full_index.faiss")
    os.makedirs(os.path.dirname(_tmp_idx), exist_ok=True)
    faiss.write_index(index, _tmp_idx)
    shutil.move(_tmp_idx, INDEX_FILE)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # R1: 构建 BM25 索引（字符级，文本 + 图片 OCR 文本，供混合检索）
    texts_for_bm25 = [m["content"] for m in metadata]
    bm25 = BM25Index().fit(texts_for_bm25)
    bm25_path = os.path.join(OUT_DIR, "bm25_index.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25 索引已保存: {bm25_path}")

    print(f"索引已保存: {os.path.relpath(INDEX_FILE, HERE)}")
    print(f"元数据已保存: {os.path.relpath(METADATA_FILE, HERE)}")
    print(f"索引记录数: {index.ntotal}")

    write_stage2_report(index, metadata, chunks, images, dim, bm25_path)


def write_stage2_report(index, metadata, chunks, images, dim, bm25_path):
    """stage2 机器可读构建报告 + overall 一致性裁决。
    独立于构建过程：重建后由 main 调用；对历史已建好的索引可离线补生成，
    检查逻辑唯一来源（禁止另写一套口径）。"""
    n_text = sum(1 for m in metadata if m["type"] == "text")
    n_image = sum(1 for m in metadata if m["type"] == "image")
    hashes_all = [m.get("chunk_hash") for m in metadata]
    stage1_md5 = None
    if os.path.exists(BUILD_REPORT_FILE):
        try:
            stage1_md5 = (json.load(open(BUILD_REPORT_FILE, encoding="utf-8"))
                          .get("stages", {}).get("stage1_parse", {}).get("chunks_md5"))
        except Exception:
            stage1_md5 = None
    checks = {
        "dim_match_settings": dim == EMBEDDING_DIM,
        "index_count_eq_metadata": index.ntotal == len(metadata),
        "total_eq_text_plus_image": len(metadata) == n_text + n_image,
        "text_count_eq_chunks": n_text == len(chunks),
        "image_count_eq_discovered": n_image == len(images),
        "chunk_hash_full_coverage": all(hashes_all),
        "chunk_hash_unique": len(set(hashes_all)) == len(hashes_all),
        "bm25_built": os.path.exists(bm25_path),
        # None=尚无 stage1 报告可比对（未验证，不判失败，由 warnings 提示）
        "chunks_file_unchanged_since_stage1":
            (file_md5(CHUNKS_FILE) == stage1_md5) if stage1_md5 else None,
    }
    warnings = []
    prev = {}
    if os.path.exists(BUILD_REPORT_FILE):
        try:
            prev = json.load(open(BUILD_REPORT_FILE, encoding="utf-8")).get("stages", {})
        except Exception:
            prev = {}
    if not prev.get("stage1_parse"):
        warnings.append("stage1_parse 报告缺失（chunks.jsonl 可能非本流水线最新产出）")
    elif not prev["stage1_parse"].get("ok"):
        warnings.append("stage1_parse 报告 ok=False（存在解析失败文档）")
    if not prev.get("stage3_image_textify"):
        warnings.append("stage3_image_textify 报告缺失（图片 OCR/caption 未运行）")
    elif prev["stage3_image_textify"].get("images_n") != n_image:
        warnings.append("stage3 图片文本数与入库图片数不一致")
    index_md5 = file_md5(INDEX_FILE)
    stage2 = {
        "index_md5": index_md5, "index_total": index.ntotal, "index_dim": dim,
        "embedding_model": EMBEDDING_MODEL,
        "n_text": n_text, "n_image": n_image,
        "metadata_md5": file_md5(METADATA_FILE),
        "bm25_md5": file_md5(bm25_path),
        "checks": checks, "warnings": warnings,
        "ok": all(v is True for v in checks.values()),  # None=未验证，不否决
    }
    update_build_report("stage2_index", stage2)
    # overall：最新 stage2 裁决即整体完成度（1/3 阶段为依赖项，异常进 warnings）
    doc = json.load(open(BUILD_REPORT_FILE, encoding="utf-8"))
    doc["overall"] = {"ok": stage2["ok"] and not warnings,
                      "index_md5": index_md5, "index_total": index.ntotal,
                      "index_dim": dim, "embedding_model": EMBEDDING_MODEL,
                      "checks": checks, "warnings": warnings}
    tmp = BUILD_REPORT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    os.replace(tmp, BUILD_REPORT_FILE)
    print(f"构建报告: {os.path.relpath(BUILD_REPORT_FILE, HERE)} "
          f"overall_ok={doc['overall']['ok']}" + (f" warnings={warnings}" if warnings else ""))
    return doc["overall"]


if __name__ == "__main__":
    main()

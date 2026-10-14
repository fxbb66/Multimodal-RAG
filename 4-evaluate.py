# -*- coding: utf-8 -*-
"""
E1 评估体系 v2：从现有 chunk 反向生成 QA 对，量化检索与回答质量
================================================
无需外部资料：用 LLM 从知识库 chunk 生成问答对（含金标答案），作为 ground truth。
检索侧指标：Recall@k（严格=gold 命中 / 宽松=同源同章节命中）、MRR
答案侧指标（3 个检索档 × n 题，每题真实生成一次）：
  evidence_coverage 证据覆盖率 / answer_hit 无条件答案命中 /
  answer_hit_given_evidence 证据到位时命中（生成侧真实水平）/ gen_loss_when_evidence
对比维度：
  - 纯向量 vs 混合检索（BM25+向量 RRF）vs 混合+Rerank
  - rerank 生效核验：RERANK_STATS（calls/fallbacks），降级即暴露
性能：
  - 检索/生成均线程池并发（EVAL_WORKERS，默认 8）
  - 生成答案按 (模型,prompt版本,问题,上下文) 哈希落盘 eval_ans_cache.json，
    跨档相同上下文只生成一次；代码/索引/问题未变时重跑零 LLM 调用
注意：QA 条目携带 chunk_hash（= 切片阶段 sha1(去空白正文)，图片按相对路径）。
gold_idx 仅在生成时记录，每次加载按 chunk_hash 重新映射到当前索引位置——
索引重建/重排后尺子免费复用，仅当金标块在新库中消失（内容被删改）时才需剔除/重生成。
三档"线上同源"口径（2026-10-02 起）：
  - 答案档 prompt_style="online"：system/上下文编号 [n] 与 frontend/app.py 逐字同源
    （ONLINE_SYSTEM_PROMPT + retriever.build_context），答案判定前剥离 [n] 引用标记
  - 检索档 use_rewrite=True：intent_rewrite 改写后 query 走向量/BM25/rerank，
    多意图走 search_multi，与线上 retrieve() 同分支；追问短路题记未命中并单列
  - 意图专用金标集 eval_intent_gold.json：人工静态标注（不调 LLM 生成、不绑索引），
    判 action 准确率 / 追问 P/R / 类型命中 / 多意图拆分 / 消解词保留
输出：output/eval_qa_v2.json + output/eval_results.json + 控制台汇总表
"""
import os
import re
import sys
import json
import time
import pickle
import hashlib
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "lib"))
from retriever import (HybridRetriever, expand_query_terms,  # noqa: E402
                       ascii_faiss_path, RERANK_STATS, intent_rewrite,
                       build_context)
from settings import (DASHSCOPE_API_KEY, EMBEDDING_API_KEY, GEN_API_KEY,  # noqa: E402
                      REWRITE_API_KEY,
                      AGICTO_BASE_URL, EMBEDDING_MODEL, EMBEDDING_DIM,
                      REWRITE_MODEL, GEN_MODEL, RERANK_TOPN,
                      MAX_PER_SOURCE, USE_MMR, MMR_LAMBDA, EVAL_WORKERS,
                      INTENT_ENABLED, INTENT_CONFIDENCE_MIN, INTENT_MAX_SUB_QUERIES,
                      ONLINE_SYSTEM_PROMPT)

OUT_DIR = os.path.join(_HERE, "output")
INDEX_FILE = os.path.join(OUT_DIR, "disney_full_index.faiss")
METADATA_FILE = os.path.join(OUT_DIR, "disney_full_metadata.json")
BM25_FILE = os.path.join(OUT_DIR, "bm25_index.pkl")
QA_FILE = os.path.join(OUT_DIR, "eval_qa_v2.json")
RESULT_FILE = os.path.join(OUT_DIR, "eval_results.json")
INTENT_GOLD_FILE = os.path.join(_HERE, "eval_intent_gold.json")  # 人工静态尺子（进版本库，非生成物）

if DASHSCOPE_API_KEY:
    try:
        import dashscope
        dashscope.api_key = DASHSCOPE_API_KEY  # 保留用于 rerank 降级兜底
    except ImportError:
        pass

ANSWER_EVAL_N = 80  # N2: 答案侧有效样本量（原 40 过小）
MAX_QA_PER_SOURCE = 2  # N2: 每来源文件最多抽取的 gold chunk 数，缓解同源偏差
# QA 尺子结构：chunk_hash 指纹 remap（2026-10-02 起 gold_idx 不再绑索引位置）；改 remap 规则必须 bump
QA_SCHEMA_VER = "chunk_hash_v1"
# 金标块在新索引中的失配率超过此值 → 知识库内容已大变，尺子判废重新生成（少量失配仅剔除）
QA_REBUILD_MISS_RATE = 0.5
# 检索管线版本：MMR 位次相关性 + rerank 空结果守卫（2026-10-01）；改检索/后处理逻辑必须 bump
PIPELINE_VER = "mmr-rankpos-v2"

# N11: 结构化日志（output/run.log + 控制台），关键节点可追溯
import logging
_RUN_LOG = os.path.join(OUT_DIR, "run.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(_RUN_LOG, encoding="utf-8"),
              logging.StreamHandler()],
)
log = logging.getLogger("rag_eval")

# 答案缓存命中/未命中计数（N11 可观测）
ANS_CACHE_HIT = {"hit": 0, "miss": 0}


_EMB_CACHE_FILE = os.path.join(OUT_DIR, "eval_emb_cache.npz")
_emb_disk = {}
# 缓存元信息键（不参与向量查询），用于换模型时自动失效
_CACHE_META_KEY = "__cache_meta__"
_CACHE_META = {"model": EMBEDDING_MODEL, "dim": EMBEDDING_DIM,
               "ts": int(__import__("time").time())}


def _load_emb_cache():
    """从磁盘加载 embedding 缓存，使检索侧评估不依赖 API 实时调用。
    缓存带模型指纹：模型/维度不匹配则丢弃旧缓存，避免跨模型向量污染。"""
    global _emb_disk
    if os.path.exists(_EMB_CACHE_FILE):
        d = np.load(_EMB_CACHE_FILE, allow_pickle=True)
        if _CACHE_META_KEY in d.files:
            meta = d[_CACHE_META_KEY].item()
            if meta.get("model") != _CACHE_META["model"] or meta.get("dim") != _CACHE_META["dim"]:
                print(f"[缓存失效] 模型 {meta.get('model')}→{_CACHE_META['model']}，丢弃旧缓存")
                _emb_disk = {}
                return
        _emb_disk = {k: d[k] for k in d.files if k != _CACHE_META_KEY}
        print(f"已加载 embedding 缓存: {len(_emb_disk)} 条 (model={_CACHE_META['model']})")


def _save_emb_cache():
    if _emb_disk:
        payload = dict(_emb_disk)
        payload[_CACHE_META_KEY] = np.array(_CACHE_META, dtype=object)
        np.savez(_EMB_CACHE_FILE, **payload)


def get_embedding(text):
    if text in _emb_disk:
        return _emb_disk[text].copy()
    import requests as _rq
    url = f"{AGICTO_BASE_URL.rstrip('/')}/embeddings"
    h = {"Authorization": f"Bearer {EMBEDDING_API_KEY}", "Content-Type": "application/json"}
    body = {"model": EMBEDDING_MODEL,
            "input": {"contents": [{"text": text}]},
            "encoding_format": "float"}
    resp = _rq.post(url, headers=h, json=body, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    embs = data.get("output", {}).get("embeddings") or data.get("data")
    if not embs:
        raise RuntimeError(f"Embedding API 返回空: {str(data)[:200]}")
    v = np.array(embs[0]["embedding"], dtype='float32')
    v = v / (np.linalg.norm(v) or 1.0)
    _emb_disk[text] = v.copy()
    return v.copy()


# ---------------------------------------------------------------- 意图改写磁盘缓存
# 改写结果（含置信不足/异常的降级结果）按 (模型,阈值,query,history) 落盘，
# "改写 query 检索档"/意图评测/意图金标三处共用，跨档跨运行零重复 LLM 调用、可复现。
REWRITE_CACHE_FILE = os.path.join(OUT_DIR, "eval_rewrite_cache.json")
REWRITE_CACHE_VER = "v2"  # v2(2026-10-02)：retriever 新增短问/三连问确定性规则门；改 intent prompt/规则时 bump
_rw_cache = {}
_rw_lock = threading.Lock()
RW_CACHE_HIT = {"hit": 0, "miss": 0}


def _load_rewrite_cache():
    global _rw_cache
    if not os.path.exists(REWRITE_CACHE_FILE):
        return
    try:
        with open(REWRITE_CACHE_FILE, encoding="utf-8") as f:
            raw = json.load(f)
        meta = raw.get("__cache_meta__") if isinstance(raw, dict) else None
        expect = {"rewrite_model": REWRITE_MODEL, "cache_ver": REWRITE_CACHE_VER,
                  "confidence_min": INTENT_CONFIDENCE_MIN,
                  "max_sub": INTENT_MAX_SUB_QUERIES}
        if not meta or any(meta.get(k) != v for k, v in expect.items()):
            print(f"[改写缓存口径不符，整份作废] cache={meta} expect={expect}")
            log.warning("rewrite cache invalidated: cache=%s expect=%s", meta, expect)
            _rw_cache = {}
            return
        _rw_cache = {k: v for k, v in raw.items() if k != "__cache_meta__"}
        print(f"已加载改写缓存: {len(_rw_cache)} 条 (model={REWRITE_MODEL})")
    except Exception as e:
        print(f"[改写缓存读取失败，重建] {e}")
        _rw_cache = {}


def _save_rewrite_cache():
    with _rw_lock:
        snapshot = dict(_rw_cache)
    snapshot["__cache_meta__"] = {
        "rewrite_model": REWRITE_MODEL, "cache_ver": REWRITE_CACHE_VER,
        "confidence_min": INTENT_CONFIDENCE_MIN,
        "max_sub": INTENT_MAX_SUB_QUERIES, "ts": int(time.time())}
    tmp = REWRITE_CACHE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False)
    os.replace(tmp, REWRITE_CACHE_FILE)


def _rw_cache_key(query, history):
    hist = json.dumps(history or [], ensure_ascii=False, sort_keys=True)
    raw = (f"{REWRITE_MODEL}|{REWRITE_CACHE_VER}|{INTENT_CONFIDENCE_MIN}|"
           f"{INTENT_MAX_SUB_QUERIES}|{query}|{hist}")
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def cached_intent_rewrite(query, history=None):
    """intent_rewrite 的磁盘缓存包装（线上同参数）。每题在评估内仅被一个 worker 处理，
    锁仅保护 dict；降级/追问结果同样缓存，保证跨档复跑严格可复现。"""
    key = _rw_cache_key(query, history)
    with _rw_lock:
        if key in _rw_cache:
            RW_CACHE_HIT["hit"] += 1
            return _rw_cache[key]
    RW_CACHE_HIT["miss"] += 1
    r = intent_rewrite(query, history=history, model=REWRITE_MODEL,
                       max_sub_queries=INTENT_MAX_SUB_QUERIES,
                       confidence_min=INTENT_CONFIDENCE_MIN)
    with _rw_lock:
        _rw_cache[key] = r
    return r


# ---------------------------------------------------------------- LLM 客户端单例
# OpenAI 客户端内部持有 httpx 连接池，线程安全可共享；
# 切勿在每次调用时新建（旧实现 320 次重复 TLS 握手是主要耗时来源之一）。
_GEN_CLI = None
_QA_CLI = None


def _get_gen_cli():
    global _GEN_CLI
    if _GEN_CLI is None:
        _GEN_CLI = OpenAI(api_key=GEN_API_KEY, base_url=AGICTO_BASE_URL, timeout=120)
    return _GEN_CLI


def _get_qa_cli():
    global _QA_CLI
    if _QA_CLI is None:
        _QA_CLI = OpenAI(api_key=REWRITE_API_KEY, base_url=AGICTO_BASE_URL, timeout=120)
    return _QA_CLI


# ---------------------------------------------------------------- 生成答案磁盘缓存
_ANS_CACHE_FILE = os.path.join(OUT_DIR, "eval_ans_cache.json")
_ANS_PROMPT_VER = "v2"  # v2: key 纳入 k/MMR/限额/index_md5；改 prompt 再 bump
_ans_cache = {}
_ans_lock = threading.Lock()
# 当前答案评估口径（由 evaluate_answers 在每档开跑前设置；档位串行执行，线程池内共享只读）
# style: "short"=30字短答评估口径 / "online"=线上客服 system+[n] 引用同源口径
_ANS_SCOPE = {"index_md5": None, "k": None, "use_mmr": None,
              "mmr_lambda": None, "max_per_source": None, "style": "short"}
_INDEX_MD5_CACHE = None


def get_index_md5():
    """索引文件 md5（前 12 位），懒加载缓存。所有评估产物用它做同源校验。"""
    global _INDEX_MD5_CACHE
    if _INDEX_MD5_CACHE is None:
        h = hashlib.md5()
        with open(INDEX_FILE, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        _INDEX_MD5_CACHE = h.hexdigest()[:12]
    return _INDEX_MD5_CACHE


def _load_ans_cache():
    """加载已生成答案缓存。
    文件含 __cache_meta__={gen_model,prompt_ver,index_md5}，任一不符则整份作废重建
    （杜绝换索引/改口径后复用旧答案——旧 MMR 失效事故的机制性防线）。"""
    global _ans_cache
    if not os.path.exists(_ANS_CACHE_FILE):
        return
    try:
        with open(_ANS_CACHE_FILE, encoding="utf-8") as f:
            raw = json.load(f)
        meta = raw.get("__cache_meta__") if isinstance(raw, dict) else None
        expect = {"gen_model": GEN_MODEL, "prompt_ver": _ANS_PROMPT_VER,
                  "index_md5": get_index_md5()}
        if not meta or any(meta.get(k) != v for k, v in expect.items()):
            print(f"[答案缓存口径不符，整份作废] 缓存meta={meta} 当前={expect}")
            log.warning("ans cache invalidated: cache=%s expect=%s", meta, expect)
            _ans_cache = {}
            return
        _ans_cache = {k: v for k, v in raw.items() if k != "__cache_meta__"}
        print(f"已加载答案缓存: {len(_ans_cache)} 条 (model={GEN_MODEL}, index={expect['index_md5']})")
    except Exception as e:
        print(f"[答案缓存读取失败，重建] {e}")
        _ans_cache = {}


def _save_ans_cache():
    """落盘（先写 tmp 再 replace，防并发/中断产生半截文件）。"""
    with _ans_lock:
        snapshot = dict(_ans_cache)
    snapshot["__cache_meta__"] = {
        "gen_model": GEN_MODEL, "prompt_ver": _ANS_PROMPT_VER,
        "index_md5": get_index_md5(), "ts": int(time.time())}
    tmp = _ANS_CACHE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False)
    os.replace(tmp, _ANS_CACHE_FILE)


def _ans_cache_key(question, context):
    """key 冻结完整生效口径：模型|prompt|索引指纹|k|MMR|λ|限额|风格|问题|上下文。
    任一口径变化 key 自动变化，跨档只有口径+上下文完全相同才复用。
    short 档刻意省略 style 段以保持 v2 时期 key 逐字节不变（历史缓存零浪费）；
    online 档插入 style=online| 独立命名空间。"""
    s = _ANS_SCOPE
    style_seg = "" if s.get("style", "short") == "short" else f"style={s['style']}|"
    raw = (f"{GEN_MODEL}|{_ANS_PROMPT_VER}|{s['index_md5']}|k={s['k']}|"
           f"mmr={s['use_mmr']}|lam={s['mmr_lambda']}|mps={s['max_per_source']}|"
           f"{style_seg}{question}|{context}")
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def gen_qa(chunk_content, n=2):
    """从 chunk 内容生成 n 个问答对（问题|答案），返回 [(q, a), ...]。
    兼容多种输出格式：a|b、a｜b、问题：a 答案：b、Q: a A: b。
    含网络超时指数退避重试（AGICTO 偶发 SSL handshake timeout）。"""
    import re as _re
    import time as _time
    prompt = (
        f"基于以下知识库片段，生成 {n} 组\"问题|答案\"，每行一组用竖线|分隔，"
        f"问题不超过20字，答案不超过30字且必须能在片段中找到依据，不要编号：\n\n{chunk_content[:600]}"
    )
    _qa_cli = _get_qa_cli()
    for attempt in range(3):  # 网络失败重试
        try:
            resp = _qa_cli.chat.completions.create(
                model=REWRITE_MODEL, max_tokens=250,
                messages=[{"role": "user", "content": prompt}])
            break
        except Exception as e:
            if attempt == 2:
                raise
            wait = 2 ** (attempt + 1)
            print(f"    [gen_qa 重试 {attempt+1}/2] {str(e)[:60]}，{wait}s 后重试")
            _time.sleep(wait)
    text = resp.choices[0].message.content.strip()
    out = []
    for line in text.splitlines():
        line = line.strip().lstrip("0123456789.、-• ")
        if not line:
            continue
        m = _re.search(r"问题[:：]?\s*(.+?)\s*答案[:：]?\s*(.+)", line)
        if m:
            q, a = m.group(1), m.group(2)
        else:
            parts = _re.split(r"[|｜]", line, maxsplit=1)
            if len(parts) != 2:
                continue
            q, a = parts[0].strip(), parts[1].strip()
        q, a = q.strip(), a.strip()
        if 4 <= len(q) <= 40 and 1 <= len(a) <= 60:
            out.append((q, a))
    if out:
        return out[:n]
    _time.sleep(0.5)
    return []


def _load_metadata():
    with open(METADATA_FILE, encoding="utf-8") as f:
        return json.load(f)


def _chunk_hash_index(meta):
    """当前索引的 chunk_hash -> 位置映射。返回 (h2i, n_no_hash, dup_hashes)。"""
    h2i, no_hash, seen, dup = {}, 0, set(), []
    for i, m in enumerate(meta):
        h = m.get("chunk_hash")
        if not h:
            no_hash += 1
            continue
        if h in seen:
            dup.append(h)
        seen.add(h)
        h2i[h] = i
    return h2i, no_hash, dup


def _remap_qa(items, h2i):
    """按 chunk_hash 把 gold_idx 重映射到当前索引位置。
    金标块在新索引中不存在（内容被删改/重切后消失）的整条剔除——
    悬空 gold_idx 会让 Recall 恒为 0，污染所有档位指标。
    返回 (kept, missing)。"""
    kept, missing = [], []
    for it in items:
        h = it.get("chunk_hash")
        if not h or h not in h2i:
            missing.append(it)
            continue
        it["gold_idx"] = h2i[h]
        kept.append(it)
    return kept, missing


def _write_qa(items, qmeta):
    """原子落盘 QA 尺子（tmp + replace，防中断产生半截文件）。"""
    tmp = QA_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"__qa_meta__": qmeta, "items": items}, f,
                  ensure_ascii=False, indent=2)
    os.replace(tmp, QA_FILE)


def build_qa_set(n_chunks=80, index_md5=None):
    """从代表性 chunk 生成 QA 对（含金标答案与金标块 chunk_hash）。
    类别轮询抽样 + 每来源文件限额（MAX_QA_PER_SOURCE），缓解评估集同源偏差。
    gold_idx 只是出生索引位置；跨索引复用靠 chunk_hash remap，不依赖该位置。"""
    meta = _load_metadata()
    if not any(m.get("chunk_hash") for m in meta):
        raise SystemExit(
            "[元数据缺指纹] metadata 无 chunk_hash 字段；请重跑 2-build_index.py 后再生成 QA 集")
    texts = [(i, m) for i, m in enumerate(meta)
             if m["type"] == "text" and 100 < len(m["content"]) < 800]
    by_cat = {}
    for i, m in texts:
        by_cat.setdefault(m["category"], []).append((i, m))
    per_cat = max(1, n_chunks // max(len(by_cat), 1))
    cat_lists = []
    for items in by_cat.values():
        step = max(1, len(items) // per_cat)
        cat_lists.append(list(items[::step]))

    samples, src_cnt = [], {}
    pos = 0
    while len(samples) < n_chunks and any(cat_lists):
        cl = cat_lists[pos % len(cat_lists)]
        pos += 1
        while cl:
            i, m = cl.pop(0)
            src = m["source"]
            if src_cnt.get(src, 0) >= MAX_QA_PER_SOURCE:
                continue  # 同一文件已抽够，跳过
            samples.append((i, m))
            src_cnt[src] = src_cnt.get(src, 0) + 1
            break
        if all(not cl for cl in cat_lists):
            break

    qa = []
    for pos, (i, m) in enumerate(samples):
        pairs = gen_qa(m["content"])
        for q, a in pairs:
            qa.append({"query": q, "gold_idx": i,
                       "chunk_hash": m["chunk_hash"],
                       "gold_answer": a,
                       "source": m["source"], "heading": m.get("heading", "")})
        print(f"  [{pos + 1}/{len(samples)}] 生成 {len(pairs)} 对（{m['source'][:24]}）")
        time.sleep(0.3)
    _write_qa(qa, {"schema_ver": QA_SCHEMA_VER,
                   "born_index_md5": index_md5 or get_index_md5(),
                   "n_chunks": n_chunks, "ts": int(time.time()),
                   "remap": []})
    n_src = len(set(x["source"] for x in qa))
    print(f"QA 集: {len(qa)} 对 / {n_src} 个来源文件（含金标答案+chunk_hash），保存至 {os.path.relpath(QA_FILE, _HERE)}")
    return qa


def load_or_build_qa(idx_md5):
    """加载 QA 尺子并按 chunk_hash 与当前索引对齐（尺子与索引位置解耦）。
    - 有指纹 QA：gold_idx 按内容指纹 remap 到当前位置；失配率 <= 阈值时剔除失配条目后复用，
      并回写对齐结果与 remap 履历；失配率超阈值说明知识库内容已大变，尺子判废重生成。
    - 无指纹旧 QA（扁平 list / 条目无 chunk_hash）：无法跨索引对齐，直接重新生成。"""
    meta = _load_metadata()
    h2i, no_hash, dup = _chunk_hash_index(meta)
    if no_hash:
        log.warning("metadata 中 %d 条无 chunk_hash（旧库残留），不能作为 QA 金标", no_hash)
    if dup:
        log.warning("chunk_hash 出现 %d 个重复指纹，remap 以末位置为准", len(dup))

    raw = None
    if os.path.exists(QA_FILE):
        try:
            with open(QA_FILE, encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            log.warning("QA 文件损坏，重新生成: %s", e)

    items, qmeta = None, {}
    if isinstance(raw, dict) and "items" in raw:
        items, qmeta = raw["items"], raw.get("__qa_meta__", {})
    elif isinstance(raw, list):
        items, qmeta = raw, {}

    if items is not None and any(x.get("chunk_hash") for x in items):
        kept, missing = _remap_qa(items, h2i)
        miss_rate = len(missing) / max(len(items), 1)
        if miss_rate > QA_REBUILD_MISS_RATE:
            print(f"[QA 失配率 {miss_rate:.0%} > {QA_REBUILD_MISS_RATE:.0%}，知识库内容已大变，重新生成尺子]")
            log.warning("QA remap miss_rate=%.2f 超阈值，尺子判废重生成", miss_rate)
        else:
            born = qmeta.get("born_index_md5") or qmeta.get("index_md5") or idx_md5
            hist = list(qmeta.get("remap", []))
            hist.append({"index_md5": idx_md5, "ts": int(time.time()),
                         "total": len(items), "mapped": len(kept),
                         "missing": len(missing)})
            _write_qa(kept, {"schema_ver": QA_SCHEMA_VER,
                             "born_index_md5": born,
                             "n_chunks": qmeta.get("n_chunks", 0),
                             "ts": int(time.time()), "remap": hist})
            msg = f"QA 尺子按指纹对齐: {len(kept)}/{len(items)} 条金标命中当前索引（{idx_md5}）"
            if missing:
                msg += f"，剔除 {len(missing)} 条已消失金标"
                for it in missing[:5]:
                    print(f"    [失配剔除] Q={it['query'][:30]} src={it.get('source', '')[:30]}")
            print(msg)
            log.info("QA remap %s -> %s: kept=%d missing=%d",
                     born, idx_md5, len(kept), len(missing))
            return kept
    elif items is not None:
        print(f"[QA 尺子无 chunk_hash 指纹（{len(items)} 条旧结构），无法跨索引对齐，重新生成]")
        log.warning("QA 无指纹旧结构 %d 条，重生成", len(items))

    print("生成 QA 集（80 chunk，每 chunk 2 对）...")
    t_qa = time.time()
    qa = build_qa_set(n_chunks=80, index_md5=idx_md5)
    log.info("QA 集生成完成: %d 对, %.0fs", len(qa), time.time() - t_qa)
    return qa


def _run_concurrent(work_items, workers, label, on_done=None, checkpoint=None):
    """通用并发执行：work_items 为 [fn]，按下标顺序返回结果列表。
    on_done(done,total,secs) 进度回调；checkpoint() 周期性落盘缓存。"""
    results = [None] * len(work_items)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn): pos for pos, fn in enumerate(work_items)}
        done = 0
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
            done += 1
            if done % 10 == 0 or done == len(work_items):
                if checkpoint:
                    checkpoint()
                if on_done:
                    on_done(done, len(work_items), time.time() - t0)
    return results


def evaluate(retriever, qa, vec_cache, k=5, use_rerank=False, label="",
             max_per_source=None, use_mmr=False, mmr_lambda=None,
             workers=EVAL_WORKERS, use_rewrite=False):
    """检索侧评估：严格 Recall@k / MRR + 宽松 Recall（同源同章节算命中）。
    线程池并发（workers 路）；embedding 提交前串行预热（通常全命中缓存）。
    use_rewrite=True 时复刻线上 retrieve() 口径：intent_rewrite 改写后的子问题
    走向量/BM25/rerank（单意图 search、多意图 search_multi、rerank_q 拼接规则一致）；
    action=clarify 的题线上会短路不检索，记 strict/lenient=0 并在 clarify_rate 单列。"""
    rw_map = {}
    if use_rewrite:
        for item in qa:  # 改写串行（走磁盘缓存），再预热全部子问题向量
            q = item["query"]
            intent = cached_intent_rewrite(q, history=None)
            rw_map[q] = intent
            if intent["action"] != "retrieve":
                continue
            for sq in intent["queries"]:
                if sq not in vec_cache:
                    vec_cache[sq] = get_embedding(expand_query_terms(sq))
    else:
        for item in qa:  # 预热 query 向量，避免并发写 vec_cache
            q = item["query"]
            if q not in vec_cache:
                vec_cache[q] = get_embedding(expand_query_terms(q))  # 跨配置复用

    def _one(item):
        q, gold = item["query"], item["gold_idx"]
        detail = {"query": q, "mode": "rewrite" if use_rewrite else "original"}
        if use_rewrite:
            intent = rw_map[q]
            detail["types"] = intent.get("query_types", [])
            if intent["action"] == "clarify":  # 线上短路：本题不进检索
                detail.update(clarify=True, clarify_msg=intent.get("clarify", ""))
                return 0, 0, 0.0, detail
            subs = intent["queries"]
            detail.update(clarify=False, subs=subs)
            detail["multi_intent"] = bool(intent.get("multi_intent"))
            detail["fallback"] = (not intent.get("query_types") and len(subs) == 1
                                  and float(intent.get("confidence") or 0) >= 0.999)
            if len(subs) > 1:  # 与 app.retrieve 多意图分支一致
                ranked = retriever.search_multi(
                    [vec_cache[s] for s in subs], subs, k=k,
                    use_rerank=use_rerank, rerank_query=q + " " + " ".join(subs),
                    max_per_source=max_per_source,
                    use_mmr=use_mmr, mmr_lambda=mmr_lambda)
            else:
                ranked = retriever.search(vec_cache[subs[0]], subs[0], k=k,
                                          use_rerank=use_rerank, rerank_query=subs[0],
                                          max_per_source=max_per_source,
                                          use_mmr=use_mmr, mmr_lambda=mmr_lambda)
        else:
            ranked = retriever.search(vec_cache[q], q, k=k,
                                      use_rerank=use_rerank, rerank_query=q,
                                      max_per_source=max_per_source,
                                      use_mmr=use_mmr, mmr_lambda=mmr_lambda)
        idxs = [i for i, _ in ranked]
        gold_meta = retriever.meta[gold]
        strict = gold in idxs
        lenient = strict or any(
            retriever.meta[i].get("source") == gold_meta.get("source")
            and retriever.meta[i].get("heading") == gold_meta.get("heading")
            for i in idxs)
        rr = 1.0 / (idxs.index(gold) + 1) if strict else 0.0
        detail["hit_idxs"] = idxs
        return 1 if strict else 0, 1 if lenient else 0, rr, detail

    def _progress(done, total, secs):
        print(f"    [{label}] {done}/{total}  ({secs:.0f}s)")

    outs = _run_concurrent(
        [lambda it=item: _one(it) for item in qa], workers, label,
        on_done=_progress, checkpoint=_save_rewrite_cache if use_rewrite else None)
    recalls = [o[0] for o in outs]
    lenients = [o[1] for o in outs]
    mrr_sum = sum(o[2] for o in outs)
    details = [o[3] for o in outs]
    n = len(qa)
    res = {"label": label, "k": k, "use_rerank": use_rerank,
           "max_per_source": max_per_source, "use_mmr": use_mmr,
           "mmr_lambda": mmr_lambda, "query_mode": "rewrite" if use_rewrite else "original",
           "recall@k": sum(recalls) / n if n else 0,
           "recall_lenient": sum(lenients) / n if n else 0,
           "mrr": mrr_sum / n if n else 0, "n": n}
    if use_rewrite:
        n_clarify = sum(1 for d in details if d.get("clarify"))
        n_multi = sum(1 for d in details if d.get("multi_intent"))
        n_fb = sum(1 for d in details if d.get("fallback"))
        n_search = n - n_clarify
        res.update({
            "clarify_rate": n_clarify / n if n else 0,
            "multi_intent_rate": n_multi / n if n else 0,
            "fallback_rate": n_fb / n if n else 0,
            # 排除短路题后的条件 Recall（实际进检索的题上的命中水平）
            "recall@k_excl_clarify": (sum(recalls) / n_search) if n_search else None,
            "rewrite_details": details})
        print(f"  [{label}] Recall@{k}={res['recall@k']:.3f}  "
              f"宽松Recall={res['recall_lenient']:.3f}  MRR={res['mrr']:.3f}  "
              f"(n={n}, 追问短路={n_clarify}, 多意图={n_multi}, 回退={n_fb})")
    else:
        print(f"  [{label}] Recall@{k}={res['recall@k']:.3f}  "
              f"宽松Recall={res['recall_lenient']:.3f}  MRR={res['mrr']:.3f}  (n={n})")
    return res


_STOP_TERMS = {"可以", "需要", "怎么", "如何", "什么", "哪些", "一个", "以及", "或者", "的话", "时候"}
_CN_DIGIT = {"零": "0", "一": "1", "二": "2", "两": "2", "三": "3", "四": "4",
             "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}


def _cn_run_to_int(run):
    """标准中文数字转换（0-9999，正式写法：三百零五=305；三百五按正式=305）。"""
    val, cur = 0, 0
    for ch in run:
        if ch in _CN_DIGIT:
            cur = int(_CN_DIGIT[ch])
        elif ch in ("十", "百", "千"):
            u = {"十": 10, "百": 100, "千": 1000}[ch]
            val += (cur or 1) * u
            cur = 0
    return val + cur


def _cn_to_arabic(s):
    """中文数字写法归一为阿拉伯数字（七->7、三十->30、三百零五->305）。"""
    import re as _re
    if not s:
        return s

    def repl(m):
        return str(_cn_run_to_int(m.group(0)))

    # 连续中文数字段（含十百千）整体转换；注意“两”也算 2
    return _re.sub(r"[零一二两三四五六七八九十百千]+", repl, s)


def extract_gold_keys(text):
    """抽取金标答案的关键事实：数字（必中，中文数字归一）+ 2字以上实体词（同样归一化）。"""
    import re as _re
    norm = _cn_to_arabic(text)
    nums = list(dict.fromkeys(_re.findall(r"\d+(?:\.\d+)?", norm)))
    words = [w for w in dict.fromkeys(_re.findall(r"[一-龥A-Za-z]{2,}", norm))
             if w not in _STOP_TERMS]
    return nums, words


def _keys_hit(text, nums, words):
    norm_text = _cn_to_arabic(text)
    num_ok = all(n in norm_text for n in nums)
    word_need = max(1, (len(words) + 1) // 2)
    word_hit = sum(1 for w in words if w in norm_text)
    return num_ok and word_hit >= word_need


def gen_answer(question, context, prompt_style="short"):
    """生成答案。两种 prompt 口径：
    - short（评估专用）：30 字短答、无 system 角色、无引用要求
    - online（线上同源）：settings.ONLINE_SYSTEM_PROMPT 客服 5 规则 + 【用户问题】模板，
      上下文须由 retriever.build_context 编号组装，max_tokens=600 容纳表格/多段回答
    命中 eval_ans_cache.json 时直接返回（跨档同上下文/跨运行零调用）；失败不缓存。"""
    key = _ans_cache_key(question, context)
    with _ans_lock:
        if key in _ans_cache:
            ANS_CACHE_HIT["hit"] += 1
            return _ans_cache[key]
    ANS_CACHE_HIT["miss"] += 1
    log.info("gen_answer cache miss: style=%s q=%s", prompt_style, question[:30])
    if prompt_style == "online":
        messages = [{"role": "system", "content": ONLINE_SYSTEM_PROMPT},
                    {"role": "user",
                     "content": f"【背景知识】\n{context}\n\n【用户问题】\n{question}"}]
        max_tokens = 600
    else:
        prompt = (
            "仅依据背景知识回答问题，30字以内，直接给出答案不要解释。\n\n"
            f"【背景知识】\n{context}\n\n【问题】{question}"
        )
        messages = [{"role": "user", "content": prompt}]
        max_tokens = 100
    cli = _get_gen_cli()
    for attempt in range(3):
        try:
            resp = cli.chat.completions.create(
                model=GEN_MODEL, max_tokens=max_tokens, messages=messages)
            ans = resp.choices[0].message.content.strip()
            with _ans_lock:
                _ans_cache[key] = ans
            return ans
        except Exception as e:
            if attempt == 2:
                print(f"    [生成失败] {str(e)[:60]}")
                return None
            wait = 2 ** (attempt + 1)
            time.sleep(wait)
    return None


def answer_hit(question, gold_answer, context, prompt_style="short"):
    """N2 修正后的答案侧判定（严格口径）：
    - evidence: 金标关键事实是否出现在【检索到的上下文】（证据是否召回到位）
    - gen_ok  : 金标关键事实是否出现在【生成回答】中（不再把上下文计入，去污染）
    online 档答案带 [n] 行内引用，判定前剥离引用标记——来源编号（如 [10]）
    可能恰好包含金标数字（如"1"），不剥离会虚高命中率。
    返回 (evidence, gen_ok, answer)。"""
    ans = gen_answer(question, context, prompt_style=prompt_style)
    if not ans:
        return 0, 0, ""
    nums, words = extract_gold_keys(gold_answer)
    ans_for_judge = re.sub(r"\[\d+\]", "", ans) if prompt_style == "online" else ans
    if not nums and not words:
        ok = 1 if gold_answer[:6] in ans_for_judge or ans in gold_answer else 0
        return ok, ok, ans
    evidence = 1 if _keys_hit(context, nums, words) else 0
    gen_ok = 1 if _keys_hit(ans_for_judge, nums, words) else 0
    return evidence, gen_ok, ans


def evaluate_answers(retriever, qa, vec_cache, label, use_rerank, n=ANSWER_EVAL_N,
                     max_per_source=None, use_mmr=False, mmr_lambda=None, k=3,
                     workers=EVAL_WORKERS, prompt_style="short"):
    """答案侧评估（top-k 上下文），输出 4 个指标：
    证据覆盖率 / 无条件答案命中率 / 证据到位条件下的生成命中率（生成侧真实差距）
    / 证据在却答错比例。
    线程池并发；检索+生成在同一 worker 内完成，生成答案走磁盘缓存，
    不同档检索到相同上下文时自动去重，不重复调 LLM。
    prompt_style="online" 时上下文走与线上逐字同源的 build_context（[n] 编号+同源合并）。
    口径（k/MMR/λ/限额/style/index_md5）写入缓存 key，见 _ANS_SCOPE。"""
    global _ANS_SCOPE
    _ANS_SCOPE = {"index_md5": get_index_md5(), "k": k,
                  "use_mmr": bool(use_mmr), "mmr_lambda": mmr_lambda,
                  "max_per_source": max_per_source, "style": prompt_style}
    subset = qa[:n]
    for item in subset:  # 预热 query 向量，避免并发写 vec_cache
        q = item["query"]
        if q not in vec_cache:
            vec_cache[q] = get_embedding(expand_query_terms(q))

    def _one(item):
        q = item["query"]
        ranked = retriever.search(vec_cache[q], q, k=k,
                                  use_rerank=use_rerank, rerank_query=q,
                                  max_per_source=max_per_source,
                                  use_mmr=use_mmr, mmr_lambda=mmr_lambda)
        if prompt_style == "online":  # 线上同构：来源分组+[n]编号+邻块去重
            ctx, _refs = build_context(ranked, retriever.meta)
        else:
            ctx = "\n\n".join(retriever.meta[i].get("parent_context",
                              retriever.meta[i]["content"]) for i, _ in ranked)
        ev, ok, ans = answer_hit(q, item["gold_answer"], ctx,
                                 prompt_style=prompt_style)
        return ev, ok, ans, item

    def _progress(done, total, secs):
        print(f"    [{label}] {done}/{total}  ({secs:.0f}s)")

    outs = _run_concurrent(
        [lambda it=item: _one(it) for item in subset], workers, label,
        on_done=_progress, checkpoint=_save_ans_cache)

    n_ev = n_gen = n_ev_gen = n_fail = 0
    shown = 0
    for ev, ok, ans, item in outs:  # 按 subset 原顺序聚合，失败样例稳定可复现
        q = item["query"]
        n_ev += ev
        n_gen += ok
        n_ev_gen += (1 if ev and ok else 0)
        if ev and not ok:
            n_fail += 1
            if shown < 5:
                print(f"    [证据在但答错] Q={q} | 金标={item['gold_answer']} | 生成={ans[:40]}")
                shown += 1
    _save_ans_cache()
    n = len(subset)
    res = {
        "evidence_coverage": n_ev / n if n else 0,          # 证据是否召回到位
        "answer_hit": n_gen / n if n else 0,                # 无条件答案命中（严格，仅看answer）
        "answer_hit_given_evidence": n_ev_gen / n_ev if n_ev else None,  # 生成侧真实水平
        "gen_loss_when_evidence": n_fail / n if n else 0,   # 证据在却答错的比例
        "n": n, "k": k, "use_rerank": use_rerank,
        "max_per_source": max_per_source, "use_mmr": use_mmr,
        "mmr_lambda": mmr_lambda, "prompt_style": prompt_style,
    }
    cond = res["answer_hit_given_evidence"]
    cond_str = f"{cond:.3f}" if cond is not None else "N/A"
    print(f"  [{label}] 证据覆盖={res['evidence_coverage']:.3f}  "
          f"答案命中={res['answer_hit']:.3f}  "
          f"证据到位时命中={cond_str}  (n={n}, prompt={prompt_style})")
    return res


def evaluate_intents(qa, workers=EVAL_WORKERS):
    """N5: 跑线上同款 intent_rewrite 流水线，统计意图分布与短路率。
    - clarify_rate：信息不足直接追问的比例（这些问题不走向量检索，省检索/改写）
    - fallback_rate：置信度不足/识别失败按原 query 检索的比例
    - multi_intent_rate：多意图拆分比例；query_types 为类型分布
    每条 QA 的意图结果随记录落盘，便于人工抽检。"""
    from collections import Counter
    subset = qa[:ANSWER_EVAL_N]

    def _one(item):
        q = item["query"]
        try:
            r = cached_intent_rewrite(q, history=None)
        except Exception as e:
            r = {"action": "retrieve", "query_types": [], "multi_intent": False,
                 "effective": q, "confidence": 1.0, "clarify": "", "queries": [q]}
            log.warning("intent failed for %s: %s", q[:20], str(e)[:60])
        return {"query": q, "action": r["action"], "types": r.get("query_types", []),
                "multi_intent": bool(r.get("multi_intent")),
                "confidence": r.get("confidence"),
                "n_subqueries": len(r.get("queries", [])),
                "clarify": r.get("clarify", "")}

    outs = _run_concurrent([lambda it=it: _one(it) for it in subset], workers, "意图评测")

    n = len(outs)
    n_clarify = sum(1 for o in outs if o["action"] == "clarify")
    n_multi = sum(1 for o in outs if o["multi_intent"])
    # fallback 特征：无类型 + 单子问题 + 置信度被置为兜底值 1.0
    n_fb = sum(1 for o in outs if not o["types"] and o["n_subqueries"] == 1
               and float(o["confidence"] or 0) >= 0.999)
    type_cnt = Counter(t for o in outs for t in o["types"])
    confs = [float(o["confidence"]) for o in outs if o["types"]]
    res = {
        "label": "意图评测(N5)",
        "n": n,
        "clarify_rate": n_clarify / n if n else 0,        # 追问短路率
        "fallback_rate": n_fb / n if n else 0,           # 置信不足→原query
        "multi_intent_rate": n_multi / n if n else 0,
        "query_type_dist": dict(type_cnt),
        "mean_confidence": sum(confs) / len(confs) if confs else None,
        "confidence_threshold": INTENT_CONFIDENCE_MIN,
        "details": outs,
    }
    print(f"  [意图评测] 追问短路={res['clarify_rate']:.3f}  "
          f"置信不足回退={res['fallback_rate']:.3f}  "
          f"多意图={res['multi_intent_rate']:.3f}  类型分布={dict(type_cnt)}")
    return res


def evaluate_intent_gold(workers=EVAL_WORKERS):
    """意图专用金标集评估（人工静态尺子 eval_intent_gold.json，不绑索引、不调 LLM 生成）。
    每条标注：gold_action(retrieve/clarify) / gold_types(主类型在前) /
    gold_multi(是否多意图) / must_include(消解后必须保留的事实词) / history。
    指标：
      action_acc            行为判定（检索 vs 追问）准确率
      clarify_p/r/f1        追问（正类=clarify）精确率/召回率/F1
      type_hit_rate         gold 主类型出现在预测类型集合中的比例
      multi_acc             多意图拆分判定准确率
      term_keep_rate        retrieve 题 must_include 词在改写后子问题中的覆盖率（消解质量）
    金标文件缺失时返回 None（调用方跳过，不报错）。"""
    if not os.path.exists(INTENT_GOLD_FILE):
        log.warning("意图金标集缺失，跳过: %s", os.path.relpath(INTENT_GOLD_FILE, _HERE))
        return None
    with open(INTENT_GOLD_FILE, encoding="utf-8") as f:
        gold_doc = json.load(f)
    items = gold_doc["items"]
    print(f"\n=== 意图金标评估（{len(items)} 条人工标注，{gold_doc.get('__meta__', {}).get('schema_ver', '?')}） ===")

    def _one(g):
        try:
            r = cached_intent_rewrite(g["query"], history=g.get("history"))
        except Exception as e:
            r = {"action": "retrieve", "query_types": [], "multi_intent": False,
                 "queries": [g["query"]], "confidence": 1.0}
            log.warning("intent gold failed for %s: %s", g["query"][:20], str(e)[:60])
        gold_types = g.get("gold_types", [])
        pred_types = r.get("query_types", [])
        type_hit = (not gold_types) or (gold_types[0] in pred_types)
        # 消解词保留：仅对实际进入检索的题判定；全部 must_include 词出现才算通过
        joined = " ".join(r.get("queries", []))
        terms = g.get("must_include", [])
        term_hit = None
        if r["action"] == "retrieve" and terms:
            term_hit = all(t in joined for t in terms)
        return {
            "query": g["query"], "history_n": len(g.get("history", [])),
            "gold_action": g["gold_action"], "pred_action": r["action"],
            "action_ok": g["gold_action"] == r["action"],
            "gold_types": gold_types, "pred_types": pred_types,
            "type_hit": type_hit,
            "gold_multi": bool(g.get("gold_multi", False)),
            "pred_multi": bool(r.get("multi_intent")),
            "multi_ok": bool(g.get("gold_multi", False)) == bool(r.get("multi_intent")),
            "term_hit": term_hit, "must_include": terms,
            "queries": r.get("queries", []),
            "clarify": r.get("clarify", ""), "note": g.get("note", "")}

    outs = _run_concurrent([lambda g=g: _one(g) for g in items], workers, "意图金标")

    n = len(outs)
    n_action = sum(o["action_ok"] for o in outs)
    n_type = sum(o["type_hit"] for o in outs)
    n_multi = sum(o["multi_ok"] for o in outs)
    # clarify 作为二分类正类
    tp = sum(1 for o in outs if o["gold_action"] == "clarify" and o["pred_action"] == "clarify")
    fp = sum(1 for o in outs if o["gold_action"] != "clarify" and o["pred_action"] == "clarify")
    fn = sum(1 for o in outs if o["gold_action"] == "clarify" and o["pred_action"] != "clarify")
    p = tp / (tp + fp) if (tp + fp) else None
    rcl = tp / (tp + fn) if (tp + fn) else None
    if p is None or rcl is None or (p + rcl) == 0:
        f1 = 0.0 if (fn > 0 or fp > 0) else None  # 有一方全错记 0；两边都无样本则无定义
    else:
        f1 = 2 * p * rcl / (p + rcl)
    term_rows = [o["term_hit"] for o in outs if o["term_hit"] is not None]
    shown = 0
    for o in outs:  # 行为判错样例先暴露（最多 5 条）
        if not o["action_ok"] and shown < 5:
            print(f"    [行为判错] Q={o['query'][:28]} gold={o['gold_action']} "
                  f"pred={o['pred_action']} types={o['pred_types']}")
            shown += 1
    res = {"label": "意图金标评估", "n": n,
           "schema_ver": gold_doc.get("__meta__", {}).get("schema_ver"),
           "action_acc": n_action / n,
           "clarify_precision": p, "clarify_recall": rcl, "clarify_f1": f1,
           "type_hit_rate": n_type / n,
           "multi_acc": n_multi / n,
           "term_keep_rate": (sum(term_rows) / len(term_rows)) if term_rows else None,
           "term_eval_n": len(term_rows),
           "details": outs}
    f1s = f"{f1:.3f}" if f1 is not None else "N/A"
    ps = f"{p:.3f}" if p is not None else "N/A"
    rs = f"{rcl:.3f}" if rcl is not None else "N/A"
    print(f"  [意图金标] action_acc={res['action_acc']:.3f}  "
          f"追问 P={ps} R={rs} F1={f1s}  类型命中={res['type_hit_rate']:.3f}  "
          f"多意图判定={res['multi_acc']:.3f}  消解词保留={res['term_keep_rate']}  (n={n})")
    return res


def main():
    import faiss
    t_start = time.time()
    idx_md5 = get_index_md5()  # 所有产物同源校验的基准指纹
    # 启动校验：旧 eval_results.json 若来自不同索引/旧管线，显式标记为不可引用
    if os.path.exists(RESULT_FILE):
        try:
            old = json.load(open(RESULT_FILE, encoding="utf-8")).get("config", {})
            if old.get("index_md5") != idx_md5:
                log.warning("旧 eval_results 索引指纹 %s != 当前 %s，本次将覆盖",
                            old.get("index_md5"), idx_md5)
            elif old.get("retriever_pipeline_ver") != PIPELINE_VER:
                log.warning("旧 eval_results 管线版本 %s 早于当前 %s（代码已改未重跑），本次将覆盖",
                            old.get("retriever_pipeline_ver"), PIPELINE_VER)
        except Exception:
            pass
    log.info("=== 评估开始 === workers=%s emb=%s gen=%s index_md5=%s pipeline=%s",
             EVAL_WORKERS, EMBEDDING_MODEL, GEN_MODEL, idx_md5, PIPELINE_VER)
    _load_emb_cache()
    _load_ans_cache()
    _load_rewrite_cache()
    index = faiss.read_index(ascii_faiss_path(INDEX_FILE))
    # N6: 索引维度与 settings 强一致校验
    if index.d != EMBEDDING_DIM:
        raise SystemExit(f"[维度不匹配] 索引 {index.d} 维 != EMBEDDING_DIM={EMBEDDING_DIM}，请重建索引")
    with open(METADATA_FILE, encoding="utf-8") as f:
        meta = json.load(f)
    with open(BM25_FILE, "rb") as f:
        bm25 = pickle.load(f)

    # QA 集（带索引指纹；指纹不符/无指纹且无法举证同源则重生成）
    qa = load_or_build_qa(idx_md5)

    vec_cache = {}
    print("\n=== 检索评估 ===")
    results = []

    # 1) 纯向量（关闭 BM25）
    r_vec = HybridRetriever(index, meta, bm25=None, use_bm25=False, use_rerank=False)
    results.append(evaluate(r_vec, qa, vec_cache, k=5, label="纯向量"))

    # 2) 混合检索（向量 + BM25 RRF）
    r_hyb = HybridRetriever(index, meta, bm25=bm25, use_bm25=True, use_rerank=False)
    results.append(evaluate(r_hyb, qa, vec_cache, k=5, label="混合(BM25+向量)"))

    # 3) 混合 + Rerank（P0-2 核验：calls/fallbacks 暴露是否真正生效）
    RERANK_STATS["calls"] = RERANK_STATS["fallbacks"] = 0
    r_rr = HybridRetriever(index, meta, bm25=bm25, use_bm25=True,
                           use_rerank=True, rerank_topn=RERANK_TOPN)
    results.append(evaluate(r_rr, qa, vec_cache, k=5, use_rerank=True, label="混合+Rerank"))
    # 生效判据：calls>0 且有模型真正出序（last_model 非空）；首选模型挂掉但成功降级也算生效。
    # preferred_ok 仅表示"首选模型零降级"，上游计费故障时为 False 属外部问题，不等于 rerank 失效。
    # bool() 包裹：and 会直接返回后一个操作数（模型名字符串），导致落盘 ok 为 str 而非布尔
    rerank_effective = bool(RERANK_STATS["calls"] > 0 and RERANK_STATS["last_model"])
    preferred_ok = rerank_effective and RERANK_STATS["fallbacks"] == 0
    results.append({"label": "rerank生效", "calls": RERANK_STATS["calls"],
                    "fallbacks": RERANK_STATS["fallbacks"],
                    "last_model": RERANK_STATS["last_model"],
                    "last_source": RERANK_STATS["last_source"],
                    "ok": rerank_effective, "preferred_ok": preferred_ok})
    print(f"  rerank 生效核验: calls={RERANK_STATS['calls']}, fallbacks={RERANK_STATS['fallbacks']}, "
          f"last={RERANK_STATS['last_model']}@{RERANK_STATS['last_source']}, "
          f"effective={rerank_effective}, preferred_ok={preferred_ok}")
    log.info("rerank calls=%s fallbacks=%s last=%s@%s effective=%s",
             RERANK_STATS["calls"], RERANK_STATS["fallbacks"],
             RERANK_STATS["last_model"], RERANK_STATS["last_source"], rerank_effective)
    _save_emb_cache()  # 持久化 embedding，下次检索侧评估无需 API

    # 4) 2×2 网格（检索侧）：{k=3,k=5} × {后处理开(MMR+线上限额), 关}，分离 k 与后处理的净贡献
    print("\n=== 检索 2×2 网格（k × 后处理） ===")
    results.append(evaluate(r_rr, qa, vec_cache, k=3, use_rerank=True,
                            label="检索(k=3,后处理关)"))
    RERANK_STATS["calls"] = RERANK_STATS["fallbacks"] = 0
    results.append(evaluate(r_rr, qa, vec_cache, k=3, use_rerank=True,
                            label=f"检索(k=3,后处理开:MMR+限额{MAX_PER_SOURCE})",
                            max_per_source=MAX_PER_SOURCE,
                            use_mmr=USE_MMR, mmr_lambda=MMR_LAMBDA))
    results.append(evaluate(r_rr, qa, vec_cache, k=5, use_rerank=True,
                            label="检索(k=5,后处理关)"))
    results.append(evaluate(r_rr, qa, vec_cache, k=5, use_rerank=True,
                            label=f"检索(k=5,后处理开:MMR+限额{MAX_PER_SOURCE})",
                            max_per_source=MAX_PER_SOURCE,
                            use_mmr=USE_MMR, mmr_lambda=MMR_LAMBDA))

    # 线上口径检索档：与上一档完全同参，仅把原句 query 换成 intent_rewrite 改写后 query
    # （线上 retrieve(use_rewrite=True) 同分支），量化改写净贡献；改写走磁盘缓存可复跑
    if INTENT_ENABLED:
        print("\n=== 检索：线上改写 query 档（与后处理开档仅 query 来源不同） ===")
        t_rw = time.time()
        results.append(evaluate(r_rr, qa, vec_cache, k=5, use_rerank=True,
                                label=f"检索(k=5,线上口径:后处理开+改写query,限额{MAX_PER_SOURCE})",
                                max_per_source=MAX_PER_SOURCE,
                                use_mmr=USE_MMR, mmr_lambda=MMR_LAMBDA,
                                use_rewrite=True))
        _save_rewrite_cache()
        log.info("改写检索档完成 %.0fs", time.time() - t_rw)

    # 阈值扫描（检索侧，环境变量覆盖；与网格档同管线，仅参考不进主结论）
    print("\n=== 阈值扫描（参考） ===")
    SWEEP_LAMBDA = float(os.environ.get("SWEEP_LAMBDA", "0.55"))
    SWEEP_MAX_PER_SOURCE = int(os.environ.get("SWEEP_MAX_PER_SOURCE", "3"))
    results.append(evaluate(r_rr, qa, vec_cache, k=5, use_rerank=True,
                            label=f"扫描(k=5+限额{SWEEP_MAX_PER_SOURCE}+MMR{MMR_LAMBDA})",
                            max_per_source=SWEEP_MAX_PER_SOURCE,
                            use_mmr=True, mmr_lambda=MMR_LAMBDA))
    results.append(evaluate(r_rr, qa, vec_cache, k=5, use_rerank=True,
                            label=f"扫描(k=5+限额{MAX_PER_SOURCE}+MMR{SWEEP_LAMBDA})",
                            max_per_source=MAX_PER_SOURCE,
                            use_mmr=True, mmr_lambda=SWEEP_LAMBDA))
    log.info("扫描完成 lambda_sweep=%s mps_sweep=%s", SWEEP_LAMBDA, SWEEP_MAX_PER_SOURCE)

    # N5: 意图识别评测（线上同款 intent_rewrite 流水线，无金标仅分布统计）
    if INTENT_ENABLED:
        print("\n=== N5 意图识别评测 ===")
        t_intent = time.time()
        results.append(evaluate_intents(qa))
        _save_rewrite_cache()
        log.info("意图评测完成 %.0fs", time.time() - t_intent)

    # 意图专用金标集（人工静态尺子：action/追问/类型/多意图/消解词，不绑索引）
    if INTENT_ENABLED:
        t_ig = time.time()
        gold_res = evaluate_intent_gold()
        if gold_res is not None:
            results.append(gold_res)
            _save_rewrite_cache()
            log.info("意图金标评估完成 %.0fs", time.time() - t_ig)

    # 答案侧 2×2 网格：{k=3,k=5} × {后处理开(MMR+线上限额), 关}——四格同管线、同金标，
    # 这才是决定答案侧 k 定稿与 MMR 净贡献的有效判据（旧版漏传 MMR 的对照已作废）。
    print("\n=== 答案侧 2×2 网格（k × 后处理，short prompt，并发） ===")
    ans_rows = [
        ("答案(k=3,后处理关)", dict(k=3)),
        (f"答案(k=3,后处理开:MMR+限额{MAX_PER_SOURCE})",
         dict(k=3, max_per_source=MAX_PER_SOURCE, use_mmr=USE_MMR, mmr_lambda=MMR_LAMBDA)),
        ("答案(k=5,后处理关)", dict(k=5)),
        (f"答案(k=5,后处理开:MMR+限额{MAX_PER_SOURCE})",
         dict(k=5, max_per_source=MAX_PER_SOURCE, use_mmr=USE_MMR, mmr_lambda=MMR_LAMBDA)),
    ]
    for label, kw in ans_rows:
        res = evaluate_answers(r_rr, qa, vec_cache, label, use_rerank=True, **kw)
        results.append({"label": label, **res})
        _save_ans_cache()
    # 线上同 prompt 答案档：客服 system + [n] 编号上下文（与 app.py 同源），k=5+线上后处理参数
    print("\n=== 答案侧：线上同 prompt 档（客服system+[n]引用，k=5，线上后处理） ===")
    label_online = f"答案(线上同prompt,k=5,限额{MAX_PER_SOURCE})"
    ans_online = evaluate_answers(r_rr, qa, vec_cache, label_online, use_rerank=True,
                                  k=5, max_per_source=MAX_PER_SOURCE,
                                  use_mmr=USE_MMR, mmr_lambda=MMR_LAMBDA,
                                  prompt_style="online")
    results.append({"label": label_online, **ans_online})
    _save_ans_cache()
    # 参考基线：无 rerank 的纯混合（k=5），量化 rerank 对答案侧的价值
    ans_hyb = evaluate_answers(r_hyb, qa, vec_cache, "答案(混合无rerank,k=5)",
                               use_rerank=False, k=5)
    results.append({"label": "答案(混合无rerank,k=5)", **ans_hyb})
    _save_ans_cache()
    log.info("答案缓存 hit=%s miss=%s", ANS_CACHE_HIT["hit"], ANS_CACHE_HIT["miss"])

    # 配置快照（可复现性 + 同源/同管线举证）
    config_snapshot = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "retriever_pipeline_ver": PIPELINE_VER,
        "gen_model": GEN_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "rewrite_model": REWRITE_MODEL,
        "rerank_topn": RERANK_TOPN,
        "rerank_chain": __import__("settings").RERANK_MODELS,
        "max_per_source": MAX_PER_SOURCE,
        "use_mmr": USE_MMR,
        "mmr_lambda": MMR_LAMBDA,
        "sweep_lambda": SWEEP_LAMBDA,
        "sweep_max_per_source": SWEEP_MAX_PER_SOURCE,
        "intent_confidence_min": INTENT_CONFIDENCE_MIN,
        "index_total": index.ntotal,
        "index_dim": index.d,
        "index_md5": idx_md5,
        "qa_count": len(qa),
        "answer_eval_n": ANSWER_EVAL_N,
        "eval_workers": EVAL_WORKERS,
        "ans_prompt_ver": _ANS_PROMPT_VER,
        "answer_prompt_styles": ["short", "online"],
        # 线上 system prompt 指纹：prompt 漂移时该值变化，产物可与线上代码交叉核对
        "online_system_prompt_sha1": hashlib.sha1(
            ONLINE_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12],
        "ans_cache_hit": ANS_CACHE_HIT["hit"],
        "ans_cache_miss": ANS_CACHE_HIT["miss"],
        "rewrite_cache_ver": REWRITE_CACHE_VER,
        "rewrite_cache_hit": RW_CACHE_HIT["hit"],
        "rewrite_cache_miss": RW_CACHE_HIT["miss"],
        "intent_gold_file": os.path.basename(INTENT_GOLD_FILE),
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    output = {"config": config_snapshot, "results": results}
    tmp = RESULT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    os.replace(tmp, RESULT_FILE)
    _save_rewrite_cache()
    print(f"\n结果已保存: {os.path.relpath(RESULT_FILE, _HERE)}")
    log.info("=== 评估结束 === 总耗时 %.0fs, 结果 %s",
             time.time() - t_start, os.path.basename(RESULT_FILE))


if __name__ == "__main__":
    main()

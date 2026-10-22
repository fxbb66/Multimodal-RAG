# -*- coding: utf-8 -*-
"""
迪士尼RAG助手（完整知识库版 · 优化后）- 前端服务
================================================
检索侧：R1 BM25+向量RRF / R2 agicto rerank-v3.5→qwen3-rerank 降级链 / R3 改写 / R4 过滤 /
        R5 同义词 / R6 余弦 / I1 父子块 / 类别预过滤 / 同源限额 / MMR
生成侧：G1 行内引用[n]+文件定位 / G2 qwen-plus / G3 LRU+TTL 会话 / G4 SSE
工程：密钥与超参全部来自 lib/settings.py（.env），embedding LRU 缓存、
      简单限流、检索耗时日志、/api/health 可观测
启动：.venv/Scripts/python.exe 8-金标扩充版本/frontend/app.py
"""
import os
import sys
import json
import uuid
import time
import pickle
import queue
import threading
from collections import OrderedDict, deque
import numpy as np
import dashscope
from http import HTTPStatus
from openai import OpenAI
from flask import Flask, request, jsonify, render_template, send_from_directory, Response, stream_with_context

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VERSION_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
KB_ROOT = os.path.normpath(os.path.join(VERSION_ROOT, "..", "迪士尼RAG知识库（完整）"))
OUT_DIR = os.path.join(VERSION_ROOT, "output")

# 引入 lib（settings 自动加载 .env，密钥零硬编码）
sys.path.insert(0, os.path.join(VERSION_ROOT, "lib"))
from retriever import (HybridRetriever, BM25Index, expand_query_terms,  # noqa: E402
                       intent_rewrite, ascii_faiss_path, RERANK_STATS,
                       build_context, _rule_gate_short_no_entity,
                       INTENT_STATS)
from settings import (DASHSCOPE_API_KEY, EMBEDDING_API_KEY, GEN_API_KEY,  # noqa: E402
                      RERANK_API_KEY, require_keys,
                      AGICTO_BASE_URL, DASHSCOPE_BASE_URL,
                      EMBEDDING_MODEL, GEN_MODEL, DASHSCOPE_GEN_MODEL,
                      REWRITE_MODEL, RERANK_TOPN, MAX_PER_SOURCE, USE_MMR,
                      MMR_LAMBDA, MMR_LAMBDA_MULTI, BM25_WEIGHT,
                      MEDIA_SIM_THRESHOLD, MEDIA_GAP,
                      SESSION_MAX, SESSION_TTL, EMBED_CACHE_SIZE, RATE_LIMIT_PER_MIN,
                      RERANK_MODELS, INTENT_ENABLED, INTENT_CONFIDENCE_MIN,
                      INTENT_MAX_CLARIFY, INTENT_MAX_SUB_QUERIES,
                      EVIDENCE_SELF_CHECK, EVIDENCE_MAX_RETRY,
                      ONLINE_SYSTEM_PROMPT as SYSTEM_PROMPT)

# FAISS Windows 中文路径规避：复制到 ASCII 临时路径再读（不 os.chdir 污染全局 cwd）
INDEX_FILE = os.path.join(OUT_DIR, "disney_full_index.faiss")
METADATA_FILE = os.path.join(OUT_DIR, "disney_full_metadata.json")
BM25_FILE = os.path.join(OUT_DIR, "bm25_index.pkl")

if DASHSCOPE_API_KEY:
    dashscope.api_key = DASHSCOPE_API_KEY  # 保留用于 rerank 降级兜底
client = OpenAI(api_key=GEN_API_KEY, base_url=AGICTO_BASE_URL)  # 默认 AGICTO 生成

# 前端可切换回答生成平台：百炼客户端懒加载（未配置密钥时切换会收到明确报错）
_client_bailian = None
_client_lock = threading.Lock()

# 允许前端选择的生成平台白名单：provider -> (展示名, 实际模型名)
GEN_PROVIDERS = {
    "agicto": ("AGICTO 平台", GEN_MODEL, GEN_API_KEY),
    "bailian": ("百炼平台", DASHSCOPE_GEN_MODEL, DASHSCOPE_API_KEY),
}


def get_gen_client(provider):
    """按前端选择返回 (client, model)；未知 provider 回退 AGICTO。"""
    global _client_bailian
    if provider == "bailian":
        if not DASHSCOPE_API_KEY:
            raise RuntimeError("未配置百炼平台密钥（DASHSCOPE_API_KEY），无法切换到百炼生成")
        if _client_bailian is None:
            with _client_lock:
                if _client_bailian is None:
                    _client_bailian = OpenAI(
                        api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
        return _client_bailian, DASHSCOPE_GEN_MODEL
    return client, GEN_MODEL

IMAGE_KEYWORDS = ["图片", "海报", "照片", "看看", "长什么样", "图"]

index = None
metadata = None
bm25 = None
retriever = None
categories = []
INDEX_MD5 = None  # load_index 时计算一次，供 /api/health 暴露配置指纹
START_TS = time.time()

# G3 会话存储：LRU 上限 + TTL，防内存泄漏
sessions = OrderedDict()  # sid -> {"msgs": [...], "ts": float}
# P1 embedding LRU 缓存（相同查询不重复计费/等待）
_emb_cache = OrderedDict()
_emb_lock = threading.Lock()
# embedding 缓存命中计数（/api/metrics 与 /api/health 暴露命中率，前端圆饼据此渲染）
_emb_metrics = {"hits": 0, "misses": 0}
# P1 简单限流：ip -> 最近 60s 请求时间戳队列
_rate_buckets = {}
_rate_lock = threading.Lock()
# G3 会话存储锁：threaded=True 下并发读写 sessions 必须串行化，否则 move_to_end/popitem/
# 读写交错会致 LRU 失效、KeyError 或会话串号（embedding/限流已有锁，此处补齐会话锁）
_sess_lock = threading.Lock()


def sess_get(sid):
    with _sess_lock:
        if not sid or sid not in sessions:
            return []
        s = sessions[sid]
        if time.time() - s["ts"] > SESSION_TTL:
            sessions.pop(sid, None)
            return []
        s["ts"] = time.time()
        sessions.move_to_end(sid)
        return s["msgs"]


def sess_append(sid, msgs):
    with _sess_lock:
        s = sessions.get(sid) or {"msgs": [], "ts": time.time(), "clarify_n": 0}
        s["msgs"] = (s["msgs"] + msgs)[-20:]  # 单会话最多保留 10 轮
        s.setdefault("clarify_n", 0)
        s["ts"] = time.time()
        sessions[sid] = s
        sessions.move_to_end(sid)
        while len(sessions) > SESSION_MAX:
            sessions.popitem(last=False)


def sess_clarify_get(sid):
    """连续追问计数：防追问死循环；用户正常提问后归零。"""
    with _sess_lock:
        if not sid or sid not in sessions:
            return 0
        return sessions[sid].get("clarify_n", 0)


def sess_clarify_incr(sid):
    if not sid:
        return
    with _sess_lock:
        s = sessions.get(sid) or {"msgs": [], "ts": time.time(), "clarify_n": 0}
        s["clarify_n"] = s.get("clarify_n", 0) + 1
        sessions[sid] = s


def sess_clarify_reset(sid):
    with _sess_lock:
        if sid and sid in sessions:
            sessions[sid]["clarify_n"] = 0


def sess_prune():
    with _sess_lock:
        now = time.time()
        for k in [k for k, v in sessions.items() if now - v["ts"] > SESSION_TTL]:
            sessions.pop(k, None)


def _maintenance_loop(interval=300):
    """周期清扫 daemon：过期会话 + 限流空桶。

    sess_prune 原先只在 /api/session/new 时触发，限流桶更只在同一 IP 再次
    请求时才清理过期项——不常来的 IP/会话会一直驻留内存。SESSION_MAX LRU
    与单 IP deque 长度各自有界，但周期清扫让 TTL 语义真正成立，长跑内存
    上界不依赖访客行为。清扫异常不影响主服务。"""
    while True:
        time.sleep(interval)
        try:
            sess_prune()
            now = time.time()
            with _rate_lock:
                dead = [k for k, q in _rate_buckets.items()
                        if not q or now - q[-1] > 60]
                for k in dead:
                    _rate_buckets.pop(k, None)
        except Exception as e:
            print(f"[maintenance] {e}")


def load_index():
    global index, metadata, bm25, retriever, categories, INDEX_MD5
    import faiss
    import hashlib
    # P0 配置 fail-fast：embedding/gen 为每个查询必需，缺失即启动报错（避免运行期晦涩失败）；
    # rerank 至少需一个 Key，否则仅告警不阻断（可在查询中关 use_rerank，缺失时检索期已有明确报错）
    require_keys("EMBEDDING_API_KEY", "GEN_API_KEY")
    if not (RERANK_API_KEY or DASHSCOPE_API_KEY):
        print("[配置告警] 未配置 RERANK_API_KEY/DASHSCOPE_API_KEY，rerank 不可用；"
              "如需使用请在查询中关闭 use_rerank，否则检索期将明确报错。")
    index = faiss.read_index(ascii_faiss_path(INDEX_FILE))
    h = hashlib.md5()
    with open(INDEX_FILE, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    INDEX_MD5 = h.hexdigest()[:12]
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(BM25_FILE, "rb") as f:
        bm25 = pickle.load(f)
    retriever = HybridRetriever(index, metadata, bm25=bm25,
                                use_bm25=True, use_rerank=True,
                                rerank_topn=RERANK_TOPN)
    stat = {}
    text_n = img_n = 0
    for m in metadata:
        if m.get("is_shadow"):
            continue  # 双向量影子槽与主条目同图，不占类目统计
        cat = m.get("category", "未分类")
        c = stat.setdefault(cat, {"text_count": 0, "image_count": 0})
        if m["type"] == "text":
            c["text_count"] += 1
            text_n += 1
        else:
            c["image_count"] += 1
            img_n += 1
    categories = [{"name": k, **v} for k, v in
                  sorted(stat.items(), key=lambda x: -sum(x[1].values()))]
    print(f"[启动] 索引 {index.ntotal} 槽 / {len(metadata)} 块"
          f"（文本 {text_n} + 图片 {img_n}，图片双向量每张占 2 槽）, "
          f"{len(categories)} 类别, 生成模型={GEN_MODEL}")


# AGICTO 业务错误信封（HTTP 200 + {"error": {...}}）的稳定 code → 中文提示。
# 以 error.code 为键，原始 message 仅作明细附带，避免直接抛出难懂的底层异常。
_AGICTO_ERR_HINTS = {
    "insufficient_quota": "AGICTO 账户余额不足，请前往平台充值后重试",
    "invalid_api_key": "AGICTO API Key 无效或已过期（EMBEDDING_API_KEY）",
    "model_not_found": "AGICTO 上未找到 embedding 模型，请检查 EMBEDDING_MODEL 配置",
}


def get_text_embedding(text):
    with _emb_lock:
        if text in _emb_cache:
            _emb_cache.move_to_end(text)
            _emb_metrics["hits"] += 1
            return _emb_cache[text].copy()
    # tongyi-embedding-vision-plus：百炼原生 contents 格式
    import requests as _rq
    url = f"{AGICTO_BASE_URL.rstrip('/')}/embeddings"
    h = {"Authorization": f"Bearer {EMBEDDING_API_KEY}", "Content-Type": "application/json"}
    body = {"model": EMBEDDING_MODEL,
            "input": {"contents": [{"text": text}]},
            "encoding_format": "float"}
    resp = _rq.post(url, headers=h, json=body, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    embs = (data.get("output") or {}).get("embeddings") or data.get("data")
    if not embs:
        # 余额不足/Key 失效等场景 AGICTO 返回 200 错误信封，无 data/output——
        # 若不拦截会演变成 "'NoneType' object is not subscriptable"，无法定位
        err = data.get("error") if isinstance(data, dict) else None
        code = (err or {}).get("code") or (err or {}).get("type") or "unknown"
        detail = (err or {}).get("message") or str(data)[:160]
        hint = _AGICTO_ERR_HINTS.get(code, f"AGICTO embedding 接口返回错误（{code}）")
        raise RuntimeError(f"{hint}：{detail}")
    v = np.array(embs[0]["embedding"], dtype='float32')
    v = v / (np.linalg.norm(v) or 1.0)
    with _emb_lock:
        _emb_cache[text] = v.copy()
        _emb_cache.move_to_end(text)
        _emb_metrics["misses"] += 1
        while len(_emb_cache) > EMBED_CACHE_SIZE:
            _emb_cache.popitem(last=False)
    return v.copy()


def detect_image_intent(query):
    q = query.lower()
    return any(k in q for k in IMAGE_KEYWORDS)


def chunk_brief(m, idx, score):
    return {
        "id": int(idx),
        "type": m["type"],
        "source": m.get("source", ""),
        "category": m.get("category", ""),
        "heading": m.get("heading", ""),
        "tags": m.get("tags", []),
        "content": m["content"],
        "similarity": round(float(score), 4),   # R6: 已是余弦相似度
        "distance": round(float(1 - score), 4),  # 兼容旧字段
        "path": m.get("path", ""),
        "rel_path": m.get("rel_path", ""),
        "url": m.get("url", ""),
    }


def retrieve(query, top_k=5, category=None, use_rerank=True, use_rewrite=False,
             history=None, session_id=None, bm25_weight=None, on_stage=None):
    """意图识别 + 检索。返回 (action, clarify, ranked, media_ranked, effective_query, trace)。
    action: "retrieve" 正常检索 / "clarify" 需追问（此时 ranked 为空）。
    bm25_weight: 单查询临时 RRF 权重份额，None→settings.BM25_WEIGHT。
    on_stage: 阶段回调。真实阶段边界：发起 LLM 意图改写前 on_stage("rewrite")、
              查询向量化（embedding 远程调用）前 on_stage("embed")、
              进入本地召回（FAISS/BM25）前 on_stage("retrieve")、
              真正发起 rerank 前由 retriever 回调 on_stage("rerank")，
              供工作人员模式展示真实进度。on_stage 第二参数 detail 用于携带
              rerank 降级/失败信息。"""
    t0 = time.time()
    trace = {"rerank": use_rerank, "rewrite": use_rewrite, "category": category or "全部"}

    # R3 v2 意图识别 + 分类型改写（追问/多意图/指代消歧）
    intent = {"action": "retrieve", "clarify": "", "queries": [query],
              "query_types": [], "multi_intent": False, "effective": query}
    # 短问确定性门始终生效（不依赖改写开关/不调 LLM）：无历史极短无实体问直接追问，
    # 避免"多少钱"类查询在意图功能关闭时裸奔检索。三连问门属改写拆分，仍随开关。
    if INTENT_ENABLED:
        _gated = _rule_gate_short_no_entity(query, history)
        if _gated is not None:
            intent = _gated
            trace["intent_types"] = intent["query_types"]
            trace["intent_confidence"] = intent["confidence"]
            trace["rule_gate"] = intent.get("rule_gate")
    if use_rewrite and INTENT_ENABLED and intent["action"] != "clarify":
        if on_stage:
            on_stage("rewrite")
        intent = intent_rewrite(query, history=history, model=REWRITE_MODEL,
                                max_sub_queries=INTENT_MAX_SUB_QUERIES,
                                confidence_min=INTENT_CONFIDENCE_MIN)
        trace["intent_types"] = intent["query_types"]
        trace["intent_confidence"] = intent["confidence"]
        if intent.get("elapsed_ms") is not None:
            trace["intent_ms"] = intent["elapsed_ms"]
        # 降级可见化：告诉前端这次是否退化成"原 query 直接检索"以及为什么，
        # 前端在工作人员面板如实标注，不再让用户以为改写生效了。
        if intent.get("degraded"):
            trace["intent_degraded"] = True
            trace["intent_degrade_reason"] = intent.get("degrade_reason", "unknown")
        if intent.get("rule_gate"):
            trace["rule_gate"] = intent["rule_gate"]

    # 追问短路：不超上限才追问，超了按原 query 强行检索作答
    if intent["action"] == "clarify" and sess_clarify_get(session_id) < INTENT_MAX_CLARIFY:
        trace["total_ms"] = int((time.time() - t0) * 1000)
        trace["clarify"] = True
        return "clarify", intent["clarify"], [], [], query, trace
    sess_clarify_reset(session_id)

    sub_queries = intent["queries"]
    effective_query = intent["effective"]

    # R3 修复：rerank 必须使用消解/改写后的查询。子问题携带补全的实体
    # （如"这个乐园"→"虫虫乐园"），若仍用原始句重排，改写在召回阶段补回的实体会在
    # 最后一跳丢失（2026-10-02 虫虫乐园证据压线第3、被上海块淹没事件）。
    # 多意图：原句保整体意图 + 拼接各子问题的消解实体。
    rerank_q = (sub_queries[0] if len(sub_queries) == 1
                else query + " " + " ".join(sub_queries))
    trace["rerank_query"] = rerank_q

    # 阶段：改写结束，先做查询向量化（远程 embedding 调用，逐子问题），
    # 再进入本地 FAISS/BM25 召回
    if on_stage:
        on_stage("embed")

    # R5 同义词扩展 + embedding（逐子问题）
    t_emb = time.time()
    vecs = []
    for sq in sub_queries:
        vecs.append(get_text_embedding(expand_query_terms(sq)))
    trace["emb_ms"] = int((time.time() - t_emb) * 1000)

    # P1: 类别预过滤（allowed_idxs）
    allowed_idxs = None
    if category and category != "全部":
        allowed_idxs = [i for i, m in enumerate(metadata) if m.get("category") == category]
        if not allowed_idxs:
            allowed_idxs = None
    # 意图驱动 MMR：多意图候选跨子问题、冗余风险高，用更激进的 λ 去重；
    # 单意图维持标定值 0.72。实际取值写入 trace 供审计。
    eff_bm_weight = (BM25_WEIGHT if bm25_weight is None
                     else min(1.0, max(0.0, float(bm25_weight))))
    mmr_lam = MMR_LAMBDA_MULTI if intent.get("multi_intent") else MMR_LAMBDA
    trace["mmr_lambda"] = mmr_lam
    trace["bm25_weight"] = round(eff_bm_weight, 3)
    # 阶段：查询向量已就绪，进入本地混合召回
    if on_stage:
        on_stage("retrieve")
    t_rec = time.time()
    if len(sub_queries) > 1:
        # 多意图：合并召回统一 rerank，避免子问题互相挤占 top_k
        ranked = retriever.search_multi(vecs, sub_queries, k=top_k,
                                        use_rerank=use_rerank, rerank_query=rerank_q,
                                        allowed_idxs=allowed_idxs,
                                        max_per_source=MAX_PER_SOURCE,
                                        use_mmr=USE_MMR, mmr_lambda=mmr_lam,
                                        bm25_weight=eff_bm_weight,
                                        on_stage=on_stage)
        trace["sub_queries"] = sub_queries
        trace["evidence_check"] = {"triggered": False, "skipped": "multi_intent"}
    else:
        # S3：单意图分支接证据充分性自检（多意图分支本轮不接，trace 标记 skipped，
        # 避免每个子问题各触发一次自检把延迟与成本放大 N 倍）
        ranked, ev_info = retriever.search_with_evidence_check(
            vecs[0], sub_queries[0], k=top_k,
            use_rerank=use_rerank, rerank_query=rerank_q,
            allowed_idxs=allowed_idxs,
            max_per_source=MAX_PER_SOURCE,
            use_mmr=USE_MMR, mmr_lambda=mmr_lam,
            bm25_weight=eff_bm_weight,
            on_stage=on_stage,
            self_check=EVIDENCE_SELF_CHECK, max_retry=EVIDENCE_MAX_RETRY)
        if ev_info["triggered"]:
            trace["evidence_check"] = {k2: v for k2, v in ev_info.items()
                                       if k2 != "missing"}
            trace["evidence_check"]["missing"] = ev_info["missing"][:60]
        else:
            trace["evidence_check"] = {"triggered": False}
    trace["retrieve_ms"] = int((time.time() - t_rec) * 1000)

    # 媒体匹配：用主意图向量全库扫描图片（双向量下按 group 归并——影子槽不重复占位、
    # 且映射回主条目，避免 pick_image 看到同一张图两个候选或 content="" 的影子卡；
    # 与 retriever._dedup_vec_ranks 同口径：同组取最高分）
    media_scores, media_idxs = index.search(np.array([vecs[0]], dtype="float32"), index.ntotal)
    _media_best = {}
    for i, s in zip(media_idxs[0], media_scores[0]):
        if i == -1:
            continue
        m = metadata[int(i)]
        prim = m["shadow_of"] if m.get("is_shadow") else int(i)
        g = m.get("group_key") or m.get("chunk_hash") or prim
        s = float(s)
        if g not in _media_best or s > _media_best[g][1]:
            _media_best[g] = (prim, s)
    media_ranked = sorted(_media_best.values(), key=lambda x: x[1], reverse=True)
    trace["total_ms"] = int((time.time() - t0) * 1000)
    return "retrieve", "", ranked, media_ranked, effective_query, trace


def render_refs(refs):
    """N10: 确定性地追加文件定位清单（即使模型漏标也能看到出处）。"""
    lines = ["\n\n---\n**参考来源：**"]
    for r in refs:
        loc = f"（{r['heading']}）" if r["heading"] else ""
        lines.append(f"- [{r['n']}] {r['source']}{loc}")
    return "\n".join(lines)


def pick_image(media_ranked, query, want=True):
    """P1 媒体 gap 规则：top1 领先不足 MEDIA_GAP 视为歧义，对候选直接打 BM25 词面分决胜。
    阈值按【图片候选池自身最强分】校准——图片 OCR 文本短，与全文本库比长度归一化
    会导致图片永远无法胜出；池内相对比较即可防跨主题价格图错配。"""
    if not want:
        return None
    cands = [(i, s) for i, s in media_ranked
             if metadata[i]["type"] == "image" and s >= MEDIA_SIM_THRESHOLD]
    if not cands:
        return None
    top_i, top_s = cands[0]
    if len(cands) > 1 and top_s - cands[1][1] < MEDIA_GAP:
        exp_q = expand_query_terms(query)
        pool = [(i, s, bm25.score(exp_q, i)) for i, s in cands[:3]]
        pool_bm = [sc for _, _, sc in pool if sc > 0]
        if not pool_bm:
            print(f"[media] 歧义且词面无命中(top1={top_s:.3f} top2={cands[1][1]:.3f})，放弃匹配")
            return None
        max_bm = max(pool_bm)
        best = max(pool, key=lambda x: x[2])
        print(f"[media] 视觉歧义(top1={top_s:.3f} top2={cands[1][1]:.3f}) -> "
              f"池内词面决胜 idx={best[0]} bm={best[2]:.1f}/{max_bm:.1f}")
        return chunk_brief(metadata[best[0]], best[0], best[1])
    return chunk_brief(metadata[top_i], top_i, top_s)


def prepare_turn(query, top_k, category, use_rerank, use_rewrite, session_id,
                 gen_provider="agicto", bm25_weight=None, on_stage=None):
    """意图识别 + 检索 + 上下文 + 会话历史 + prompt，流式/非流式共用。
    返回 (messages, refs, meta, clarify)：clarify 非空表示需追问，messages 此时为 None。
    on_stage 透传给 retrieve，工作人员模式据此展示真实阶段。"""
    history = sess_get(session_id)[-6:]
    action, clarify, ranked, media_ranked, effective_query, trace = retrieve(
        query, top_k, category, use_rerank, use_rewrite,
        history=history, session_id=session_id, bm25_weight=bm25_weight,
        on_stage=on_stage)

    gen_model = GEN_PROVIDERS.get(gen_provider, GEN_PROVIDERS["agicto"])[1]
    if action == "clarify":
        sess_clarify_incr(session_id)
        meta = {"query": query, "effective_query": query, "top_k": top_k,
                "category": category or "全部", "chunks": [], "matched_image": None,
                "model": gen_model, "gen_provider": gen_provider,
                "trace": trace, "rerank_stats": dict(RERANK_STATS)}
        return None, [], meta, clarify

    context_str, refs = build_context(ranked, metadata)
    matched_image = pick_image(media_ranked, query, detect_image_intent(query))
    results = [chunk_brief(metadata[i], i, s) for i, s in ranked]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user",
                     "content": f"【背景知识】\n{context_str}\n\n【用户问题】\n{query}"})
    meta = {
        "query": query,
        "effective_query": effective_query,
        "top_k": top_k,
        "category": category or "全部",
        "chunks": results,
        "matched_image": matched_image,
        "model": gen_model,
        "gen_provider": gen_provider,
        "trace": trace,
        "rerank_stats": dict(RERANK_STATS),
    }
    return messages, refs, meta, ""


# ------------------ Flask ------------------
app = Flask(__name__, template_folder="templates", static_folder="static")


def rate_limited():
    """简单滑动窗口限流：每 IP 每分钟 RATE_LIMIT_PER_MIN 次（查询类接口）。

    键取 remote_addr：本服务绑 127.0.0.1，X-Forwarded-For 完全由客户端可控，
    信任它等于把限流开关交给请求方（伪造轮换头即可绕过）；仅当 remote_addr
    缺失（非常规代理场景）才回退 XFF。"""
    ip = request.remote_addr or request.headers.get("X-Forwarded-For", "?")
    now = time.time()
    with _rate_lock:
        q = _rate_buckets.setdefault(ip, deque())
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT_PER_MIN:
            return True
        q.append(now)
    return False


@app.route("/")
def index_page():
    return render_template("index.html", model=GEN_MODEL)


@app.route("/kb_image/<path:rel_path>")
def serve_kb_image(rel_path):
    return send_from_directory(KB_ROOT, rel_path)


@app.route("/api/categories")
def api_categories():
    # total=块数（双向量后槽位数多于块数，统计口径以块为准，影子槽不单列）
    return jsonify({"categories": categories, "total": len(metadata),
                    "model": GEN_MODEL})


@app.route("/api/stats")
def api_stats():
    """口径统一：总数动态计算，前端不再写死。双向量索引按块计数（影子槽不计）。"""
    text_n = sum(1 for m in metadata if m["type"] == "text")
    image_n = sum(1 for m in metadata
                  if m["type"] == "image" and not m.get("is_shadow"))
    return jsonify({
        "total": len(metadata),
        "text": text_n,
        "image": image_n,
        "index_slots": index.ntotal,  # 双向量：每张图片占 2 槽，槽位数 >= 块数
        "categories": len(categories),
        "gen_model": GEN_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "rerank_model": RERANK_MODELS[0] if RERANK_MODELS else "none",
        # 前端工作人员面板据此判断"意图改写"阶段是否真实存在
        "intent_enabled": INTENT_ENABLED,
        # 前端平台选择器据此渲染（未配置密钥的平台置灰）
        "gen_providers": {
            key: {"label": f"{label} · {model}", "model": model,
                  "available": bool(api_key)}
            for key, (label, model, api_key) in GEN_PROVIDERS.items()
        },
    })


@app.route("/api/health")
def api_health():
    """可观测：服务状态、索引规模与配置指纹、rerank 调用/降级、会话与缓存占用、运行时长。
    配置指纹（index_md5/模型链/阈值/prompt 指纹）供与 eval_results.json 交叉核对同源性。"""
    import hashlib
    build_overall = None
    br_path = os.path.join(OUT_DIR, "build_report.json")
    try:
        with open(br_path, encoding="utf-8") as f:
            build_overall = json.load(f).get("overall")
    except Exception:
        build_overall = None
    emb_total = _emb_metrics["hits"] + _emb_metrics["misses"]
    emb_hit_rate = (_emb_metrics["hits"] / emb_total) if emb_total else None
    return jsonify({
        "status": "ok",
        "index_total": index.ntotal,
        "index_dim": index.d,
        "index_md5": INDEX_MD5,
        "embedding_model": EMBEDDING_MODEL,
        "gen_model": GEN_MODEL,
        "rewrite_model": REWRITE_MODEL,
        "rerank_chain": RERANK_MODELS,
        "rerank_topn": RERANK_TOPN,
        "max_per_source": MAX_PER_SOURCE,
        "use_mmr": USE_MMR,
        "mmr_lambda": MMR_LAMBDA,
        "mmr_lambda_multi": MMR_LAMBDA_MULTI,
        "bm25_weight": BM25_WEIGHT,
        "intent_enabled": INTENT_ENABLED,
        "online_system_prompt_sha1": hashlib.sha1(
            SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12],
        "build_overall": build_overall,
        "sessions": len(sessions),
        "embed_cache": len(_emb_cache),
        "embed_cache_hit_rate": round(emb_hit_rate, 4) if emb_hit_rate is not None else None,
        "rerank_cumulative": dict(RERANK_STATS),
        "intent_cumulative": {
            "step1_ok": INTENT_STATS["step1_ok"],
            "step1_degraded": INTENT_STATS["step1_degraded"],
            "step2_degraded": INTENT_STATS["step2_degraded"],
            "low_confidence": INTENT_STATS["low_confidence"],
            "rule_gate_hit": INTENT_STATS["rule_gate_hit"],
            "pronoun_gate_hit": INTENT_STATS.get("pronoun_gate_hit", 0),
            "llm_retried": INTENT_STATS["llm_retried"],
            "llm_calls_failed": INTENT_STATS["llm_calls_failed"],
            "json_salvaged": INTENT_STATS.get("json_salvaged", 0),
            "degraded_by": dict(INTENT_STATS["degraded_by"]),
        },
        "uptime_sec": int(time.time() - START_TS),
    })


@app.route("/api/metrics")
def api_metrics():
    """运行时轻量指标：embedding 缓存命中率/计数（前端圆饼轮询此端点）、
    rerank 累计、会话数、运行时长。"""
    hits, misses = _emb_metrics["hits"], _emb_metrics["misses"]
    total = hits + misses
    return jsonify({
        "embed_cache": {
            "hits": hits, "misses": misses, "size": len(_emb_cache),
            "hit_rate": round(hits / total, 4) if total else None,
        },
        "rerank_cumulative": dict(RERANK_STATS),
        "sessions": len(sessions),
        "uptime_sec": int(time.time() - START_TS),
    })


@app.route("/api/chunks")
def api_chunks():
    category = request.args.get("category", "全部")
    q = (request.args.get("q") or "").strip().lower()
    out = []
    for idx, m in enumerate(metadata):
        if category != "全部" and m.get("category", "") != category:
            continue
        brief = chunk_brief(m, idx, 0.0)
        brief["similarity"] = None
        brief["distance"] = None
        if q:
            hay = (m["content"] + m.get("source", "") + m.get("heading", "")).lower()
            if q not in hay:
                continue
        out.append(brief)
    return jsonify({"category": category, "count": len(out), "chunks": out})


def _parse_query_payload():
    data = request.get_json(force=True, silent=True) or {}
    query = (data.get("query") or "").strip()
    # P1-5: query 长度上限，防止超长输入撑爆 embedding/LLM
    if len(query) > 1000:
        query = query[:1000]
    top_k = max(1, min(int(data.get("top_k", 5)), 20))
    category = data.get("category") or "全部"
    session_id = data.get("session_id")
    use_rerank = bool(data.get("use_rerank", True))
    use_rewrite = bool(data.get("use_rewrite", False))
    gen_provider = data.get("gen_provider", "agicto")
    if gen_provider not in GEN_PROVIDERS:
        gen_provider = "agicto"
    # 单查询临时 RRF 权重（缺省 None → settings.BM25_WEIGHT），非法值忽略
    bm25_weight = data.get("bm25_weight")
    if bm25_weight is not None:
        try:
            bm25_weight = min(1.0, max(0.0, float(bm25_weight)))
        except Exception:
            bm25_weight = None
    return (data, query, top_k, category, session_id, use_rerank,
            use_rewrite, gen_provider, bm25_weight)


@app.route("/api/query", methods=["POST"])
def api_query():
    (_, query, top_k, category, session_id, use_rerank,
     use_rewrite, gen_provider, bm25_weight) = _parse_query_payload()
    if not query:
        return jsonify({"error": "请输入查询内容"}), 400
    if rate_limited():
        return jsonify({"error": "请求过于频繁，请稍后再试"}), 429
    try:
        messages, refs, meta, clarify = prepare_turn(
            query, top_k, category, use_rerank, use_rewrite, session_id,
            gen_provider=gen_provider, bm25_weight=bm25_weight)
        if clarify:
            # 追问：直接返回追问话术，不进检索/生成；写入会话历史保证下轮上下文
            meta["answer"] = clarify
            meta["clarify"] = True
            if session_id:
                sess_append(session_id, [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": clarify},
                ])
            print(f"[clarify] q={query[:24]} -> {clarify[:40]}")
            return jsonify(meta)
        gen_cli, gen_model = get_gen_client(gen_provider)
        completion = gen_cli.chat.completions.create(model=gen_model, messages=messages)
        answer = completion.choices[0].message.content + render_refs(refs)
        if session_id:
            sess_append(session_id, [
                {"role": "user", "content": query},
                {"role": "assistant", "content": answer},
            ])
        meta["answer"] = answer
        print(f"[query] q={query[:24]} trace={meta['trace']}")
        return jsonify(meta)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/query_stream", methods=["POST"])
def api_query_stream():
    """G4 SSE：先 meta（chunks/引用映射/trace），再 token 流，末尾追加来源清单。
    R2/R3 开关由请求体透传，与非流式共用 prepare_turn，字段一致。"""
    (_, query, top_k, category, session_id, use_rerank,
     use_rewrite, gen_provider, bm25_weight) = _parse_query_payload()
    if not query:
        return jsonify({"error": "请输入查询内容"}), 400
    if rate_limited():
        return jsonify({"error": "请求过于频繁，请稍后再试"}), 429
    # 进入 SSE 前先解析客户端/模型：密钥缺失等配置错误以普通 JSON 返回，
    # 否则流已开始再抛异常，前端只能收到中断的事件流
    try:
        gen_cli, gen_model = get_gen_client(gen_provider)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    def sse(obj):
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    def generate():
        # 检索管线放在工作线程执行，阶段回调经队列实时送达本生成器——工作人员模式
        # 看到的 rewrite/embed/retrieve/rerank 是后端真实阶段边界，而不是前端定时器猜出来的假进度。
        q = queue.Queue()

        def on_stage(stage, detail=None):
            q.put(("stage", stage, detail))

        def worker():
            try:
                msgs, rfs, mt, cl = prepare_turn(
                    query, top_k, category, use_rerank, use_rewrite, session_id,
                    gen_provider=gen_provider, bm25_weight=bm25_weight,
                    on_stage=on_stage)
                q.put(("ready", msgs, rfs, mt, cl))
            except Exception as ex:
                q.put(("error", str(ex)))

        threading.Thread(target=worker, daemon=True).start()

        while True:
            item = q.get()
            if item[0] == "stage":
                evt = {"type": "stage", "stage": item[1]}
                # rerank 降级（备用模型成功）/ 全失败（未重排）等真实附加信息
                if item[2]:
                    evt["detail"] = item[2]
                yield sse(evt)
            elif item[0] == "error":
                print(f"[stream-error] q={query[:24]} {item[1]}")
                yield sse({"type": "error", "error": item[1]})
                yield "data: [DONE]\n\n"
                return
            else:  # ready：检索管线全部完成
                _, messages, refs, meta, clarify = item
                break

        if clarify:
            # 追问：一次性返回话术（走 SSE 保持前端路径一致），不进生成
            meta["refs"] = []
            meta["clarify"] = True
            yield sse({"type": "meta", **meta})
            yield sse({"type": "token", "content": clarify})
            if session_id:
                sess_append(session_id, [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": clarify},
                ])
            print(f"[clarify-stream] q={query[:24]} -> {clarify[:40]}")
            yield "data: [DONE]\n\n"
            return

        meta["refs"] = refs  # 来源编号映射，前端也可据此渲染
        yield sse({"type": "meta", **meta})
        # 工作人员面板：LLM 阶段在真正发起生成调用前推送（真实边界）
        yield sse({"type": "stage", "stage": "generate"})
        stream = gen_cli.chat.completions.create(
            model=gen_model, messages=messages, stream=True)
        full = []
        for chunk in stream:
            # 部分网关末尾会发 choices=[] 的 usage 统计块，需判空
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                full.append(delta)
                yield f"data: {json.dumps({'type': 'token', 'content': delta}, ensure_ascii=False)}\n\n"
        tail = render_refs(refs)  # N10: 确定性追加文件定位
        yield f"data: {json.dumps({'type': 'token', 'content': tail}, ensure_ascii=False)}\n\n"
        answer = "".join(full) + tail
        if session_id:
            sess_append(session_id, [
                {"role": "user", "content": query},
                {"role": "assistant", "content": answer},
            ])
        print(f"[stream] q={query[:24]} trace={meta['trace']}")
        yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream")


@app.route("/api/session/new")
def new_session():
    sid = uuid.uuid4().hex
    sess_prune()
    sess_append(sid, [])
    return jsonify({"session_id": sid})


if __name__ == "__main__":
    load_index()
    threading.Thread(target=_maintenance_loop, daemon=True).start()
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)

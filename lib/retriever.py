# -*- coding: utf-8 -*-
"""
迪士尼完整知识库 - 检索核心库
==============================
集中实现 RAG 检索侧优化，供 build_index 与 frontend/app 共用：
  R6  向量 L2 归一化，使用 IndexFlatIP（内积=余弦相似度）
  R1  混合检索：字符级 BM25（无 jieba 依赖）+ 向量，RRF 融合
  R2  重排：agicto rerank-v3.5（主用）→ qwen3-rerank（降级），AGICTO 整体不可用时百炼主 Key 兜底
      （上游 HTTP200 但空 results 的计费/故障信封按确定性失败处理：不重试、熔断 10 分钟、
       自动降级；2026-10-01 AGICTO 欠费期间每查询曾白等 ~6s；fallbacks/circuit_skips 可见）
  R3  查询改写：qwen3.8-flash 扩展查询（低置信时触发）
  R4  检索前过滤：丢弃超短无意义 chunk
  R5  专有名词词典：查询改写与同义词扩展
  I1  父子 chunk：子块召回、父块回填上下文
  M2  标签：从文件名/章节自动抽取
"""
import os
import re
import math
import json
import time
import threading
import numpy as np
from collections import Counter, OrderedDict

# ---------------------------------------------------------------- R5 专有名词词典
DISNEY_DICT = {
    # 标准名 -> 别名/常见写法
    "上海迪士尼乐园": ["上海迪士尼", "上海迪斯尼", "上海迪土尼"],
    "香港迪士尼乐园": ["香港迪士尼", "港迪"],
    "东京迪士尼乐园": ["东京迪士尼", "东京迪斯尼", "东迪", "东京迪士尼海洋"],
    "达菲家族": ["达菲", "Duffy", "雪莉玫", "星黛露", "玲娜贝儿", "Linabell"],
    "快速通行证": ["FastPass", "FP", "快速通行卡", "fastpass"],
    "迪士尼尊享卡": ["尊享卡", "导览卡", "VIP卡"],
    "门票": ["票价", "票种", "票券"],
    "退款": ["退票", "退订", "退钱"],
    "年票": ["年卡", "年卡会员", "奇妙金卡", "银卡", "水晶卡"],
}
SYNONYM_REVERSE = {}  # alias -> canon
for canon, aliases in DISNEY_DICT.items():
    SYNONYM_REVERSE[canon] = canon
    for a in aliases:
        SYNONYM_REVERSE[a] = canon


def expand_query_terms(text):
    """用专有名词词典扩展查询文本（追加标准名）。"""
    extra = []
    for alias, canon in SYNONYM_REVERSE.items():
        if alias in text and canon not in text:
            extra.append(canon)
    if extra:
        return text + " " + " ".join(sorted(set(extra)))
    return text


# ---------------------------------------------------------------- R1 字符级 BM25（无外部依赖）

def char_ngrams(s, n=2):
    s = re.sub(r"\s+", "", s or "")
    if len(s) < n:
        return [s] if s else []
    return [s[i:i + n] for i in range(len(s) - n + 1)]


class BM25Index:
    """基于字符二元组的 BM25，对中文短查询友好，无需分词。"""
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = []          # list of token lists
        self.df = Counter()
        self.idf = {}
        self.avgdl = 0.0
        self.n = 0

    def fit(self, texts):
        self.docs = [char_ngrams(t) for t in texts]
        self.n = len(self.docs)
        for d in self.docs:
            for t in set(d):
                self.df[t] += 1
        self.avgdl = sum(len(d) for d in self.docs) / max(self.n, 1)
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5))
                    for t, f in self.df.items()}
        return self

    def score(self, query, doc_idx):
        d = self.docs[doc_idx]
        if not d:
            return 0.0
        q_terms = char_ngrams(query)
        tf = Counter(d)
        dl = len(d)
        s = 0.0
        for t in set(q_terms):
            if t not in self.idf:
                continue
            f = tf.get(t, 0)
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            s += self.idf[t] * (f * (self.k1 + 1)) / (denom if denom else 1)
        return s

    def search(self, query, topk=50):
        scores = [(i, self.score(query, i)) for i in range(self.n)]
        scores = [(i, s) for i, s in scores if s > 0]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topk]


# ---------------------------------------------------------------- RRF 融合

def rrf_fuse(*ranked_lists, k=60, topk=10, weights=None):
    """加权 Reciprocal Rank Fusion。每个 list 是 [(doc_idx, score), ...]。
    weights: 与 ranked_lists 对齐的每路权重（缺省等权 1.0）；某路贡献 =
             weight / (k + rank + 1)。权重只改变各路相对话语权，不改变 RRF 的位次语义。"""
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    agg = {}
    for rl, w in zip(ranked_lists, weights):
        for rank, (didx, _) in enumerate(rl):
            agg[didx] = agg.get(didx, 0.0) + float(w) / (k + rank + 1)
    return sorted(agg.items(), key=lambda x: x[1], reverse=True)[:topk]


# ---------------------------------------------------------------- R2 Rerank（agicto rerank-v3.5 主用，失败降级）
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from settings import (AGICTO_BASE_URL, RERANK_API_KEY, REWRITE_API_KEY,  # noqa: E402
                      REWRITE_MODEL, RERANK_MODELS, RERANK_DASHSCOPE_MODELS,
                      BM25_WEIGHT)

RERANK_STATS = {"calls": 0, "fallbacks": 0, "last_model": None, "last_source": None,
                "circuit_skips": 0}  # P0-2: 核验 rerank 是否真正生效
_STATS_LOCK = threading.Lock()  # 评估脚本多线程并发 rerank 时保护计数器
# 熔断器：欠费/计费类确定性失败在进程内不会自愈，记录失败时刻，冷却期内整级跳过，
# 避免每个查询都为死模型白付 HTTP 往返与重试退避（AGICTO 欠费事故的直接教训）。
_CIRCUIT = {}          # (source, model) -> 失败时刻 epoch
_CIRCUIT_TTL = 600.0   # 冷却 10 分钟后放行一次探测（充值后无需重启进程）
# 候选池流水诊断：search() 各阶段候选数累加（配合 queries 求均值），用于定位后处理损失发生在哪一段
POOL_STATS = {"queries": 0, "rrf_out": 0, "after_r4": 0, "rerank_out": 0, "mmr_in": 0, "final": 0}


def ascii_faiss_path(src_path):
    """把 FAISS 索引复制到 ASCII 临时路径，规避 Windows fopen 不支持中文路径。"""
    import shutil
    import tempfile
    d = os.path.join(tempfile.gettempdir(), "disney_rag_faiss")
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, os.path.basename(src_path))
    shutil.copyfile(src_path, dst)
    return dst


def _rerank_via_openai_api(base_url, api_key, model, query, candidates, top_n, timeout=30):
    """AGICTO /v1/rerank（Cohere 风格，rerank-v3.5 与 qwen3-rerank 同契约）。

    契约依据：https://agicto.apifox.cn/282831591e0 （2026-10-02 双模型实测一致）
      请求：POST {base_url}/rerank，四字段【均必需】
            {"model", "query", "top_n", "documents":[str,...]}
      成功：HTTP200 {"results":[{"index":int,"relevance_score":float}], "usage":{...}}
      实测：
        - qwen3-rerank 正常返回上述结构；
        - rerank-v3.5 欠费时同样 HTTP200 但 body 为 {"id":...,"message":
          "Please add ... payment method at dashboard.cohere.com/billing ..."}
          且无 results —— 能走到 Cohere 计费环节本身即说明请求格式正确，
          这是计费故障不是参数错误。
    因此本函数对 AGICTO 上的 rerank-v3.5 / qwen3-rerank 共用一套格式，无需按模型分支；
    百炼平台的 qwen3-rerank 是另一套协议（dashscope SDK 原生调用），见 qwen_rerank()。

    返回 [(orig_idx, score)]；失败抛异常。
    必须把"200 空结果"判为失败，否则调用方会把空列表当成功造成 rerank 静默空转。"""
    import requests
    resp = requests.post(
        f"{base_url.rstrip('/')}/rerank",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "query": query, "documents": candidates, "top_n": top_n},
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"status={resp.status_code} {resp.text[:120]}")
    data = resp.json()
    items = data.get("results") or data.get("data") or []
    if not items:
        envelope = str(data.get("message") or data.get("error") or data)[:160]
        raise RuntimeError(f"rerank 空结果（疑似上游计费/故障信封）: {envelope}")
    return [(int(it["index"]), float(it["relevance_score"])) for it in items]


def qwen_rerank(query, candidates, topk=5, info=None):
    """
    用重排模型对候选重排。
    降级链：
      AGICTO（AGICTO_BASE_URL+RERANK_API_KEY）：rerank-v3.5（主用）-> qwen3-rerank（降级）
      若 AGICTO 整体不可用，再用百炼主 Key 兜底（仅 qwen3-rerank，
      rerank-v3.5 是 Cohere 模型名，百炼无此模型）。
    401/403/404 属确定性失败，直接跳下一级不重试。
    candidates: List[str]（chunk 正文）
    info: 可选字典，成功后写入 {"degraded": bool, "model": 实际模型, "source": 平台}，
          供调用方向前端展示"已降级备用模型"标记。
    返回排序后的 (orig_idx, score) 列表。
    """
    import dashscope
    from http import HTTPStatus
    if not candidates:
        return []
    with _STATS_LOCK:
        RERANK_STATS["calls"] += 1
    chain = [("AGICTO", m) for m in RERANK_MODELS] if RERANK_API_KEY else []
    main_key = getattr(dashscope, "api_key", None)
    if main_key and main_key != RERANK_API_KEY:
        chain += [("百炼", m) for m in RERANK_DASHSCOPE_MODELS]  # 百炼兜底（千问系列）
    if not chain:
        raise RuntimeError("无可用 rerank API Key（RERANK_API_KEY 和 DASHSCOPE_API_KEY 均缺失）")
    last_err = None
    for level, (source, model) in enumerate(chain):
        key = (source, model)
        with _STATS_LOCK:
            dead_since = _CIRCUIT.get(key)
        if dead_since is not None and time.time() - dead_since < _CIRCUIT_TTL:
            with _STATS_LOCK:
                RERANK_STATS["circuit_skips"] += 1
            continue  # 熔断中：本进程已知该级确定性失败，跳过不发请求
        for attempt in range(2):
            try:
                # 协议按平台自动适配（差异在平台而非模型名）：
                #   AGICTO → Cohere 风格 REST /v1/rerank（rerank-v3.5、qwen3-rerank 同格式）
                #   百炼   → dashscope TextReRank 原生 SDK（端点/信封均不同）
                if source == "AGICTO":
                    out = _rerank_via_openai_api(
                        AGICTO_BASE_URL, RERANK_API_KEY, model, query,
                        candidates, min(topk, len(candidates)))
                else:
                    resp = dashscope.TextReRank.call(
                        model=model, query=query, documents=candidates,
                        top_n=min(topk, len(candidates)),
                        return_documents=False, api_key=main_key,
                    )
                    if resp.status_code != HTTPStatus.OK:
                        raise RuntimeError(f"status={resp.status_code} {getattr(resp, 'message', '')}")
                    out = [(it.index, float(it.relevance_score)) for it in resp.output.results]
                with _STATS_LOCK:
                    _CIRCUIT.pop(key, None)  # 成功即解除熔断（如充值恢复后的首次探测）
                    if level > 0:
                        RERANK_STATS["fallbacks"] += 1  # 非首选模型命中，记一次降级
                    RERANK_STATS["last_model"] = model
                    RERANK_STATS["last_source"] = source
                if info is not None:
                    info["degraded"] = level > 0
                    info["model"] = model
                    info["source"] = source
                return out
            except Exception as e:
                last_err = f"{model}@{source}: {e}"
                # 401/403/404、余额不足与上游计费信封均为确定性失败：熔断本级并立即降级，
                # 不重试（欠费状态下重试只会在每个查询上白等约 3s 退避 + 多发一次请求）
                if any(code in str(e) for code in
                       ("401", "403", "404", "insufficient_quota",
                        "Not enough available money", "payment method")):
                    with _STATS_LOCK:
                        _CIRCUIT[key] = time.time()
                    break  # 确定性失败，换下一级
            time.sleep(2 ** attempt)
        print(f"[rerank] {last_err}，尝试降级下一级")
    raise RuntimeError(f"rerank failed: {last_err}")


# ---------------------------------------------------------------- R3 v2 意图识别 + 分类型改写

_INTENT_STEP1_PROMPT = """你是一个智能的查询分析专家，服务于上海迪士尼度假区的 RAG 问答系统。
请结合对话历史，分析用户当前查询，识别它命中了哪些类型（可同时命中多种）。

【类型定义】
1. 上下文依赖型(context_dependent) - 省略了主语/宾语，需结合上文才能补全，
   常见词："还有"、"其他"、"那"、"也"、"呢"、"再"、单独的"怎么去/多少钱/开放吗"
2. 对比型(comparison) - 需要在两个或以上对象间比较，
   常见词："哪个"、"比较"、"更"、"区别"、"还是"、"vs"、"性价比高"
3. 模糊指代型(pronoun) - 包含指代词且所指对象无法从本句确定，
   常见词："它"、"他们"、"这个"、"那里"、"都"（指代前文集合时）
4. 多意图型(multi_intent) - 包含 2 个及以上相互独立的问题，
   通常用"、"或"？"或"还有……吗"分隔，每个子问题可单独回答
5. 反问/情绪型(rhetorical) - 表面是反问或抱怨，背后有真实诉求，
   常见词："不会吧"、"难道"、"凭什么"、"怎么又"
6. 信息不足型(underspecified) - 关键参数缺失，不追问就无法给出有效答案。
   判定标准（极严格，宁缺勿滥）：仅限以下场景，且缺失参数在历史中也找不到：
   - 问门票/年卡价格但未指明哪个园区，且历史中也无园区信息
   - 问"适合吗/推荐吗"但未说明同行人群（儿童/老人/情侣等）且历史无此信息
   - 问规则类问题（退票/员工票/尊享卡）但主体不明
   以下情况【绝不能】判为信息不足型：
   - 问题涉及具体命名实体/角色（如 Bob、米奇、达菲、玲娜贝儿等），应直接检索
   - 问题含指代词（它/这个/你刚提到的）且历史可消解，应判为模糊指代型/上下文依赖型
   - 知识库能直接回答的泛化问题（如"必玩项目有哪些"）
   原则：能检索就不追问，宁检索不追问
7. 明确型(clear) - 以上均不命中，可直接检索

【优先级（从高到低，冲突时按此排序）】
多意图型 > 反问/情绪型 > 上下文依赖型 ≈ 模糊指代型 > 对比型 > 信息不足型 > 明确型
说明：信息不足型优先级降为最低——只要查询还能被改写（消歧/补全/拆分），就先改写再检索，
仅当查询完全无法从历史获得任何线索、且确属参数化事实查询时才追问。

【输出要求】
仅输出 JSON，不要任何解释文字：
{
  "query_types": ["命中类型1", "命中类型2"],
  "need_clarification": true/false,
  "clarification_question": "需要追问用户的问题（need_clarification 为 false 时填空字符串）",
  "rewritten_query": "初步改写后的查询（多意图型填子问题数组，其余填字符串；无需改写则填原句）",
  "confidence": 0.0-1.0
}

【对话历史】
{history}

【当前查询】
{query}"""

_STEP2_PROMPTS = {
    "context_dependent": """你是一个智能的查询优化助手。请分析用户的当前问题以及前序对话历史，判断当前问题是否依赖上下文。

规则：
1. 若依赖：把当前问题改写为独立、完整的问题，显式补全主语、宾语和上文已确认的关键条件
   （如园区名称、人群、日期）。补全的信息必须来自对话历史，禁止臆造。
2. "还有/其他/也"类追问：改写时保留上文的限定条件，并将诉求改写为"除已提过的 X、Y 之外还有哪些 Z"。
3. 若不依赖：原样返回，不做任何改动。
4. 只输出改写后的最终问题，不要解释，不要输出"原问题/改写后"等标签。

【对话历史】
{history}

【当前问题】
{query}""",
    "comparison": """你是一个查询分析专家。请结合对话上下文，识别问题中需要比较的多个对象，并把原始问题改写成
更明确、更适合知识库检索的对比性查询。

规则：
1. 显式列出全部对比对象（如"快速通行证 vs 迪士尼尊享卡"）；若某对象只在上文出现过，从历史上文补全。
2. 若用户未指明对比维度，在改写中补全常见维度（价格、适用人群、限制条件、使用方式），
   写成："A 和 B 在价格、适用人群、限制条件上分别有什么区别？"
3. 对比对象不超过 3 个；超过时保留用户明确提到的前 3 个。
4. 只输出改写后的查询，不要解释。

【对话历史】
{history}

【当前问题】
{query}""",
    "pronoun": """你是一个消除语言歧义的专家。请分析用户的当前问题和对话历史，找出"它"、"他们"、"这个"、
"那里"、"都"等指代词具体指向的对象，并替换为明确名称。

规则：
1. 指代对象必须来自对话历史或当前问题本身，禁止猜测历史上不存在的实体。
2. "都"指代集合时，展开为"X、Y 和 Z 都……"或改写为关于该集合整体的问句。
3. 若某个指代词在历史上找不到对应对象，保留原词。
4. 只输出消歧后的新问题，不要解释。

【对话历史】
{history}

【当前问题】
{query}""",
    "multi_intent": """你是一个任务分解机器人。请将用户的复杂问题分解成多个相互独立、可单独回答的简单问题。

规则：
1. 每个子问题必须是完整问句，自带主语，不依赖其他子问题即可理解。
2. 子问题之间是并列关系才拆分；若后一问题是对前一问题的追问限定（如"门票多少钱？儿童有优惠吗？"中
   "儿童"修饰"门票"），则合并为"儿童门票有优惠吗？"这类带限定的子问题。
3. 最多拆 {max_sub} 个；超出时合并语义相近的。
4. 子问题共有的上下文限定词（园区、日期等）要复制到每个子问题中。
5. 只输出 JSON 数组，不要解释。

示例：
原始查询：门票多少钱？需要提前预约吗？停车费怎么收？
输出：["上海迪士尼门票多少钱？", "上海迪士尼门票需要提前预约吗？", "上海迪士尼停车费怎么收？"]

【对话历史】
{history}

【原始查询】
{query}""",
    "rhetorical": """你是一个沟通理解大师。请分析用户的反问或带情绪的陈述，识别其背后真实的意图和诉求，
改写成中立、客观、可直接用于知识库检索的问题。

规则：
1. 剥离情绪，只保留事实诉求。例："怎么又要排队两小时！" → 真实诉求是排队时间/免排队方法。
2. 若反问暗含对规则的质疑（"凭什么员工票不能转让"），改写为该规则本身的查询
   （"迪士尼员工票的使用规则和转让限制"）。
3. 若陈述隐含求助（"带孩子根本玩不了几个项目"），改写为对应的攻略类查询
   （"带儿童游玩上海迪士尼的项目推荐和攻略"）。
4. 只输出改写后的中立问句，不要复述情绪，不要解释。

【对话历史】
{history}

【用户输入】
{query}""",
}

# 类型执行优先级（从高到低）：信息不足已由 Step1 直接短路，不进入此链
_TYPE_PRIORITY = ["multi_intent", "rhetorical", "context_dependent", "pronoun", "comparison"]


def _format_history(history):
    """history: [{"role": "user"/"assistant", "content": ...}] → 纯文本。"""
    if not history:
        return "无"
    lines = []
    for m in history:
        role = "用户" if m.get("role") == "user" else "助手"
        lines.append(f"{role}：{m.get('content', '')[:200]}")
    return "\n".join(lines)


def _intent_llm(prompt, max_tokens=400, model=None):
    """统一走 AGICTO OpenAI 兼容接口；失败抛异常由调用方降级。"""
    from openai import OpenAI
    cli = OpenAI(api_key=REWRITE_API_KEY, base_url=AGICTO_BASE_URL)
    resp = cli.chat.completions.create(
        model=model or REWRITE_MODEL, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content.strip()


def _extract_json(text):
    """从 LLM 输出中提取首个 JSON 对象/数组，容错 markdown 代码块包裹。"""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    for i, ch in enumerate(text):
        if ch in "{[":
            close = "}" if ch == "{" else "]"
            depth = 0
            for j in range(i, len(text)):
                if text[j] == ch:
                    depth += 1
                elif text[j] == close:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[i:j + 1])
                        except Exception:
                            break
    return None


# Step1 类型 LLM 偶尔回显 prompt 里的中文名，统一归一到英文枚举（N5 评测发现 6/80 泄漏）
_TYPE_ALIASES = {
    "信息不足型": "underspecified", "上下文依赖型": "context_dependent",
    "对比型": "comparison", "模糊指代型": "pronoun",
    "多意图型": "multi_intent", "反问/情绪型": "rhetorical",
    "反问型": "rhetorical", "情绪型": "rhetorical", "明确型": "clear",
}
_VALID_TYPES = set(_TYPE_PRIORITY) | {"underspecified", "clear"}


def _normalize_type(t):
    t = str(t).strip()
    # 兼容 "信息不足型(underspecified)" / "underspecified（信息不足型）" 混写
    m = re.search(r"[a-z_]{4,}", t)
    if m and m.group(0) in _VALID_TYPES:
        return m.group(0)
    t = t.split("(")[0].split("（")[0].strip()
    return _TYPE_ALIASES.get(t)


def analyze_intent(query, history=None, model=None):
    """Step 1 意图识别。返回规范化 dict；失败返回 None（调用方应降级为原 query）。"""
    # 模板内含 JSON 示例花括号，用 replace 而非 .format 注入
    prompt = (_INTENT_STEP1_PROMPT
              .replace("{history}", _format_history(history))
              .replace("{query}", query))
    raw = _intent_llm(prompt, max_tokens=400, model=model)
    data = _extract_json(raw)
    if not isinstance(data, dict):
        return None
    types = data.get("query_types") or []
    if isinstance(types, str):
        types = [types]
    norm_types, seen = [], set()
    for t in types:
        nt = _normalize_type(t) if isinstance(t, str) else None
        if nt and nt not in seen:
            norm_types.append(nt)
            seen.add(nt)
    return {
        "query_types": norm_types,
        "need_clarification": bool(data.get("need_clarification", False)),
        "clarification_question": str(data.get("clarification_question", "") or "").strip(),
        "rewritten_query": data.get("rewritten_query", query),
        "confidence": float(data.get("confidence", 0.0) or 0.0),
    }


def _apply_step2(query, qtype, history, model=None, max_sub=4):
    """对单条 query 套用某类型的二级改写 prompt，返回改写文本（失败返回原文）。"""
    tpl = _STEP2_PROMPTS.get(qtype)
    if not tpl:
        return query
    try:
        prompt = tpl.format(history=_format_history(history), query=query, max_sub=max_sub)
        out = _intent_llm(prompt, max_tokens=300, model=model)
        if qtype == "multi_intent":
            arr = _extract_json(out)
            if isinstance(arr, list) and arr:
                return [str(x).strip() for x in arr if str(x).strip()][:max_sub] or query
            return query
        return out.strip() or query
    except Exception as e:
        print(f"[intent step2 {qtype} 降级] {e}")
        return query


# ---------------------------------------------------------------- 意图规则门（确定性，不调模型）
# P1-7（2026-10-02，金标 eval_intent_gold.json 驱动）：
#   "多少钱/几点开始/在哪买/怎么预约" 这类无主体极短问，LLM 一律误判为可直接检索。
#   规则：无会话历史 + ≤8 字 + 剥离疑问/功能词后零内容残留 -> 直接 clarify。
#   反向护栏（金标+QA 共 106 问 dry-run 零误伤）：
#     1) 有历史不触发——"那儿童票呢"(5字)是承接性追问，须走指代消解；
#     2) 含领域实体词（门票/酒店/迪士尼…）不触发；
#     3) "天生一对是哪一年"(7字)剥离后仍有"天生对"残留，不触发。
SHORT_GATE_MAX_LEN = 8
_GATE_FUNC_WORDS = [
    "请问", "有没有", "多少钱", "几点", "什么时候", "怎么", "怎样", "如何",
    "什么", "多少", "哪里", "哪儿", "哪个", "哪些", "为啥", "为什么",
    "可以", "能不", "能", "会", "需要", "预约", "购买", "买", "开始", "结束",
    "时间", "位置", "地方", "吗", "呢", "啊", "呀", "吧", "嘛", "哦",
    "的", "是", "有", "在", "和", "与", "及", "或", "还", "也", "都", "就",
    "要", "想", "该", "这", "那", "个", "些", "一", "几", "点", "多", "少",
    "钱", "票", "哪", "谁", "咋",
]
_GATE_ENTITY_HINTS = [
    "门票", "儿童票", "成人票", "员工票", "年卡", "年票", "尊享卡", "通行证",
    "酒店", "餐厅", "烧烤", "停车", "车场", "乐园", "迪士尼", "园区", "烟花",
    "邮轮", "项目", "矿山车", "漂流", "光轮", "气球", "演出", "排队",
]
_GATE_PUNCT = "？?，。、,.~!！:：;；“”‘’\"'()（）-—_/\\ \t\n"
SHORT_GATE_CLARIFY = (
    "您的问题信息有点少，为了更准确地帮到您，能补充一下具体对象吗？"
    "例如：想咨询的是哪个乐园（上海/香港/东京等）的门票价格、项目身高限制、"
    "演出时间、酒店房型还是停车收费？")


def _rule_gate_short_no_entity(query, history):
    """无历史极短无实体问 -> clarify；其余返回 None 交给 LLM。"""
    if history:  # 承接性追问须走指代消解
        return None
    q = (query or "").strip()
    if not q or len(q) > SHORT_GATE_MAX_LEN:
        return None
    if any(h in q for h in _GATE_ENTITY_HINTS):
        return None
    s = q
    for w in _GATE_FUNC_WORDS:
        s = s.replace(w, "")
    for ch in _GATE_PUNCT:
        s = s.replace(ch, "")
    if s == "":
        return {"action": "clarify", "clarify": SHORT_GATE_CLARIFY,
                "queries": [query], "query_types": ["underspecified"],
                "multi_intent": False, "effective": query, "confidence": 1.0,
                "rule_gate": "short_no_entity"}
    return None


def _rule_split_multi_question(query, min_segments=3):
    """按问号切分并列问句；片段数 >= min_segments 时返回列表，否则 None。
    只认问号不认感叹号——反问/吐槽（"也太坑了吧！"）不触发。"""
    parts = re.split(r"[？?]", query or "")
    segs = [s.strip(_GATE_PUNCT) for s in parts]
    segs = [s for s in segs if s]
    return segs if len(segs) >= min_segments else None


def intent_rewrite(query, history=None, model=None, max_sub_queries=4,
                   confidence_min=0.6):
    """
    意图识别 + 改写流水线入口。
    返回 dict:
      action        : "clarify" | "retrieve"
      clarify       : 追问话术（action=clarify 时）
      queries       : 检索用子问题列表（action=retrieve 时，至少 1 条）
      query_types   : Step1 识别出的类型
      multi_intent  : 是否多意图拆分
      effective     : 单查询场景下的改写结果（用于展示）
      confidence    : Step1 置信度
      rule_gate     : 命中确定性规则门时的门名（审计用），否则无此键
    任何内部失败都降级为 {"action": "retrieve", "queries": [query]}。
    """
    fallback = {"action": "retrieve", "clarify": "", "queries": [query],
                "query_types": [], "multi_intent": False, "effective": query,
                "confidence": 1.0}
    # 规则门 1：无历史的极短无实体问（"多少钱/几点开始/在哪买"），LLM 此前一律误判直接检索。
    # 有历史时不触发——"那儿童票呢"这类承接性追问需走 LLM 指代消解（金标已验证）。
    gated = _rule_gate_short_no_entity(query, history)
    if gated is not None:
        print("[intent 规则门] 短问无实体 -> clarify")
        return gated
    try:
        intent = analyze_intent(query, history=history, model=model)
    except Exception as e:
        print(f"[intent step1 降级] {e}")
        return fallback
    if not intent or intent["confidence"] < confidence_min:
        print(f"[intent] 置信度不足或识别失败，按原 query 检索")
        return fallback

    types = list(intent["query_types"])
    # 规则门 2：≥3 个并列问句而 LLM 未判多意图（三连问漏判兜底），按问号确定性切分
    rule_multi = False
    if "multi_intent" not in types:
        segs = _rule_split_multi_question(query, min_segments=3)
        if segs:
            print(f"[intent 规则门] 三连问漏判 -> 强制多意图拆分 {len(segs)} 段")
            types.append("multi_intent")
            intent["rewritten_query"] = segs  # 直接喂多意图路径，跳过 LLM 再拆
            rule_multi = True
    hit = [t for t in _TYPE_PRIORITY if t in types]
    # 信息不足型：仅当查询无可改写类型（无指代/上下文依赖/多意图等）时才追问。
    # 若同时命中可改写类型，先走改写链——指代消解后通常就不再信息不足，
    # 强行追问反而会打断指代链（如"你刚提到的人物在哪见"被误追问）。
    if (intent["need_clarification"] and intent["clarification_question"]
            and not hit):
        return {"action": "clarify", "clarify": intent["clarification_question"],
                "queries": [query], "query_types": types, "multi_intent": False,
                "effective": query, "confidence": intent["confidence"]}

    if not hit:
        return fallback  # 明确型

    # 多意图型：先拆分，再对每个子问题串行走剩余改写链
    if "multi_intent" in hit:
        subs = intent.get("rewritten_query")
        if not (isinstance(subs, list) and len(subs) >= 2):
            subs = _apply_step2(query, "multi_intent", history, model, max_sub_queries)
        if isinstance(subs, str):
            subs = [subs]
        rest = [t for t in hit if t != "multi_intent"]
        out_qs = []
        for sq in subs[:max_sub_queries]:
            cur = str(sq).strip()
            for t in rest:
                cur = _apply_step2(cur, t, history, model, max_sub_queries)
                if isinstance(cur, list):  # 链中不应再出数组，防御性拍平
                    cur = " ".join(str(x) for x in cur)
            if cur:
                out_qs.append(cur)
        out_qs = out_qs or [query]
        return {"action": "retrieve", "clarify": "", "queries": out_qs,
                "query_types": types, "multi_intent": True,
                "effective": " | ".join(out_qs), "confidence": intent["confidence"],
                **({"rule_gate": "multi_question_ge3"} if rule_multi else {})}

    # 单意图（可多类型串联）
    cur = query
    for t in hit:
        cur = _apply_step2(cur, t, history, model, max_sub_queries)
        if isinstance(cur, list):
            cur = " ".join(str(x) for x in cur)
    cur = (cur or query).strip()
    return {"action": "retrieve", "clarify": "", "queries": [cur],
            "query_types": types, "multi_intent": False,
            "effective": cur, "confidence": intent["confidence"]}


# ---------------------------------------------------------------- 综合检索器

class HybridRetriever:
    """向量 + BM25 混合检索，可选 rerank。"""
    def __init__(self, faiss_index, metadata, bm25=None,
                 use_bm25=True, use_rerank=False, rerank_topn=10):
        self.index = faiss_index
        self.meta = metadata
        self.bm25 = bm25
        self.use_bm25 = use_bm25 and bm25 is not None
        self.use_rerank = use_rerank
        self.rerank_topn = rerank_topn

    def _vec_search(self, query_vec, k):
        v = np.array([query_vec], dtype="float32")
        scores, idxs = self.index.search(v, k)
        # IndexFlatIP: score 即余弦相似度 ([-1,1])
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]

    def _is_valid(self, m):
        """R4 前置过滤：跳过过短且非表格的无意义 chunk。"""
        content = m.get("content", "")
        body = content.split("\n", 1)[-1] if "\n" in content else content
        if len(body) < 8 and "|" not in body:
            return False
        return True

    def search(self, query_vec, query_text, k=5,
               use_rerank=None, rerank_query=None,
               allowed_idxs=None, max_per_source=None, use_mmr=False,
               mmr_lambda=None, bm25_weight=None, on_stage=None):
        """
        allowed_idxs   : 类别预过滤（P1）——只在给定候选集内召回，保证 top_k 取满
        max_per_source : 同源限额（P1）——单一文件最多返回 N 条，避免冗余
        use_mmr        : MMR 多样性选择（P1）
        bm25_weight   : BM25 在加权 RRF 中的权重份额 [0,1]（向量路=1-w），
                         None 时取 settings.BM25_WEIGHT（默认 0.5 等权）
        on_stage      : 阶段回调，重排前回调 on_stage("rerank")，供前端展示真实进度
        """
        do_rerank = self.use_rerank if use_rerank is None else use_rerank
        bm_w = BM25_WEIGHT if bm25_weight is None else bm25_weight
        bm_w = min(1.0, max(0.0, float(bm_w)))
        allowed = set(int(a) for a in allowed_idxs) if allowed_idxs is not None else None
        # 向量路（类别约束下扩大召回池，保证过滤后仍能取满）
        vec_k = max(k * 6, 40) if allowed is not None else max(k * 4, 20)
        vec_ranks = self._vec_search(query_vec, vec_k)
        # 保留余弦相似度用于展示（RRF/rerank 分不直观）
        cos_map = {i: s for i, s in vec_ranks}

        if allowed is not None:
            vec_ranks = [(i, s) for i, s in vec_ranks if i in allowed]

        # BM25 路
        if self.use_bm25:
            exp_text = expand_query_terms(query_text)
            bm_ranks = self.bm25.search(exp_text, topk=vec_k)
            if allowed is not None:
                bm_ranks = [(i, s) for i, s in bm_ranks if i in allowed]
            fused = rrf_fuse(vec_ranks, bm_ranks,
                              topk=self.rerank_topn if do_rerank else max(k * 3, 12),
                              weights=[1.0 - bm_w, bm_w])
        else:
            fused = [(i, s) for i, s in vec_ranks[:k]]

        # R4 过滤
        with _STATS_LOCK:
            POOL_STATS["queries"] += 1
            POOL_STATS["rrf_out"] += len(fused)
        fused = [(i, s) for i, s in fused if self._is_valid(self.meta[i])]
        if len(fused) < k:
            seen = {i for i, _ in fused}
            for i, s in vec_ranks:
                if i not in seen and self._is_valid(self.meta[i]):
                    fused.append((i, s))
                    seen.add(i)
                    if len(fused) >= k:
                        break
        with _STATS_LOCK:
            POOL_STATS["after_r4"] += len(fused)

        # R2 重排
        if do_rerank and len(fused) > 1:
            cands = [self.meta[i].get("content", "") for i, _ in fused]
            rr_info = {}
            try:
                if on_stage:
                    on_stage("rerank")
                rr = qwen_rerank(rerank_query or query_text, cands, topk=k,
                                 info=rr_info)
                if not rr:  # 双保险：空重排结果视为失败，走降级并计数，禁止静默空转
                    raise RuntimeError("rerank 返回空结果")
                # 主模型失败、备用模型成功：标记"已降级备用模型"（不改变阶段推进）
                if on_stage and rr_info.get("degraded"):
                    on_stage("rerank", {"degraded": True,
                                        "model": rr_info.get("model", ""),
                                        "source": rr_info.get("source", "")})
                reranked = [fused[orig_idx][0] for orig_idx, _ in rr]
                in_rr = set(reranked)
                for i, _ in fused:
                    if i not in in_rr:
                        reranked.append(i)
                order = reranked
            except Exception as e:
                with _STATS_LOCK:
                    RERANK_STATS["fallbacks"] += 1
                    n_fb = RERANK_STATS["fallbacks"]
                print(f"[rerank 降级#{n_fb}] {e}")
                # 整条降级链全失败：告知前端"重排不可用，按检索序返回"
                if on_stage:
                    on_stage("rerank", {"failed": True})
                order = [i for i, _ in fused]
        else:
            order = [i for i, _ in fused]
        with _STATS_LOCK:
            POOL_STATS["rerank_out"] += len(order)

        # P1: MMR 多样性（余弦重排选择，兼顾与查询相关性和已选集合的差异度）
        if use_mmr and k >= 3:
            with _STATS_LOCK:
                POOL_STATS["mmr_in"] += len(order)
            order = self._mmr_select(query_vec, order, k,
                                    lam=0.72 if mmr_lambda is None else mmr_lambda)

        # P1: 同源限额 —— 候选池不足 k 时从全库（类别内）向量序扩展补位，
        # 优先跨源补齐，仅在无其他来源可选时才放宽限额回填同源
        if max_per_source:
            seen = set(order)
            extra = [i for i, _ in vec_ranks
                     if i not in seen and self._is_valid(self.meta[i])]
            order = self._cap_per_source(list(order) + extra, k, max_per_source)
        else:
            order = order[:k]
        with _STATS_LOCK:
            POOL_STATS["final"] += len(order[:k])
        return [(i, cos_map.get(i, 0.0)) for i in order[:k]]

    def search_multi(self, query_vecs, query_texts, k=5,
                     use_rerank=None, rerank_query=None,
                     allowed_idxs=None, max_per_source=None, use_mmr=False,
                     mmr_lambda=None, bm25_weight=None, on_stage=None):
        """多意图合并检索：逐子问题召回 → 各路加权 RRF 融合 → 统一 rerank → MMR/限额。
        与 search() 的差别仅在召回段：把 N 个子问题的向量路+BM25路全部放入同一 RRF，
        避免子问题各自截断 top_k 互相挤占。rerank_query 应传用户原始问题。
        bm25_weight 语义同 search()（None→settings.BM25_WEIGHT）。
        on_stage 语义同 search()：重排前回调 on_stage("rerank")。"""
        do_rerank = self.use_rerank if use_rerank is None else use_rerank
        bm_w = BM25_WEIGHT if bm25_weight is None else bm25_weight
        bm_w = min(1.0, max(0.0, float(bm_w)))
        allowed = set(int(a) for a in allowed_idxs) if allowed_idxs is not None else None
        vec_k = max(k * 6, 40) if allowed is not None else max(k * 4, 20)

        ranked_lists, weights, cos_map, all_vec_ranks = [], [], {}, []
        for qv, qt in zip(query_vecs, query_texts):
            vec_ranks = self._vec_search(qv, vec_k)
            cos_map.update({i: s for i, s in vec_ranks})
            if allowed is not None:
                vec_ranks = [(i, s) for i, s in vec_ranks if i in allowed]
            all_vec_ranks.append(vec_ranks)
            ranked_lists.append(vec_ranks)
            weights.append(1.0 - bm_w)
            if self.use_bm25:
                exp_text = expand_query_terms(qt)
                bm_ranks = self.bm25.search(exp_text, topk=vec_k)
                if allowed is not None:
                    bm_ranks = [(i, s) for i, s in bm_ranks if i in allowed]
                ranked_lists.append(bm_ranks)
                weights.append(bm_w)

        # 统一加权 RRF 融合所有子问题的所有路（k=60 与 rrf_fuse 保持一致）
        fused = rrf_fuse(*ranked_lists, k=60, weights=weights,
                          topk=(self.rerank_topn if do_rerank else max(k * 3, 12)))

        # R4 过滤
        fused = [(i, s) for i, s in fused if self._is_valid(self.meta[i])]
        if len(fused) < k:  # 从第一个子问题的向量序补位
            seen = {i for i, _ in fused}
            for i, s in all_vec_ranks[0] if all_vec_ranks else []:
                if i not in seen and self._is_valid(self.meta[i]):
                    fused.append((i, s))
                    seen.add(i)
                    if len(fused) >= k:
                        break

        # R2 统一重排（用原始问题，避免子问题语义偏移）
        if do_rerank and len(fused) > 1:
            cands = [self.meta[i].get("content", "") for i, _ in fused]
            rr_info = {}
            try:
                if on_stage:
                    on_stage("rerank")
                rr = qwen_rerank(rerank_query or query_texts[0], cands, topk=k,
                                 info=rr_info)
                if not rr:  # 双保险：空重排结果视为失败，走降级并计数
                    raise RuntimeError("rerank 返回空结果")
                # 主模型失败、备用模型成功：标记"已降级备用模型"
                if on_stage and rr_info.get("degraded"):
                    on_stage("rerank", {"degraded": True,
                                        "model": rr_info.get("model", ""),
                                        "source": rr_info.get("source", "")})
                reranked = [fused[orig_idx][0] for orig_idx, _ in rr]
                in_rr = set(reranked)
                for i, _ in fused:
                    if i not in in_rr:
                        reranked.append(i)
                order = reranked
            except Exception as e:
                with _STATS_LOCK:
                    RERANK_STATS["fallbacks"] += 1
                    n_fb = RERANK_STATS["fallbacks"]
                print(f"[rerank 降级#{n_fb}] {e}")
                # 整条降级链全失败：告知前端"重排不可用，按检索序返回"
                if on_stage:
                    on_stage("rerank", {"failed": True})
                order = [i for i, _ in fused]
        else:
            order = [i for i, _ in fused]

        # MMR 用第一个子问题的向量（主意图）
        if use_mmr and k >= 3:
            order = self._mmr_select(query_vecs[0], order, k,
                                     lam=0.72 if mmr_lambda is None else mmr_lambda)

        if max_per_source:
            seen = set(order)
            extra = []
            for vr in all_vec_ranks:
                for i, _ in vr:
                    if i not in seen and self._is_valid(self.meta[i]):
                        extra.append(i)
                        seen.add(i)
            order = self._cap_per_source(list(order) + extra, k, max_per_source)
        else:
            order = order[:k]
        return [(i, cos_map.get(i, 0.0)) for i in order[:k]]

    def _mmr_select(self, query_vec, pool, k, lam=0.72):
        """MMR 贪心选择。pool: 按相关性排序的 idx 列表（rerank/RRF 序即最强相关性先验）。
        相关性项用池内位次（继承 rerank 排序信号），多样性项用向量余弦。
        旧实现用原始余弦做相关性，会把 rerank 排序整体丢弃——新空间下纯向量信号弱
        （诊断实测：MMR-only recall@5 0.884→0.407），调 λ 无法补救。"""
        pool = [int(i) for i in pool]
        if len(pool) <= k:
            return pool
        try:
            vecs = np.array([self.index.reconstruct(i) for i in pool], dtype="float32")
            n = len(pool)
            rel = [(n - j) / n for j in range(n)]  # 位次相关性：1.0 → 1/n 线性衰减
            chosen, rest = [], list(range(n))
            while rest and len(chosen) < k:
                if not chosen:
                    scores = {j: rel[j] for j in rest}
                else:
                    sel = vecs[chosen]
                    maxsim = (vecs[rest] @ sel.T).max(axis=1)
                    scores = {j: lam * rel[j] - (1 - lam) * float(maxsim[t])
                              for t, j in enumerate(rest)}
                best = max(scores, key=scores.get)
                chosen.append(best)
                rest.remove(best)
            return [pool[j] for j in chosen]
        except Exception as e:
            print(f"[MMR 降级] {e}")
            return pool[:k]

    def _cap_per_source(self, ordered, k, cap):
        """单一来源最多 cap 条；不足 k 时放宽限制补齐。"""
        if not cap or cap < 1:
            return ordered[:k]
        kept, cnt, rest = [], {}, []
        for i in ordered:
            src = self.meta[int(i)].get("source", "")
            if cnt.get(src, 0) < cap:
                kept.append(i)
                cnt[src] = cnt.get(src, 0) + 1
            else:
                rest.append(i)
            if len(kept) >= k:
                return kept
        for i in rest:
            if len(kept) >= k:
                break
            kept.append(i)
        return kept


# ---------------------------------------------------------------- 上下文构建（生成侧 G1）
def merge_overlap(a, b, max_check=400):
    """合并两段上下文，去除 a 尾部与 b 头部的重叠（parent_context 邻块重复）。"""
    if not a:
        return b
    if not b:
        return a
    for L in range(min(len(a), len(b), max_check), 19, -1):
        if a.endswith(b[:L]):
            return a + b[L:]
    return a + "\n" + b


def build_context(ranked, meta):
    """I1+N10: 同源 chunk 合并去重；每个来源编号 [n]，供模型行内引用。
    线上服务与评估"线上同 prompt"档共用此函数（meta 显式传入），保证上下文格式同源。
    返回 (context_str, refs[{n,source,heading}])。"""
    by_src = OrderedDict()
    for i, s in ranked:
        m = meta[i]
        key = m.get("source", "")
        info = by_src.setdefault(key, {"ctxs": [], "headings": []})
        info["ctxs"].append(m.get("parent_context", m["content"]))
        h = m.get("heading", "")
        if h and h not in info["headings"]:
            info["headings"].append(h)
    blocks, refs = [], []
    for n, (src, info) in enumerate(by_src.items(), 1):
        merged = info["ctxs"][0]
        for c in info["ctxs"][1:]:
            merged = merge_overlap(merged, c)
        head = f"章节: {'; '.join(info['headings'][:3])}" if info["headings"] else "章节: 无"
        blocks.append(f"[{n}] 来源: {src} | {head}\n{merged}")
        refs.append({"n": n, "source": src,
                     "heading": "; ".join(info["headings"][:3]) if info["headings"] else ""})
    return "\n\n".join(blocks), refs

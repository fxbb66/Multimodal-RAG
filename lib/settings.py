# -*- coding: utf-8 -*-
"""
集中配置（N1 密钥收敛 / P1 参数化）
==================================
所有密钥与可调超参统一从环境变量读取；首次导入时自动加载 7-完整知识库版本/.env。
代码中禁止再硬编码任何密钥。配置缺失时 fail-fast，提示复制 .env.example。
"""
import os
import sys

ROOT7 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_dotenv():
    """零依赖 .env 解析：KEY=VALUE，支持引号与 # 注释；不覆盖已有环境变量。"""
    path = os.path.join(ROOT7, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def _bool(name, default):
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes", "on")


def require(name):
    v = os.environ.get(name, "").strip()
    if not v:
        sys.exit(f"[配置错误] 缺少环境变量 {name}。请复制 .env.example 为 .env 并填入密钥（见 README）。")
    return v


# ---------------------------------------------------------------- 密钥
# DashScope（保留用于 rerank 降级兜底，不再用于 embedding/gen/rewrite/vl）
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "").strip()

# AGICTO 统一平台
AGICTO_BASE_URL = os.environ.get("AGICTO_BASE_URL", "https://api.agicto.cn/v1")
RERANK_API_KEY = os.environ.get("RERANK_API_KEY", "").strip()
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "").strip()
GEN_API_KEY = os.environ.get("GEN_API_KEY", "").strip()
REWRITE_API_KEY = os.environ.get("REWRITE_API_KEY", "").strip()
VL_API_KEY = os.environ.get("VL_API_KEY", "").strip()

# ---------------------------------------------------------------- 模型
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "tongyi-embedding-vision-plus")
# 向量维度（tongyi-embedding-vision-plus=1152，text-embedding-v3=1024）。
# 换模型时必须同步修改并删除 embeddings.npy/*.faiss 重建，启动时会校验索引维度。
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1152"))
GEN_MODEL = os.environ.get("DISNEY_GEN_MODEL", "qwen3.7-plus")
REWRITE_MODEL = os.environ.get("REWRITE_MODEL", "qwen3.8-flash")
VL_CAPTION_MODEL = os.environ.get("VL_CAPTION_MODEL", "qwen3.5-omni-flash")

# 百炼（DashScope）OpenAI 兼容端点：前端可切换回答生成平台。
# AGICTO 的 qwen3.7-plus 是平台别名，百炼侧对应模型名为 qwen-plus（可用 .env 覆盖）。
DASHSCOPE_BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DASHSCOPE_GEN_MODEL = os.environ.get("DASHSCOPE_GEN_MODEL", "qwen-plus")

# 线上生成侧 system prompt 单一真源（frontend/app.py 与 4-evaluate.py 线上同 prompt 评估档共用）。
# 修改即改变线上回答行为与评估口径，改动后须重跑主评估"线上同prompt"档留证。
ONLINE_SYSTEM_PROMPT = (
    "你是上海迪士尼度假区的官方客服助手。回答必须严格遵守以下规则：\n"
    "1. 只能依据给出的【背景知识】回答，不得编造；若背景知识不足，明确告知用户。\n"
    "2. 涉及价格、时间、身高限制、政策等数值，必须准确引用，禁止估算。\n"
    "3. 每个事实性结论后用 [n] 行内标注其来源编号（n 对应背景知识中的编号），"
    "一句话涉及多个来源时标注多个编号，如 [1][3]。\n"
    "4. 票价、流程对比等结构化信息优先使用 Markdown 表格；表格单元格中也要带来源编号。\n"
    "5. 语气专业友好，符合迪士尼服务标准，回答简洁有条理。"
)

# rerank 降级链（逗号分隔，按优先级从高到低）
# 2026-10-02 起首选 qwen3-rerank：rerank-v3.5（Cohere）所在账户长期欠费，
# 置于链首每次必失败再降级、纯增延迟；AGICTO 整体不可用时由百炼 qwen3-rerank 兜底。
# 如 Cohere 账户恢复且想重新优先，用 .env 设 RERANK_MODELS=rerank-v3.5,qwen3-rerank 即可。
RERANK_MODELS = [m.strip() for m in os.environ.get(
    "RERANK_MODELS", "qwen3-rerank").split(",") if m.strip()]
# DashScope 兜底链（Cohere 系列模型百炼无此模型，只走千问系列）
_COHERE_RERANK = {"rerank-v3.5", "rerank-multilingual-v3.0"}
RERANK_DASHSCOPE_MODELS = [m for m in RERANK_MODELS if m not in _COHERE_RERANK]

# ---------------------------------------------------------------- 检索超参
# 2026-10-02 于 tongyi-embedding-vision-plus(1152维) 新向量空间重标定
# （QA n=82/k=5，产物 output/diag_calib.json，index=5c879a20569d）：
#   限额 cap: 1→Recall 0.756，2→0.915，3→0.951，不限→0.951；cap=3 与无限额持平且保留单源护栏
#   MMR λ(cap=2): 0.40→0.854，0.55→0.902，0.72→0.915，0.85→0.915（MRR 0.760→0.764，差异不显著）
RERANK_TOPN = int(os.environ.get("RERANK_TOPN", "12"))            # 候选池=rerank 池，pool 流水 rrf_out 恒为 12
MAX_PER_SOURCE = int(os.environ.get("MAX_PER_SOURCE", "3"))       # 重标定：2→3（消除全部 strict 损失）
USE_MMR = _bool("USE_MMR", True)
MMR_LAMBDA = float(os.environ.get("MMR_LAMBDA", "0.72"))          # 重标定维持：0.72 已在平台，0.55/0.40 更差
# 多意图查询的 MMR λ（2026-10-02）：多意图召回的候选来自不同子问题，
# 跨意图冗余风险高于单意图，λ 下调以增强去重。0.55 在【单意图】QA 集上更差
# （0.902 vs 0.915），故仅用于多意图路径；待多意图金标集验证后再定标。
MMR_LAMBDA_MULTI = float(os.environ.get("MMR_LAMBDA_MULTI", "0.55"))

# ---------------------------------------------------------------- RRF 融合权重
# BM25 在加权 RRF 中的权重份额 [0,1]，向量路权重 = 1 - BM25_WEIGHT。
# 默认 0.5 = 等权 RRF（与 2026-10-02 标定并在线上运行的行为完全一致）。
# 注意：历史"0.3/0.7 硬编码"是旧加权分数方案的死参数，早已移除；无评估证据不改默认。
# 查询时可在请求体传 bm25_weight 做单查询临时调整。
BM25_WEIGHT = float(os.environ.get("BM25_WEIGHT", "0.5"))
MEDIA_SIM_THRESHOLD = float(os.environ.get("MEDIA_SIM_THRESHOLD", "0.45"))  # 待标定：尚无图片金标 query 集
MEDIA_GAP = float(os.environ.get("MEDIA_GAP", "0.02"))            # 待标定：同上，需图片评估集支撑

# ---------------------------------------------------------------- 评估
# 4-evaluate.py 检索/答案侧评估的线程池并发数；AGICTO 限流明显时调低（如 4）
EVAL_WORKERS = int(os.environ.get("EVAL_WORKERS", "8"))

# ---------------------------------------------------------------- 意图识别（R3 v2）
INTENT_ENABLED = _bool("INTENT_ENABLED", True)                    # 总开关：False 时检索一律用原 query
INTENT_CONFIDENCE_MIN = float(os.environ.get("INTENT_CONFIDENCE_MIN", "0.6"))  # 低于此置信度用原 query
INTENT_MAX_CLARIFY = int(os.environ.get("INTENT_MAX_CLARIFY", "2"))            # 单会话连续追问上限
INTENT_MAX_SUB_QUERIES = int(os.environ.get("INTENT_MAX_SUB_QUERIES", "4"))    # 多意图最多拆几个子问题

# ---------------------------------------------------------------- 服务
SESSION_MAX = int(os.environ.get("SESSION_MAX", "200"))
SESSION_TTL = int(os.environ.get("SESSION_TTL", "7200"))
EMBED_CACHE_SIZE = int(os.environ.get("EMBED_CACHE_SIZE", "256"))
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "30"))

# ---------------------------------------------------------------- 产物路径（模型指纹隔离）
_OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
# 图片 OCR+caption 缓存按 caption 模型名分文件，换模型后旧产物自动隔离不被误用
_SAFE_VL_MODEL = "".join(c if c.isalnum() else "_" for c in VL_CAPTION_MODEL)
IMAGE_TEXTS_FILE = os.path.join(_OUTPUT_DIR, f"image_texts_{_SAFE_VL_MODEL}.json")
LEGACY_IMAGE_TEXTS_FILE = os.path.join(_OUTPUT_DIR, "image_texts.json")  # 无指纹旧版，仅用于一次性迁移

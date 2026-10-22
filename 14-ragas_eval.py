# -*- coding: utf-8 -*-
"""
S5: RAGAS 三指标独立评估（2026-10-20）
====================================
在独立隔离 venv（../.venv-eval，ragas 0.4.3 + langchain-community 0.3.31）中运行，
与生产 .venv 零共享——RAGAS 核心依赖硬含 4 个 langchain 包（见 8-/LangChain接入评估
与方案报告.md），绝不能进生产环境。本脚本不得用生产 .venv 运行（import 即失败）。

评估对象：eval_results_183_final.json 中"答案(线上同prompt)"档的逐题明细
（4-evaluate.py evaluate_answers 的 details 字段，2026-10-20 起仅 online 档落盘）。
若 results 文件缺该档明细，先跑最终版 4-evaluate.py 全量重跑。

三指标（RAGAS 官方口径，均 0-1 越高越好）：
  faithfulness        答案中每条论断能否被检索上下文支持（语句分解 + NLI）
  answer_relevancy    答案与问题的相关度（反向生成 3 个问题 + embedding 相似度）
  context_precision   检索上下文相对金标答案的精准度（逐块 LLM 判定 + 位次加权）

交叉验证（本项目 answer_hit 口径 vs RAGAS 口径，Spearman 秩相关）：
  faithfulness / answer_relevancy  ↔  逐题 hit（answer_hit 0/1）
  context_precision               ↔  逐题 evidence（金标证据是否召到 0/1）
判据：|rho| ≥ 0.7 视为两把尺子同向；显著低于 0.7 时逐条人工复核分歧样本，
不得直接采信任一口径的单方面结论。

用法（必须用隔离 venv 的解释器）：
    ../.venv-eval/Scripts/python.exe 14-ragas_eval.py \
        [--results output/eval_results_183_final.json] \
        [--out output/ragas_eval_183.json] [--limit N] [--workers 6] [--force]
输出：output/ragas_eval_183.json（逐题分数 + 汇总 + 交叉验证，可断点续跑）
"""
import argparse
import asyncio
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv():
    """零依赖 .env 解析（与 lib/settings.py 同规则）：不覆盖已有环境变量。"""
    path = os.path.join(HERE, ".env")
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

import requests  # noqa: E402  仅用于失败时的连通性自检（AGICTO 端点探活）

from openai import AsyncOpenAI  # noqa: E402
from ragas.embeddings.base import embedding_factory  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics.collections import (  # noqa: E402
    AnswerRelevancy, ContextPrecision, Faithfulness)

AGICTO_BASE_URL = os.environ.get("AGICTO_BASE_URL", "https://api.agicto.cn/v1").rstrip("/")
GEN_API_KEY = os.environ.get("GEN_API_KEY", "").strip()
GEN_MODEL = os.environ.get("DISNEY_GEN_MODEL", "qwen3.7-plus")
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "").strip()
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "tongyi-embedding-vision-plus")
# 评估通道（judge 侧）可独立于生产通道：AGICTO 额度耗尽/抖动时，用
# RAGAS_LLM_BASE_URL/RAGAS_LLM_API_KEY/RAGAS_LLM_MODEL 指向 DashScope
# 兼容端点（qwen-plus 等）。被评答案仍来自生产通道（qwen3.7-plus）——
# judge 与 generator 不同源反而降低自评偏差，但必须在产物 __meta__ 留痕。
RAGAS_LLM_BASE_URL = os.environ.get("RAGAS_LLM_BASE_URL", AGICTO_BASE_URL).rstrip("/")
RAGAS_LLM_API_KEY = os.environ.get("RAGAS_LLM_API_KEY", GEN_API_KEY).strip()
RAGAS_LLM_MODEL = os.environ.get("RAGAS_LLM_MODEL", GEN_MODEL)
# embedding 侧默认跟随 judge 通道（同端点同 key）：只覆盖 LLM 侧时不会
# 出现"LLM 走 DashScope、embedding 走欠费 AGICTO"的半降级状态
RAGAS_EMB_BASE_URL = os.environ.get("RAGAS_EMB_BASE_URL", RAGAS_LLM_BASE_URL).rstrip("/")
RAGAS_EMB_API_KEY = os.environ.get("RAGAS_EMB_API_KEY", RAGAS_LLM_API_KEY).strip()
# RAGAS answer_relevancy 专用 embedding：检索用 vision-plus 走 DashScope 非标准请求体
# （input.contents 包装），OpenAI SDK 标准格式会被 AGICTO 拒绝；relevancy 只度量"答案与
# 问题是否切题"的语义相似度，与检索向量空间无关，故评估侧换用标准格式的 v3（实测可用）。
RAGAS_EMB_MODEL = os.environ.get("RAGAS_EMB_MODEL", "text-embedding-v3")

PER_CALL_TIMEOUT = 180  # answer_relevancy/context_precision 单题上限
# faithfulness 内部 = 拆句(1) + 逐句校验(N 句串行)，长答案链最长；
# AGICTO 慢时段（实测单次 7-10s）下 180s 必超时 → 单独放宽到 600s
FAITHFULNESS_TIMEOUT = 600


# ---------------------------------------------------------------- 输入定位
def load_online_details(results_file):
    """从评估结果中取"答案(线上同prompt)"档的逐题明细。"""
    if not os.path.exists(results_file):
        raise SystemExit(
            f"[缺失] {results_file} 不存在。先跑最终版 4-evaluate.py 全量重跑"
            f"（落 eval_results_183_final.json）再评估。")
    with open(results_file, encoding="utf-8") as f:
        data = json.load(f)
    rows = [r for r in data.get("results", [])
            if r.get("prompt_style") == "online" and r.get("details")]
    if not rows:
        raise SystemExit(
            "[缺失] results[] 中无 prompt_style=online 且带 details 的答案档。"
            "4-evaluate.py 自 2026-10-20 起仅 online 档落逐题明细；"
            "旧版结果文件需用最终版重跑。")
    row = max(rows, key=lambda r: r.get("n", 0))  # 防御：同档多行取样本量最大者
    cfg = data.get("config", {})
    print(f"输入: {os.path.basename(results_file)}  档位: {row.get('label')}  "
          f"n={row.get('n')}  index_md5={cfg.get('index_md5')}")
    if cfg.get("index_md5") and cfg.get("gen_model"):
        print(f"  gen_model={cfg['gen_model']}  online_prompt_sha1="
              f"{cfg.get('online_system_prompt_sha1')}")
    return row["details"], cfg


# ---------------------------------------------------------------- 指标运行
class _FloatEncodingEmbeddings:
    """包装 OpenAI 客户端的 embeddings 资源：强制 encoding_format=float。

    新版 openai SDK 未显式指定时默认 base64，DashScope 兼容端点不支持该编码，
    返回空 data 触发 "No embedding data received"（AGICTO 无此问题，包装后两通道通用）。
    仅覆盖 embeddings；chat.completions 等经 __getattr__ 透传，async 检测不受影响。"""

    def __init__(self, inner):
        self._inner = inner

    async def create(self, **kwargs):
        kwargs.setdefault("encoding_format", "float")
        return await self._inner.create(**kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _FloatEncodingClient:
    """客户端级包装：只换 embeddings 资源，其余属性透传。"""

    def __init__(self, client):
        self._client = client
        self.embeddings = _FloatEncodingEmbeddings(client.embeddings)

    def __getattr__(self, name):
        return getattr(self._client, name)


def build_metrics():
    """构造 ragas LLM/embeddings/三指标（OpenAI 兼容端点，默认 AGICTO，可降级 DashScope）。"""
    if not RAGAS_LLM_API_KEY:
        raise SystemExit("[配置错误] 缺少 judge 侧 API key（GEN_API_KEY 或 RAGAS_LLM_API_KEY，.env）")
    gen_client = AsyncOpenAI(base_url=f"{RAGAS_LLM_BASE_URL}", api_key=RAGAS_LLM_API_KEY)
    # max_tokens 必须给足：faithfulness 判定步对长答案可拆出 30+ 条语句，
    # 每条含 statement+reason+verdict；ragas 默认 1024 会被截断（finish=length）
    # 导致 JSON 不完整 → IncompleteOutputException（实测根因）
    llm = llm_factory(RAGAS_LLM_MODEL, client=gen_client, max_tokens=8192)
    emb_client = _FloatEncodingClient(
        AsyncOpenAI(base_url=f"{RAGAS_EMB_BASE_URL}", api_key=RAGAS_EMB_API_KEY))
    embeddings = embedding_factory("openai", model=RAGAS_EMB_MODEL, client=emb_client)
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=llm),
    }


async def _run_metric(metric, name, **kwargs):
    """单指标调用：异常/超时不影响同题其他指标，返回 (value, error)。"""
    limit = FAITHFULNESS_TIMEOUT if name == "faithfulness" else PER_CALL_TIMEOUT
    try:
        res = await asyncio.wait_for(metric.ascore(**kwargs), limit)
        val = getattr(res, "value", None)
        if val is None:
            return None, "empty_result"
        return float(val), None
    except asyncio.TimeoutError:
        return None, f"timeout_{limit}s"
    except Exception as e:  # noqa: BLE001 —— 评估器故障必须隔离到单指标单题
        return None, f"{type(e).__name__}: {str(e)[:100]}"


async def eval_item(metrics, item, sem):
    """一题三指标（并发执行）；返回 {query, *, *_error}。"""
    q, ans = item["query"], item["answer"]
    ctxs = [c for c in item.get("contexts") or [] if c and c.strip()]
    out = {"query": q, "hit": bool(item.get("hit")),
           "evidence": bool(item.get("evidence"))}
    async with sem:
        coros = {
            "faithfulness": _run_metric(
                metrics["faithfulness"], "faithfulness", user_input=q,
                response=ans, retrieved_contexts=ctxs),
            # relevancy 不问上下文，只问"答案是否切题"（ascore 无 retrieved_contexts 参数）
            "answer_relevancy": _run_metric(
                metrics["answer_relevancy"], "answer_relevancy", user_input=q,
                response=ans),
            "context_precision": _run_metric(
                metrics["context_precision"], "context_precision", user_input=q,
                reference=item.get("gold_answer") or "", retrieved_contexts=ctxs),
        }
        names = list(coros.keys())
        results = await asyncio.gather(*coros.values())
        for name, (val, err) in zip(names, results):
            out[name] = val
            if err:
                out[f"{name}_error"] = err
    return out


# ---------------------------------------------------------------- 交叉验证
def _rank_with_ties(vals):
    """平均秩（1-based）。"""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    """Spearman 秩相关；优先 scipy，缺失则内置平均秩+Pearson 实现。"""
    pairs = [(x, y) for x, y in zip(xs, ys)
             if x is not None and y is not None]
    if len(pairs) < 3:
        return None, len(pairs)
    vx = [p[0] for p in pairs]
    vy = [p[1] for p in pairs]
    try:
        from scipy.stats import spearmanr
        rho, _p = spearmanr(vx, vy)
        return (None if math.isnan(rho) else float(rho)), len(pairs)
    except ImportError:
        rx, ry = _rank_with_ties(vx), _rank_with_ties(vy)
        n = len(rx)
        mx, my = sum(rx) / n, sum(ry) / n
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                        * sum((b - my) ** 2 for b in ry))
        return (num / den if den else None), n


def summarize(items):
    """三指标均值 + 与 answer_hit/evidence 的 Spearman 交叉验证。"""
    metrics = ["faithfulness", "answer_relevancy", "context_precision"]
    summ = {}
    for m in metrics:
        vals = [it[m] for it in items if it.get(m) is not None]
        errs = sum(1 for it in items if it.get(f"{m}_error"))
        summ[m] = {"mean": round(sum(vals) / len(vals), 4) if vals else None,
                   "n_scored": len(vals), "n_error": errs}
    cross = {}
    for m, ref in [("faithfulness", "hit"), ("answer_relevancy", "hit"),
                   ("context_precision", "evidence")]:
        rho, n = spearman([it.get(m) for it in items],
                          [float(it.get(ref)) for it in items])
        cross[f"{m}_vs_{ref}"] = {"spearman_rho": rho, "n_pairs": n,
                                  "pass_0.7": (rho is not None and abs(rho) >= 0.7)}
    return {"metrics": summ, "cross_validation": cross}


# ---------------------------------------------------------------- 主流程
async def _amain(args):
    details, cfg = load_online_details(args.results)
    if args.limit:
        details = details[:args.limit]
    out_file = args.out
    done = {}
    if os.path.exists(out_file) and not args.force:
        with open(out_file, encoding="utf-8") as f:
            done = json.load(f).get("items", {})
        print(f"断点续跑: 已完成 {len(done)} 题，跳过")
    todo = [d for d in details if d["query"] not in done]
    print(f"待评估 {len(todo)} 题（workers={args.workers}，ragas LLM={RAGAS_LLM_MODEL}，"
          f"emb={RAGAS_EMB_MODEL}，judge_base={RAGAS_LLM_BASE_URL}）")
    if not todo:
        items = list(done.values())
    else:
        metrics = build_metrics()
        sem = asyncio.Semaphore(args.workers)
        items = list(done.values())
        t0 = time.time()
        for i in range(0, len(todo), args.batch):
            chunk = todo[i:i + args.batch]
            results = await asyncio.gather(
                *[eval_item(metrics, it, sem) for it in chunk])
            for r in results:
                done[r["query"]] = r
                items.append(r)
            elapsed = time.time() - t0
            print(f"  {min(i + args.batch, len(todo))}/{len(todo)}  "
                  f"({elapsed:.0f}s)", flush=True)
            _checkpoint(out_file, done, cfg, args)
        print(f"评估完成 {len(todo)} 题，耗时 {time.time() - t0:.0f}s")
    summary = summarize(items)
    _checkpoint(out_file, done, cfg, args, summary=summary, final=True)
    _print_summary(summary, len(items))
    return out_file


def _checkpoint(out_file, items, cfg, args, summary=None, final=False):
    payload = {
        "__meta__": {
            "ts": int(time.time()),
            "source_results": os.path.basename(args.results),
            "source_label": "答案(线上同prompt)",
            "gen_model": cfg.get("gen_model"),
            "index_md5": cfg.get("index_md5"),
            "online_prompt_sha1": cfg.get("online_system_prompt_sha1"),
            "ragas_llm_model": RAGAS_LLM_MODEL,
            "ragas_llm_base": RAGAS_LLM_BASE_URL,
            "ragas_emb_model": RAGAS_EMB_MODEL,
            "workers": args.workers,
            "n_items": len(items),
            "final": final,
        },
        "items": items,
    }
    if summary:
        payload["summary"] = summary
    tmp = out_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out_file)


def _print_summary(summary, n_items):
    print("\n=== RAGAS 三指标汇总 ===")
    for m, s in summary["metrics"].items():
        print(f"  {m:<18} mean={s['mean']}  scored={s['n_scored']}  "
              f"error={s['n_error']}")
    print("\n=== 与 answer_hit 口径的 Spearman 交叉验证 ===")
    for k, c in summary["cross_validation"].items():
        flag = "PASS" if c["pass_0.7"] else "FAIL"
        rho = f"{c['spearman_rho']:.3f}" if c["spearman_rho"] is not None else "N/A"
        print(f"  {k:<38} rho={rho}  n={c['n_pairs']}  [{flag}]")


def main():
    ap = argparse.ArgumentParser(description="RAGAS 三指标独立评估（隔离 venv）")
    ap.add_argument("--results", default=os.path.join(
        HERE, "output", "eval_results_183_final.json"))
    ap.add_argument("--out", default=os.path.join(
        HERE, "output", "ragas_eval_183.json"))
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 题（调试用）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=10, help="每批落盘题数")
    ap.add_argument("--force", action="store_true", help="忽略断点重评")
    args = ap.parse_args()
    # 防呆：本脚本必须在隔离 venv 跑（生产 venv 没有 ragas）
    try:
        import ragas  # noqa: F401
    except ImportError:
        raise SystemExit(
            "[环境错误] 当前解释器无 ragas。请用 ../.venv-eval/Scripts/python.exe "
            "运行本脚本（生产 .venv 禁止安装 ragas，见 LangChain 接入评估报告）。")
    if not RAGAS_LLM_API_KEY:
        raise SystemExit("[配置错误] 缺少 judge 侧 key（8-金标扩充版本/.env）")
    try:  # 端点探活：失败时给出可读错误而非长篇堆栈
        r = requests.get(f"{RAGAS_LLM_BASE_URL}/models",
                         headers={"Authorization": f"Bearer {RAGAS_LLM_API_KEY}"},
                         timeout=15)
        if r.status_code not in (200, 401, 403, 404):
            print(f"[警告] AGICTO 探活 status={r.status_code}，继续尝试")
    except Exception as e:
        print(f"[警告] AGICTO 探活失败: {str(e)[:80]}，继续尝试")
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()

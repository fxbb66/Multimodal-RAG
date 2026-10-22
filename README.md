# 迪士尼 RAG 助手 · 完整知识库版

> **本仓库即项目根目录**（原 `8-金标扩充版本` 内容扁平化至此，2026-10-22 更新上传）。
> 相对 2026-10-14 首版（7-完整知识库版本）的更新：① 评估金标扩充 24→135 / 82→183；
> ② 意图链路七项加固；③ 金标第二轮口径修订 10 条；④ 工程优化 S1–S6：183 金标全量重跑落新基线、
> 证据充分性自检闭环（S3）、无历史指代兜底（S4）、图片金标 VL 生成+图片检索评估（S6）、
> 证据丢失根因诊断（S2）、RAGAS 三指标隔离 venv 交叉验证（S5）；
> ⑤ S7 路径级融合并集 + S8 图片双向量入库 + 前端影子适配：混合+Rerank
> recall@5 0.929→0.956、图片金标 0.731→0.8077（索引 d59376571cd8，合跑全量基线
> `output/eval_results_183_s7s8.json`；该产物不入库，可重跑再生）。
> 逐项改动清单见 `docs/05-金标扩充说明.md`；意图链路全流程解析见
> `docs/02-全流程细节梳理-迪士尼RAG助手.md` 第 16 章；S8 双向量审核见 `docs/04-工程决策与收官报告.md` 第 13 章。
> 原始知识库资料与构建产物不入库（需自备，见文末"工程说明"）。**文档总索引见 `docs/README.md`**。

基于 69 份多格式资料（Word/PPT/PDF/图片）的多模态检索增强问答系统。
混合检索（BM25 + 向量 + RRF + Rerank）+ qwen3.7-plus 生成，Flask 前端运行于 `http://127.0.0.1:5001`。
全部模型走 AGICTO 平台（OpenAI 兼容接口，每模型独立 Key），百炼 Key 仅作 rerank 降级兜底。

## 文档导航（docs/）

全部项目文档归档在 `docs/`，索引与维护规范见 `docs/README.md`。常用入口：

| 我想…… | 去读 |
|---|---|
| 快速了解项目全貌（新成员第一站） | `docs/01-项目介绍-迪士尼RAG助手.md` |
| 搞清代码级全链路细节 / 改检索前必读 | `docs/02-全流程细节梳理-迪士尼RAG助手.md`（第 17 章 = S1–S8 全景权威口径） |
| 排查踩坑 / 提炼方法论 | `docs/03-工程复盘-踩坑记录与排查流程.md` |
| 理解"为什么这么做"（S1–S8 决策、Agent/LangChain 选型） | `docs/04-工程决策与收官报告.md` |
| 引用指标数值 / 改评估口径 | `docs/05-金标扩充说明.md` |
| 意图链 prompt 原文对照 | `docs/07-query_intent_prompts.md` |

## 目录结构

```
Multimodal-RAG/
├── .env.example          # 密钥模板（复制为 .env 后填入）
├── requirements.txt
├── 0-转换旧格式.ps1       # Word/PowerPoint COM：.doc/.ppt → .docx/.pptx（需 Windows + Office，PS7 执行）
├── 1-parse_and_chunk.py  # 解析 + 结构感知切片（表格整块、短块合并、OCR 扫描页）
├── 2-build_index.py      # embedding 入库（FAISS IndexFlatIP + BM25），断点续传
├── 3-image_textify.py    # 图片 OCR(RapidOCR 本地) + qwen3.5-omni-flash caption
├── 4-evaluate.py         # 主评估：检索网格 + 答案侧 + 改写/意图口径 + 意图金标 + 图片金标档
├── 5-diagnose.py         # 体检/同源校验（all）+ 阈值标定（calib），产物带 __diag_meta__ 指纹
├── smoke_test.py         # 线上链路烟测（检索/降级/SSE），产物 smoke_report.json
├── eval_intent_gold.json # 意图专用人工金标集（不绑索引，随意图链路演进扩充）
├── 6-expand_gold.py      # QA 金标扩充（82→183，与 4-evaluate 同源 gen_qa 管线同模型同 prompt）
├── 7-build_intent_gold.py # 意图金标构建（含口径修订记录）
├── 8-validate_gold.py    # 金标结构化+口径校验（确定性检查，不调 LLM）
├── 9-export_intent_eval.py # 意图金标评估导出（复用主评估模块，审计字段现算）
├── 10-diagnose_evidence.py # S2 证据丢失根因诊断（rerank/融合/双丢三分类 + 词面特征）
├── 11-eval_selfcheck.py  # S3 证据充分性自检 A/B（触发率/误触发/覆盖率净提升）
├── 12-build_image_gold.py # S6 图片金标构建（VL 识图生成"必须看图才能答"的 QA）
├── 13-eval_image_gold.py # S6 图片金标检索评估（线上同口径，可被 4-evaluate 调用）
├── 14-ragas_eval.py      # S5 RAGAS 三指标（faithfulness/answer_relevancy/context_precision）
│                         #   + 与 answer_hit 口径 Spearman 交叉验证；必须在 .venv-eval 运行
├── compare_baselines.py  # S1 收尾：183 新基线 vs 82 旧基线逐档对比（一次性工具，跑完可留档）
├── compare_final.py      # 82 vs 183-S1 vs 183 最终版三方对比（含 §8 触发器判定）
├── 15-rerun_polluted_tiers.py # 503 风暴波及档位的 API 恢复后干净重跑（仅零降级写回）
├── 16-gen_colloquial.py  # S6+ 口语化问句生成（183 书面 QA → 口语化变体，扩充口语输入覆盖）
├── 16-precheck_path_union.py # S7 路径级 top-k 并集零 API 预分析（TOPK 定标：9 为拐点）
├── 17-compare_s7s8.py    # S7+S8 合跑 vs 最终版逐档对比（>2pt 关注 + §8 触发器 + 指纹核对）
├── 18-rerun_polluted_grid.py # S7+S8 合跑网格档 503 污染后的干净重跑（同 15 号安全闸门）
├── lib/
│   ├── __init__.py       # lib 包标记
│   ├── settings.py       # 集中配置（密钥/超参全部来自环境变量/.env）
│   └── retriever.py      # BM25 / RRF / rerank 降级链 / HybridRetriever / MMR
├── frontend/
│   ├── app.py            # Flask 服务（SSE 流式、会话、限流、/api/health、周期维护线程）
│   ├── static/           # 本地第三方前端库（marked.min.js / purify.min.js，无 CDN 运行时依赖）
│   └── templates/index.html
├── docs/                 # 全部项目文档（索引见 docs/README.md）：01 介绍 / 02 全流程细节 / 03 工程复盘 / 04 决策与收官 / 05 金标说明 / 07 prompt 存档
└── output/               # 构建产物（gitignore）：索引、metadata、BM25、缓存、评估结果
```

## 快速开始

```powershell
# 1) 虚拟环境 + 依赖
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 2) 配置密钥（必须）
copy .env.example .env
# 编辑 .env 填入：
#   AGICTO：EMBEDDING_API_KEY / GEN_API_KEY / REWRITE_API_KEY / VL_API_KEY / RERANK_API_KEY
#   DASHSCOPE_API_KEY  可选，仅用于 rerank 降级兜底（百炼千问系列）
#   AGICTO_BASE_URL 默认 https://api.agicto.cn/v1

# 3) 首次构建（如 converted/ 已存在可跳过步骤 a）
# a. 旧格式转换（含中文，必须用 PowerShell 7，勿用 powershell -File）
pwsh -Command "& '0-转换旧格式.ps1'"
# b. 切片 → c. 图片文本化 → d. 入库
.venv\Scripts\python 1-parse_and_chunk.py
.venv\Scripts\python 3-image_textify.py
.venv\Scripts\python 2-build_index.py

# 4) 启动服务
.venv\Scripts\python frontend\app.py
```

## 评估

```powershell
# 主评估（检索 + 答案 + 意图）
.venv\Scripts\python 4-evaluate.py
# 体检/同源校验（索引↔metadata↔chunks↔QA 指纹一致性）
.venv\Scripts\python 5-diagnose.py all
# 阈值网格标定（扫描 max_per_source / mmr_lambda，落盘 output/diag_calib.json）
.venv\Scripts\python 5-diagnose.py calib
# 线上链路烟测（需先启动 app.py）
.venv\Scripts\python smoke_test.py
# S2 证据丢失根因诊断（183 QA，线上口径分阶段追踪）
.venv\Scripts\python 10-diagnose_evidence.py
# S3 证据充分性自检 A/B（off/on 两档同题对比）
.venv\Scripts\python 11-eval_selfcheck.py
# S4 意图金标复测（v3 改写缓存，零改动复测口径）
.venv\Scripts\python 9-export_intent_eval.py
# S6 图片金标：VL 生成 QA（幂等，--force 重建）→ 图片检索评估
.venv\Scripts\python 12-build_image_gold.py
.venv\Scripts\python 13-eval_image_gold.py
# API 恢复后补跑被 503 风暴波及的检索档位（幂等；仅首选 rerank 零降级才写回）
.venv\Scripts\python 15-rerun_polluted_tiers.py
# S5 RAGAS 三指标（必须用隔离 venv；生产 .venv 禁止装 ragas）
.venv-eval\Scripts\python 14-ragas_eval.py --results output\eval_results_183_final.json
#   现状：50/183 检查点（用户定收尾标准，结论限 QA 集前段子集），断点续跑即补全
# AGICTO 额度耗尽/逾期时，judge 通道可降级到 DashScope（embedding 侧自动跟随）：
#   $env:RAGAS_LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
#   $env:RAGAS_LLM_API_KEY="<DASHSCOPE_API_KEY>"; $env:RAGAS_LLM_MODEL="qwen-plus"
#   再跑上一行。判官与被评答案不同源（qwen-plus 判 qwen3.7-plus 的答案），
#   产物 __meta__.ragas_llm_base 留痕；max_tokens=8192 防长答案判定被截断
# 最终版三方对比（82 基线 vs 183 S1 vs 183 最终版，含 §8 触发器判定）
.venv\Scripts\python compare_final.py
# S7+S8 合跑对比（183 最终版 vs 183-S7S8，逐档 Δ + >2pt 关注 + §8 + 配置指纹）
.venv\Scripts\python 17-compare_s7s8.py
# 合跑网格档遭 503 污染后的干净重跑（幂等；仅首选 rerank 零降级才写回，写回前备份）
.venv\Scripts\python 18-rerun_polluted_grid.py
# S6+ 口语化问句混入：先生成补充集，再在主评估开启并入（默认关，不改 183 书面口径）
.venv\Scripts\python 16-gen_colloquial.py
#   $env:QA_INCLUDE_COLLOQUIAL="1"   # 4-evaluate.py 会把 output/eval_qa_colloquial.json 并入主 QA
.venv\Scripts\python compare_final.py
```

- **QA 尺子与索引解耦（chunk_hash remap）**：QA 集每条带 `chunk_hash`；加载时若索引已重建，按内容 hash 把 `gold_idx` 自动重映射到新位置，命中率丢失率超 50% 才判定尺子失效并重新生成。重建索引不再让评估归零。
- QA 集由 chunk 反向生成、含金标答案，每来源文件限额 2 条缓解同源偏差（n=183 检索 / n=183 答案，2026-10-20 由 82/80 全量放开；82 旧基线备份 `output/eval_results_82_baseline.json`）。
- 检索侧档位（各档参数见 `eval_results.json` 的 `results[]`）：
  1. 纯向量 (k=5)
  2. 混合 BM25+向量 (k=5)
  3. 混合+Rerank (k=5)
  4. **2×2 网格**：{k=3,k=5} × {后处理关 / 后处理开(MMR+线上限额)}——分离 k 值与后处理的净贡献
  5. **线上改写 query 档**：与"k=5 后处理开"仅 query 来源不同（原句 vs intent_rewrite 改写后），量化改写净贡献
  6. 阈值扫描 2 行（`SWEEP_LAMBDA` / `SWEEP_MAX_PER_SOURCE` 环境变量覆盖，仅参考）
- 答案侧档位：2×2 网格（short 评估 prompt）+ **线上同 prompt 档**（客服 system + `[n]` 编号上下文，与 app.py 逐字同源；判定前剥离引用标记）+ 无 rerank 参考基线。指标为证据覆盖率、答案命中率、**证据到位时命中率**（生成侧真实水平）、证据在却答错比例。
- **意图两档**：N5 意图识别（QA 集上的 action 分布统计）+ 意图金标集 `eval_intent_gold.json`（人工标注，算 action 准确率 / clarify P/R/F1 / 类型命中 / 多意图 / 消解词保留率，不绑索引）。
- **意图确定性规则门**（`retriever.py`，不调模型）：① 无会话历史 + ≤8 字 + 剥离疑问/功能词后零实体残留 → 直接 clarify（拦"多少钱/几点开始/在哪买"；有历史的承接追问、含"门票/酒店"等实体词、"天生一对是哪一年"类含专名短问均不触发，该门在 app 中独立于改写开关常生效）；② ≥3 个并列问号 → 按问号强制拆分（2026-10-20 起**前置于 LLM 之前**，LLM 失败不影响拆分，只影响细化）；③ 无历史指代兜底（S4，2026-10-21 放宽）：LLM 标了 `underspecified`、query 含"这个/那个/那里"等指代词且无会话历史 → 改判 clarify。不再看 `need_clarification`——183 全量重跑实测两种自相矛盾模式都会漏（LLM 说不用澄清却标 underspecified／LLM 说澄清却被"改写优先"覆盖回 retrieve），而无历史时指代不可消解，追问是唯一正确行为；金标集"指代词+无历史"仅 1 条（标注 clarify），放宽零误伤，7 场景 mock 单测通过。
- **意图链路加固（2026-10-20）**：LLM 客户端单例 + `INTENT_TIMEOUT=45` + 重试 2 次（指数退避+抖动，内容审核/欠费不重试）+ JSON 容错抢救 + Step2 阶段总预算 `INTENT_TOTAL_BUDGET=18`（Step1 必跑，预算只约束改写）；降级不再静默——`INTENT_STATS` 计数、trace 透出 `intent_degraded`/`intent_degrade_reason`/`intent_ms`、`/api/health` 增加 `intent_cumulative`。
- **证据充分性自检（S3，2026-10-20 实测，默认关）**：检索后 LLM 判定 top-k 是否足以回答问题，不足时用 `expand_query_terms` 扩展 query **替换**首轮重检索一次（`max_retry=1`，防越查越偏）；自检失败按"充分"处理不阻塞。**183 对 A/B 实测（`output/eval_selfcheck_183.json`）判定不达标，维持 `EVIDENCE_SELF_CHECK=0`**：触发率 13.1%（24 题），其中 87.5%（21 题）为误触发——gold 已在首轮 top-5（多数在 pos=0）却被判不充分，自检模型系统性地过严；净收益 0 题、净损失 0 题（替换策略未伤已命中题）；词典扩展重检索几乎不改变排序（多数触发题 on_pos == off_pos），且出现 1 题危险模式"重试后自检判充分但 gold 仍不在 top-5"（被错误证据说服）；自检本身 4.4% 降级率、单次中位 6.1s。结论：当前瓶颈在检索召回（见 S2 诊断），充分性判断层无正贡献，留待召回提升后再评估。
- rerank 降级直接暴露在结果的 `rerank生效` 字段（calls/fallbacks/last_model/preferred_ok；首选模型欠费时 preferred_ok=False 属外部计费问题，不等于 rerank 失效）。
- `eval_results.json` 含配置快照（时间、模型链、超参、索引 md5/ntotal/dim、rerank_chain、online_system_prompt_sha1、缓存命中），格式为 `{config, results}`，tmp+replace 原子落盘。
- 性能：检索/生成均线程池并发（`EVAL_WORKERS` 默认 8，可在 `.env` 调整；遇平台限流调低到 4）。
- 缓存（均带模型/版本指纹，口径变化自动失效）：embedding→`output/eval_emb_cache.npz`（模型+dim 指纹）；答案→`output/eval_ans_cache.json`（模型+prompt版本+index_md5+问题+上下文 哈希，short 档 key 保持旧版兼容）；意图改写→`output/eval_rewrite_cache.json`（改写模型+置信度阈值+拆分上限指纹）。跨档相同上下文自动去重，代码/索引未变时重跑零 LLM 调用。
- 诊断产物 `diag_*.json` 带 `__diag_meta__`（index_md5 等）；阈值标定结论与原始网格写入 `output/diag_calib.json`，标定后才允许回写 `settings.py`。
- 构建完成度：1/2/3 脚本分阶段写 `output/build_report.json`（各产物计数与 md5、模型指纹、硬一致性检查），stage2 汇总 `overall.ok` 与 warnings。历史索引可离线回填 stage1/stage3（载荷带 `backfilled: true` 与依据说明，无法回溯的解析期瞬时计数置 null，不伪造），再由 stage2 原检查逻辑重算 overall。
- **当前基线读数（S7+S8 合跑，2026-10-21，`output/eval_results_183_s7s8.json`；索引 d59376571cd8）**：混合+Rerank recall@5 **0.956**（旧 0.929，+2.7pt，§8 未触发）、线上口径 0.945、图片金标 image_recall@5 **0.8077**（旧 0.731，lost_at_fusion 清零）、online 答案档 覆盖 0.634 / 命中 0.530、意图金标零回归。网格 7 档运行期遭 rerank 503 风暴（31 次降级），读数为保守下偏，干净重跑见 `18-rerun_polluted_grid.py`；online"证据到位时命中"0.791→0.767 经翻转分析判定为生成采样噪声（z=-0.441，噪声带 ±4pt）。逐档明细与证据链见 `docs/04-工程决策与收官报告.md` 第 13 章第十节。

## 主要 HTTP 接口

| 接口 | 说明 |
|---|---|
| `POST /api/query_stream` | SSE 问答（默认）。body：query/top_k/category/session_id/use_rerank/use_rewrite |
| `POST /api/query` | 非流式问答，字段与流式一致 |
| `GET /api/stats` | 动态口径（总量/文本/图片/类别数/模型） |
| `GET /api/health` | 索引规模与 md5、模型链/阈值/prompt 指纹、build_report 裁决、rerank 调用降级、会话/缓存占用 |
| `GET /api/categories` / `GET /api/chunks` | 知识库浏览 |
| `GET /api/session/new` | 新建多轮会话 |
| `GET /api/metrics` | 运行时轻量指标（embedding 缓存命中率 / rerank 累计 / 会话数 / uptime） |
| `GET /kb_image/<path:rel_path>` | 知识库原图服务（send_from_directory，防路径穿越） |

## 可调超参（.env，均有默认值）

`RERANK_TOPN=12` · `MAX_PER_SOURCE=3`（单文件结果上限，1152 维索引阈值标定后回写，标定网格见 `output/diag_calib.json`）· `USE_MMR=1` · `MMR_LAMBDA=0.72`
`MEDIA_SIM_THRESHOLD=0.45` · `MEDIA_GAP=0.02`（媒体图视觉近似时按图片候选池 BM25 词面分决胜；尚无图片金标集，待标定）
`DISNEY_GEN_MODEL=qwen3.7-plus` · `REWRITE_MODEL=qwen3.8-flash` · `EMBEDDING_MODEL=tongyi-embedding-vision-plus`（1152 维）
`RERANK_MODELS=qwen3-rerank`（逗号分隔，优先级从高到低；默认首选 AGICTO qwen3-rerank）
`SESSION_MAX=200` · `SESSION_TTL=7200` · `EMBED_CACHE_SIZE=256` · `RATE_LIMIT_PER_MIN=30`
意图链路（2026-10-20 新增，均有默认值）：`INTENT_TIMEOUT=45`（单次 LLM 超时秒）· `INTENT_MAX_RETRY=2`（可重试错误重试次数）· `INTENT_BACKOFF_BASE=0.8`（退避基数秒，指数增长+抖动）· `INTENT_TOTAL_BUDGET=18`（Step2 改写阶段总预算秒，超出停止后续改写不阻塞回答）
证据充分性自检（S3，2026-10-20 新增，均有默认值）：`EVIDENCE_SELF_CHECK=0`（1=开启自检闭环）· `EVIDENCE_MAX_RETRY=1`（重检索次数上限）· `EVIDENCE_CHECK_TIMEOUT=20`（单次自检超时秒，失败按"充分"处理不阻塞）
rerank 瞬时失败重试（2026-10-21 新增，均有默认值）：`RERANK_MAX_RETRY=5`（5xx/超时每级重试次数，full-jitter 退避）· `RERANK_BACKOFF_BASE=1.0`（退避上限基数秒，实际 sleep∈[0, base·2^attempt]）。背景：AGICTO rerank 实测约 3-10% 瞬时 503，原每级只重试 2 次导致整链失败、检索静默退化为"融合未重排"序；确定性失败（401/403/404/欠费信封，含百炼 `overdue-payment`/`good standing`）不重试直接熔断降级

## 工程说明

- **密钥零硬编码**：统一经 `lib/settings.py` 从 `.env` 读取；`.env` 已 gitignore
- **Rerank 降级链**：AGICTO `qwen3-rerank`（首选，2026-10-02 起；原首选 `rerank-v3.5` 因 Cohere 账户长期欠费每次必降级，已移出默认链，账户恢复后可经 `.env` 加回）→ AGICTO 整体不可用时百炼主 Key 兜底 `qwen3-rerank`（rerank-v3.5 百炼无此模型，自动排除）
- **Embedding 模型一致性**：建索引与查询均用 `tongyi-embedding-vision-plus`（1152 维）。`output/embeddings.npy` 带模型+dim 指纹，指纹不符自动判废重嵌；更换 embedding 模型必须删 `.faiss` 重建，否则向量空间不匹配
- **构建/诊断可复现**：各阶段产物均带指纹——metadata 每条含 `chunk_hash`，图片文本缓存含 caption 模型名，OCR 缓存含引擎名/版本，整体裁决见 `output/build_report.json`；`5-diagnose.py all` 断言评估产物与当前索引同源后方可采信指标
- **Windows 中文路径**：FAISS 的 fopen 不支持中文路径，索引读写均经 ASCII 临时路径中转
- **本地模型**：仅 RapidOCR（onnxruntime）；BM25 为字符二元组纯 Python 实现，无 jieba 依赖
- **RAGAS 隔离环境**：`14-ragas_eval.py` 依赖 ragas/scipy（含 langchain 系传递依赖）严禁装入生产 `.venv`；隔离 venv `.venv-eval`（建于仓库根目录），依赖清单固化于 `requirements-eval.txt`（freeze 自该 venv：ragas 0.4.3 + scipy 1.18.1），换机可复现
- 知识库资料位于 `../迪士尼RAG知识库（完整）/`，不纳入版本库

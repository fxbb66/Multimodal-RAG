# 迪士尼 RAG 助手 · 项目介绍

> 本文基于 **8-金标扩充版本**（当前唯一活跃主线）的实际代码与产物现状撰写，面向新成员与相关方。
> 所有指标读数均标注对应基线与索引版本；所有"待标定/默认关/闲置"判断均给出代码或产物证据。
> 姊妹目录 `7-完整知识库版本` 为 2026-10-02 的只读历史副本（兼回滚源），不再演进。

---

## 0. 一页速览

| 维度 | 现状 |
|---|---|
| 定位 | 基于 69 份内部资料（Word/PPT/PDF/图片）的迪士尼主题多模态 RAG 知识问答。**确定性检索工作流，不是 Agent** |
| 形态 | Flask 单体服务（`127.0.0.1:5001`），检索管线为零框架自研 Python |
| 检索 | 字符二元组 BM25 + FAISS 向量 + 加权 RRF + rerank 降级链（含熔断）+ 位次版 MMR + 同源限额 |
| 生成 | qwen3.7-plus（AGICTO 主 / 百炼备，可切换），SSE 流式，行内引用 `[n]` + 确定性来源清单 |
| 知识规模 | 472 个文本块 + 9 张图片（图片双向量占 2 槽，共 490 槽）；索引 md5 `d59376571cd8` |
| 评估 | QA 金标 183 对 + 意图金标 135 条 + 图片金标；检索/答案/意图/图片四类档位，全部带指纹同源校验 |
| 当前基线读数 | 混合+Rerank Recall@5 **0.956**｜线上口径 **0.945**｜图片 image_recall@5 **0.8077**｜答案命中 **0.530**（证据到位时 **0.767**）｜意图类型命中 **0.9926** |
| 代码规模 | `lib/retriever.py` 1438 行、`frontend/app.py` 845 行、`4-evaluate.py` 1250 行，另有 20+ 个构建/评估/诊断脚本（合计 23 个可执行入口） |

---

## 1. 项目定位：它是什么，不是什么

**要解决的问题**：迪士尼园区知识散落在 69 份 Word/PPT/PDF/图片中（票价表、规则、演出时间、邮轮价格图等），人工查阅低效。系统需支持自然语言提问，且回答必须**有出处、数值准、不编造**。

由此派生三条硬约束，并直接塑造了全部核心设计：

1. **只能依据知识库回答** → 生成侧 system prompt 强制"依据背景知识、不得编造、不足明示"；每个事实性结论带 `[n]` 来源编号；答案档评估专门度量"证据覆盖率"与"证据到位时命中率"。
2. **数值必须准确** → prompt 层禁止估算；检索侧对价格/时间类查询有 rerank 重排与同源限额保证；评估侧答案档用关键事实命中判定。
3. **本地、低成本、可复现** → 无 GPU、无向量数据库、无编排框架；FAISS 精确检索（IndexFlatIP）+ 纯 Python BM25；全部模型走 AGICTO/百炼 API。

**它明确不是**：

- **不是 Agent**：控制流固定（检索→生成）、工具集固定（ embedding/rerank/LLM 三类）、无自主循环。这是 2026-10-20 架构评审的结论（Agent 框架判据 5 条 0.5/5 命中，手搓判据 5/5）。
- **不用 LangChain/LlamaIndex**：1.x 起 Retrievers 全部移入 legacy 包；实测新增 55 个传递依赖（含 langsmith 上报合规风险）；其 MMR 用查询余弦做相关性，正是本项目实测否决的旧实现（recall 0.884→0.407）。结论与依据见 `docs/04-工程决策与收官报告.md` 第 15 章。
- **不是通用客服**：system prompt 限定"上海迪士尼度假区官方客服助手"口吻与规则。

---

## 2. 系统全景：一次问答的完整旅程

```
【离线数据管线】（产物在 output/，索引 md5 指纹贯穿全链路）
迪士尼RAG知识库（完整）/ ──0-转换旧格式.ps1──▶ converted/（.doc→.docx / .ppt→.pptx + manifest）
                                     │
                                     ▼
                          1-parse_and_chunk.py ──▶ output/chunks.jsonl（结构感知切片 + chunk_hash）
                                     │
                          3-image_textify.py ──▶ output/image_texts_<model>.json（OCR + caption）
                                     │
                                     ▼
                          2-build_index.py ──▶ disney_full_index.faiss + metadata.json + bm25_index.pkl
                                                    （文本向量 + 图片 real/shadow 双向量，490 槽）

【在线问答】（frontend/app.py，单端口 5001）
用户 query
   │
   ├─▶ 意图识别与改写（R3 v2：规则门 → LLM 分类 → 分类型改写；失败降级原 query 且记账）
   │        └─ 命中"信息不足"且不可改写 → 直接追问（clarify），不进检索
   ▼
   查询向量化（R5 同义词词典扩展 → tongyi-embedding-vision-plus）
   ▼
   双路并行召回：向量路（FAISS 内积=余弦，图片双向量归并）+ BM25 路（字符二元组）
   ▼
   加权 RRF 融合（k=60，BM25 权重 0.5 等权）+ S7 路径级 top-9 并集保底
   ▼
   R4 无意义块过滤 → rerank 重排（qwen3-rerank 主链，失败降级/熔断）
   ▼
   MMR 多样性（位次版 λ=0.72）+ 同源限额（单文件 ≤3）+ 类别预过滤
   ▼
   build_context（同源块合并去重 + [n] 编号 + 父子块上下文回填）
   ▼
   LLM 生成（qwen3.7-plus，system prompt 单一真源）── SSE 流式推 token
   ▼
   前端渲染（游客魔法等待 / 工作人员真实阶段面板）+ 确定性来源清单追加
```

旁路体系：**评估与诊断**（`4-evaluate.py` 及 5、9–18 号脚本）复用同一条检索/生成管线，靠"金标 + 缓存指纹 + 索引 md5"保证评估与线上同源。

---

## 3. 核心流程详解

### 3.1 离线数据管线

#### 3.1.1 旧格式转换（`0-转换旧格式.ps1`）

- **做什么**：用本机 Office/WPS COM 组件把知识库中的 `.doc/.ppt` 转成 `.docx/.pptx`，产物进 `converted/` 并生成 `manifest.json`（原路径→转换后文件映射）。已存在的结果跳过（幂等）。
- **动机**：解析库对二进制旧格式支持差，COM 转换是 Windows+Office 环境最稳的路径。
- **注意**：路径已改为相对脚本解析（`$PSScriptRoot`），换机可用；需 PowerShell 7 执行。

#### 3.1.2 结构感知切片（`1-parse_and_chunk.py`）

- **运作方式**：
  - 目标 800 字/块、重叠 120 字；**段落是最小聚合单位，不从句中硬切**；
  - 表格整块保留 Markdown，超长表按行分组且每组重复表头；稀疏列（空值率 >80%）先裁剪再判表；
  - PPT 每张幻灯片一个独立 chunk（标题+正文+备注）；短文档不切碎；短块（<100 字非表格）并入前块（合并后上限 1200 字）；
  - PDF 按页提取，检测到扫描页（文本极少且含图）自动调 RapidOCR；剔除页眉页脚；
  - 每个 chunk 携带 `文档标题 > 章节标题` 路径、自动抽取标签（M2）、`chunk_hash`（正文去重，`sources` 记录全部来源）。
- **设计动机**：票价/规则类内容的价值单元是"一个完整的表/一段完整的规则"，从句中硬切会同时摧毁检索与生成；父子块设计（子块召回、父块回填，I1）让检索精度与生成上下文兼得。
- **效果**：`chunks.jsonl` 的 chunk_hash 同时是**评估尺子与索引解耦**的锚点（见 3.4.2）。

#### 3.1.3 图片文本化（`3-image_textify.py`）

- **运作方式**：对 9 张邮轮价格图跑双通道——RapidOCR 识别价格表文字 + qwen3.5-omni-flash 生成描述 caption，合并写入图片 metadata 的 content 字段；结果按 caption 模型名分文件落盘并带 `__cache_meta__` 指纹，旧版无指纹文件自动迁移。
- **动机**：图片若不文本化，文字查询（BM25 与向量文本路）永远捞不到它。OCR 保精确数值、caption 保语义可检索，两者互补。
- **效果**：图片首次可被文字检索命中，是图片金标 0.8077 的前提。

#### 3.1.4 向量入库（`2-build_index.py`）

- **运作方式**：文本批量（每批 8 条）调 `tongyi-embedding-vision-plus` 生成 1152 维向量，失败自动降级逐条；写入 FAISS **IndexFlatIP**（向量先 L2 归一化，内积即余弦，R6）；BM25Index 同步 pickle；`embeddings.npy` 断点缓存使重跑不重复计费；FAISS 文件经 ASCII 临时路径写入（规避 Windows fopen 不支持中文路径）；分阶段写 `build_report.json`（计数+md5+模型指纹+一致性检查，stage2 汇总 `overall.ok`）。
- **图片双向量设计（S8）**：每张图片占 FAISS **两行**——real（视觉向量）+ shadow（caption+OCR 文本向量），metadata 与槽位严格 1:1；检索侧 `_dedup_vec_ranks` 按 group 归并取最高分。
  - **动机**：图片既有视觉内容又有文字内容，用户可能用任一方式问到；双行让两路都能捞到，归并保证不重复占位。
  - **取舍**：相比"槽位映射表"方案，双行零映射成本、旧索引向后兼容（无 group 字段时归并逻辑恒等）；代价是槽位数 = 块数 + 图片数。
- **效果**：当前索引 490 槽（472 文本 + 9 图 ×2），`image_recall@5` 0.731→0.769（双向量本身）→0.8077（叠加 S7 后）。

### 3.2 在线检索流程（`lib/retriever.py`，核心中的核心）

#### 3.2.1 意图识别与改写（R3 v2）

**三层结构**：

1. **确定性规则门**（不调模型，零延迟）：
   - 门①：无会话历史 + ≤8 字 + 剥离疑问/功能词后零实体残留 → 直接 clarify（拦"多少钱/几点开始"类裸问；有历史的承接追问、含"门票/酒店"实体词、含专名短问均不触发）；
   - 门②：≥3 个并列问号 → 按问号强制拆分（**前置于 LLM 之前**——原实现放在 LLM 后，一旦 LLM 超时降级三连问门根本没机会执行）；
   - 门③（S4）：LLM 标 `underspecified` + 无历史 + 含指代词 → 强制 clarify（无历史时指代不可消解，追问是唯一正确行为；金标验证放宽零误伤）。
2. **Step1 意图分类**（LLM）：七种类型（上下文依赖/对比/模糊指代/多意图/反问情绪/信息不足/明确）+ 置信度 + 是否需要追问。
3. **Step2 分类型改写**（LLM）：按类型套用对应 prompt（指代消解、对比显式化、多意图拆分、情绪剥离），串行执行受 **18s 总预算**约束，超预算即停止后续改写、按已有结果继续检索。

**工程加固**（2026-10-20，源于 135 金标实测 18% 需改写题因超时被静默降级）：OpenAI 客户端**单例**（复用连接池，旧实现每次新建客户端是超时爆发首因）；显式 45s 超时；有限重试 + 指数退避 + 抖动（内容审核/欠费类不重试）；LLM 输出 JSON 三重容错（代码块剥离、截断补全、单引号/中文引号抢救）；**降级不再静默**——`INTENT_STATS` 按原因分桶记账，trace 透出 `intent_degraded`/`degrade_reason`/`intent_ms`，`/api/health` 可见累计。

#### 3.2.2 双路召回

- **向量路**：`vec_k = max(k*4, 20)`（类别过滤时 `max(k*6,40)` 保证过滤后仍取满）；图片双向量先归并再去重。
- **BM25 路**：**字符二元组** BM25（k1=1.5, b=0.75），对中文短查询友好、零分词依赖；查询先过 R5 同义词词典扩展（"港迪"→"香港迪士尼乐园"等 10 组专有名词）。
- **动机**：jieba 分词在园区专有名词（"玲娜贝儿""尊享卡"）上不稳定，字符二元组以略高计算量换来零依赖与稳定的词面命中。

#### 3.2.3 融合：加权 RRF + S7 路径级并集

- **加权 RRF**（k=60）：各候选得分 = `weight/(k+rank+1)`，向量路与 BM25 路默认 0.5/0.5 等权。
  - **动机**：两路分数不可跨路比较（余弦 vs BM25 分），RRF 只用位次、天然稳健；权重参数只改变相对话语权。
- **S7 路径级 top-9 并集**：RRF 取 top-12 后，把两路**各自 top-9** 的并集中未入池候选按 RRF 分附在池尾；仅 rerank 档生效。
  - **动机**：等权 RRF 下单路 #0（0.0082）竞争不过双路 #10（0.0141），强单路命中会被挤出 rerank 候选池——S2 诊断实测 7/183 例 `lost_at_fusion`。
  - **定标**：`16-precheck_path_union.py` 零 API 预分析 183 题：TOPK=8→5/7 进池，9→6/7（拐点），12→6/7 但候选总量 +72.8%。取 9。
  - **效果**：合跑后 `lost_at_fusion` 清零，混合+Rerank recall@5 0.929→0.956。

#### 3.2.4 重排：降级链 + 熔断 + 重试

- **链路**：AGICTO `qwen3-rerank`（Cohere 风格 REST）→ 失败降级 AGICTO `qwen3-rerank` 备用模型 → AGICTO 整体不可用时百炼 `qwen3-rerank` 兜底（dashscope SDK，协议不同自动适配）。
- **四条关键工程决策**（全部来自真实事故）：
  1. **HTTP 200 但空 results 判失败**：AGICTO 欠费时返回 200 + 计费错误信封，不拦截就会"静默空转"；
  2. **确定性失败直接熔断 10 分钟**（401/403/404/欠费信封）：不重试、不降级等待——欠费状态下每查询白等 ~6s 是实测教训；
  3. **瞬时失败指数退避 + full-jitter 重试 5 次**：503 风暴期固定退避会同步重试再同步撞 burst，抖动后才能错峰穿过；
  4. **全程计数**（calls/fallbacks/circuit_skips/last_model）：评估档位靠 `fallbacks=0` 判定读数干净（见 3.4.5）。
- **为什么主选 qwen3-rerank 而非 rerank-v3.5**：后者所在 Cohere 账户长期欠费，置链首每次必失败再降级、纯增延迟。

#### 3.2.5 后处理：过滤、MMR、限额

- **R4 过滤**：丢弃过短（<8 字）且非表格的无意义块；影子槽恒有效（由主条目承载）。
- **MMR 多样性**：`λ·相关性 − (1−λ)·最大相似度`，**相关性项用池内位次**（继承 rerank 排序信号）而非原始余弦。
  - **动机**：旧实现用查询余弦做相关性会把 rerank 排序整体丢弃——实测 MMR-only recall@5 0.884→0.407，调 λ 无法补救，这是本项目最深刻的一次"想当然翻车"。
  - λ=0.72（单意图）/ 0.55（多意图候选跨子问题、冗余风险更高）。
- **同源限额**：单文件最多 3 条（`MAX_PER_SOURCE`），候选不足时从全库向量序跨源补齐。
  - **动机**：标定网格实测 cap: 1→0.756、2→0.915、3→0.951、不限→0.951；cap=3 与无限额持平且保留单源护栏。
- **类别预过滤**：前端选类别时只在类内召回（召回池相应扩大，保证过滤后取满）。

#### 3.2.6 证据充分性自检（S3）——已实现，默认关

- **机制**：首轮检索后 LLM 判定 top-k 是否足以回答；不足则用同义词词典扩展 query **替换**首轮重检索（`max_retry=1`，防"为补证据越查越偏"）；自检本身失败按"充分"处理，不阻塞回答。
- **为什么默认关**：183 题 A/B 实测（`output/eval_selfcheck_183.json`）——触发率 13.1%（24 题）其中 87.5% 为误触发（gold 已在首轮 top-5 却被判不充分，自检模型系统性过严）；净收益 0 题、净损失 0 题；还出现 1 题危险模式"重试后自检被错误证据说服"。结论：当前瓶颈在召回不在充分性判断层，留待召回提升后再评估。详见 README 与 `11-eval_selfcheck.py`。

### 3.3 在线生成流程

#### 3.3.1 上下文构建（`build_context`）

- 同源 chunk 合并（`merge_overlap` 去尾部/头部重叠，消除父子块邻块重复）；每个来源编号 `[n]`；缺失时用 `parent_context` 回填；末尾由 `render_refs` **确定性**追加文件定位清单——即使模型漏标，用户也能看到出处。

#### 3.3.2 生成

- **system prompt 单一真源**：`settings.ONLINE_SYSTEM_PROMPT`，线上服务与评估"线上同 prompt"档逐字共用（改 prompt = 改线上行为 = 改评估口径，须重跑留证）；`/api/health` 输出其 sha1 供交叉核对。
- **双平台**：AGICTO `qwen3.7-plus` 为主，百炼 `qwen-plus` 可经前端切换（未配置密钥的平台置灰）。
- prompt 层强制：只依据背景知识、数值禁估算、事实结论带 `[n]`、结构化信息优先表格。

#### 3.3.3 SSE 流式与真实阶段进度

- `meta`（chunks/引用映射/trace）→ `stage` 事件（rewrite→embed→retrieve→rerank→generate）→ token 流 → 来源清单。
- **检索在独立工作线程执行，阶段回调经队列实时推送**——工作人员面板看到的阶段边界是后端真实边界（真正发起 LLM/embedding/rerank 调用前才推送），不是前端定时器猜的假进度。rerank 降级/失败也作为 stage detail 真实呈现。
- **动机**：假进度在排障时是负资产（"显示检索中其实卡在 rerank 欠费等待"）。

#### 3.3.4 会话与追问

- `sessions` = OrderedDict LRU（`SESSION_MAX=200`）+ TTL 7200s + `_sess_lock` 全锁（threaded=True 并发安全）+ `clarify_n` 连续追问计数（上限 2，超限按原 query 强行检索，防追问死循环）。
- 追问命中时不进检索/生成，话术直接入会话历史保证下轮上下文。
- 周期维护 daemon（5 分钟）清扫过期会话与限流空桶。

### 3.4 评估与诊断体系（本项目最被低估的资产）

#### 3.4.1 档位设计：每个档回答一个具体问题

| 类别 | 档位 | 回答的问题 |
|---|---|---|
| 检索 | 纯向量 / 混合 / 混合+Rerank | 每级组件的净贡献 |
| 检索 | 2×2 网格 {k=3,5}×{后处理关/开} | k 值与后处理（MMR+限额）的净贡献分离 |
| 检索 | 线上改写 query 档 | 意图改写对召回的净贡献 |
| 检索 | 阈值扫描（SWEEP_LAMBDA/SWEEP_MAX_PER_SOURCE） | 参数敏感性（仅参考） |
| 答案 | 2×2 网格 + **线上同 prompt 档** | 短 prompt 与线上口径分别的真实生成水平 |
| 意图 | QA 集分布统计 + 135 条人工金标 | action 准确率 / clarify P/R/F1 / 类型命中 / 多意图 / 消解词保留率 |
| 图片 | 图片金标（VL 生成"必须看图才能答"的 QA） | image_recall@5 与丢失阶段定位 |

答案档四个指标刻意分离：`evidence_coverage`（召回是否到位）、`answer_hit`（无条件命中）、`answer_hit_given_evidence`（**生成侧真实水平**）、`gen_loss_when_evidence`（证据在却答错）——因为 0.53 的命中里，必须分清"没检到"和"检到了没答对"，否则优化方向都会错。

#### 3.4.2 金标与索引解耦（chunk_hash remap）

- QA 金标每条带 `chunk_hash`；加载时若索引已重建，按内容 hash 自动把 `gold_idx` 重映射到新位置（丢失率 >50% 才判尺子失效）。**重建索引不再让评估归零**。
- 金标演进：QA 82→183 对（`6-expand_gold.py`，与主评估同源 gen_qa 管线）、意图金标 24→135 条（两轮口径修订共 10 条，原则：先定目标行为再改标注，不粉饰真实观测）、图片金标（`12-build_image_gold.py` VL 识图生成）。

#### 3.4.3 缓存与指纹（复现性的地基）

- 三个缓存全部带指纹、口径变化自动失效：embedding（模型+dim）、答案（模型+prompt 版本+**index_md5**+问题+上下文哈希）、意图改写（模型+置信度阈值+拆分上限）。代码/索引未变时重跑零 LLM 调用。
- 索引一换，答案缓存按 index_md5 **整份判废**——这是"新索引必须重跑全量基线才能引用读数"这条纪律的机制来源。

#### 3.4.4 诊断工具链

- `5-diagnose.py`：候选池流水 + 单因素消融 / MMR 修复验证 / 答案 2×2 / 阈值标定网格，产物全部带 `__diag_meta__`（index_md5 等指纹）；
- `10-diagnose_evidence.py`（S2）：证据丢失根因三分类（rerank 丢/融合丢/双丢）+ 词面特征——S7 并集方案就是被它的 7 例 `lost_at_fusion` 直接驱动的；
- `11-eval_selfcheck.py`（S3 A/B）、`13-eval_image_gold.py`（图片档）、`14-ragas_eval.py`（faithfulness/answer_relevancy/context_precision 与现有口径做 Spearman 交叉验证，隔离 venv 运行）；
- `17-compare_s7s8.py` 等对比脚本：逐档 Δ + 超 2pt 关注 + §8 触发器判定。

#### 3.4.5 污染纪律：503 风暴的教训

- 长跑评估遭上游 503 时，rerank 会静默降级，**档位读数保守下偏但看不出来**。对策：每个结果行记录 rerank 降级数；`15/18-rerun_polluted_*` 按"fallbacks=0 才写回"的安全闸门干净重跑；合跑网格档曾遭 31 次降级波及 7 档，干净重跑后才得到可引用读数。
- 纪律：**按结果行切分日志逐档归因降级，不能只看结尾汇总行**。

### 3.5 前端（`frontend/templates/index.html`）

- **双角色**：游客（魔法等待话术）/ 工作人员（RAG 真实阶段面板：rewrite→embed→retrieve→rerank→generate，由 SSE stage 事件驱动），身份选择持久化。
- **追问态 UI**：信息不足时展示追问徽标 + 快捷按钮 + 重试入口，与后端 clarify 字段配套。
- **平台选择器**：AGICTO/百炼切换，未配置密钥置灰；`/api/metrics` 轮询渲染 embedding 缓存命中率。
- **知识库浏览**：类别/chunk 列表 + 图片灯箱（`/kb_image/`）。
- **安全**：marked 渲染的答案/表格/弹窗全部过 DOMPurify 3.1.6（本地 `static/purify.min.js`，无 CDN 依赖）；KB 元数据字段走 `escapeHtml`；DOMPurify 缺失时退化为纯文本转义（丢格式不丢安全）。
- SSE 为主、`/api/query` 非流式为回退路径，两者共用 `prepare_turn`。

### 3.6 服务与运维（`frontend/app.py`）

- 启动即 `require_keys("EMBEDDING_API_KEY","GEN_API_KEY")` fail-fast（rerank Key 缺失仅告警——避免误伤 `use_rerank=False` 场景）；
- 限流每 IP 每分钟 30 次（键取 `remote_addr`，X-Forwarded-For 客户端可伪造不可信）；
- `/api/health`：索引 md5/ntotal/dim、模型链、prompt sha1、build_report 裁决、rerank/intent 累计、会话与缓存占用——**与 eval_results.json 交叉核对同源性的接口**；
- `/api/metrics`：轻量运行时指标；
- Windows 中文路径：FAISS 读写经 ASCII 临时路径中转；
- `127.0.0.1` 绑定、`debug=False`、`threaded=True`。

---

## 4. 关键设计取舍汇总

| 决策 | 被否决的备选 | 选择理由（证据） |
|---|---|---|
| 自研管线，不接 LangChain | LangChain 1.x / LlamaIndex | Retrievers 移入 legacy 包；+55 传递依赖；其 MMR 实现实测 recall 0.884→0.407 |
| 确定性工作流，不做 Agent | LangGraph 等多 Agent 框架 | 控制流固定、工具集固定、无自主循环；框架能力 100 行内可复刻 |
| BM25 用字符二元组 | jieba 分词 | 中文短查询友好、专有名词稳定、零依赖 |
| RRF 位次融合 | 加权分数融合 | 两路分数不可跨路比较；位次语义稳健 |
| rerank 主选 qwen3-rerank | rerank-v3.5（Cohere） | 后者账户长期欠费，置链首每次必失败纯增延迟 |
| MMR 位次版相关性 | 查询余弦相关性 | 余弦版丢弃 rerank 排序信号，recall 0.884→0.407 |
| 同源限额 cap=3 | cap=2 / 不限 | 标定网格：3→0.951 与不限持平且保留护栏 |
| S7 路径并集 TOPK=9 | 8 / 12 / 17 | 预分析拐点：9→6/7 进池、候选 +34.4%；12 无增益更贵 |
| 图片 real+shadow 双向量 | 槽位映射表 / 单向量 | 零映射成本、旧索引向后兼容；代价=槽位 +9 |
| S3 自检默认关 | 默认开 | 183 题 A/B：触发率 13.1%、误触发 87.5%、净收益 0 |
| FAISS IndexFlatIP 精确检索 | HNSW/IVF 近似索引 | 490 槽规模精确检索足够快；且 `_dedup_vec_ranks` 的有序性前提依赖精确检索 |
| QA 金标全量 183 | 保持 82 小集 | 82 集测量精度不足（答案档 SE≈±3.8pt），无法分辨回归与噪声 |

---

## 5. 实际效果与价值

**检索质量**（基线均为 n=183、当前索引 d59376571cd8、rerank 零降级）：

| 指标 | pre-S7/S8 | 当前 | 说明 |
|---|---|---|---|
| 混合+Rerank Recall@5 | 0.929 | **0.956** | S7+S8 合跑，+2.7pt |
| 线上口径 Recall@k（后处理开+改写） | 0.918 | **0.945** | 用户可见路径 |
| 图片金标 image_recall@5 | 0.731 | **0.8077** | lost_at_fusion 清零 |
| online 答案档命中 / 覆盖 | 0.525 / 0.628 | 0.530 / 0.634 | 答案档噪声带 ±3.8pt 内 |
| 意图金标类型命中 | — | 0.9926 | clarify_recall 1.0，零回归 |

**工程价值**（比指标更难复制的部分）：

1. **可复现**：金标-索引-缓存三级指纹，任何读数都能回答"和哪份索引、哪版 prompt、哪条 rerank 链同源"；
2. **可归因**：降级全部记账（rerank fallbacks/circuit_skips、intent degraded_by 分桶、S3 触发/降级），指标塌陷能定位到具体组件而非全链背锅；
3. **防污染**：503 风暴纪律（fallbacks=0 才写回）保住过多轮基线不被静默降级污染；
4. **踩坑沉淀**：MMR 余弦版翻车（0.884→0.407）、欠费白等 6s、假进度、同步重试撞 burst——每个都变成了代码里的确定性机制，文档见 `docs/03-工程复盘-踩坑记录与排查流程.md`。

---

## 6. 尚未使用、闲置或未完全落地的模块（客观清单）

> 原则：代码在、开关关、或工具已跑完留档。均附证据，不粉饰。

| # | 模块 | 状态 | 证据与原因 |
|---|---|---|---|
| 1 | **S3 证据充分性自检** | 已接线，**默认关** | `EVIDENCE_SELF_CHECK=0`（settings）；183 题 A/B 不达标（触发率 13.1%/误触发 87.5%/净收益 0），详见 `output/eval_selfcheck_183.json`。三个 `EVIDENCE_*` 参数随之一休眠 |
| 2 | **媒体匹配阈值** `MEDIA_SIM_THRESHOLD=0.45` / `MEDIA_GAP=0.02` | **待标定** | settings 注释明示"尚无图片金标 query 集"——现有图片金标评的是检索召回，不是 pick_image 的视觉歧义决胜规则 |
| 3 | **多意图 MMR λ** `MMR_LAMBDA_MULTI=0.55` | **待验证** | settings 注释"待多意图金标集验证后再定标"——多意图无专门金标集，0.55 是单意图集上的保守取值外推。2026-10-22 复核：λ≤0.4 方向已被标定否决（recall@5 0.854 vs 0.915，`diag_calib.json`）；线上按意图分化 λ（多意图 0.55/单意图 0.72，`app.py:380`），评估网格固定 0.72 |
| 4 | **口语化问句集** | 已生成，**默认不并入** | `output/eval_qa_colloquial.json` 在（96KB）；`QA_INCLUDE_COLLOQUIAL=1` 才并入主评估，"不改 183 书面口径" |
| 5 | **VL_RERANK_API_KEY** | 预留，**代码零接入** | `.env.example` 明示"当前代码未接入"；百炼 qwen3-vl-rerank 多模态重排备用路径。2026-10-22 复核结论：**暂不接入**——S8 双向量后图片块经 shadow 文本向量走同一 rerank 链，瓶颈在 rerank 中位通过率而非缺 VL 裁判（分析见 `docs/03` 第 10 章） |
| 6 | **RAGAS 交叉验证** | **部分完成，结论为否定** | `output/ragas_eval_183.json` 为 50/183 检查点（用户定收尾标准）；隔离环境清单已固化 `requirements-eval.txt`。2026-10-22 复核：三组 Spearman（vs 自有口径）全为负（-0.089/-0.041/-0.145）**未达 0.7 门槛 → 新口径不予采信**，仍以规则匹配口径为唯一权威；双裁判对照显示 RAGAS 自身跨裁判稳定（rho 0.585/0.775/0.732），负相关源于两把尺子不同源 |
| 7 | **一次性工具**（跑完留档） | 闲置 | `6-expand_gold.py`、`compare_baselines.py`、`compare_final.py`、`15/18-rerun_polluted_*`、`17-compare_s7s8.py`——任务已完成，保留供复现与审计 |
| 8 | **`chunk_brief.distance` 字段** | 兼容字段 | 余弦时代遗留（`1-similarity`），新前端不用，保留向后兼容 |
| 9 | **`embeddings.npy`** | 历史断点缓存 | 索引已建成；仅重建/回填时复用 |
| 10 | **`7-完整知识库版本` 目录** | 只读历史副本 | 后端/检索/评估停更于 2026-10-02；价值 = 回滚源（含旧索引备份）与前端分叉修复来源 |
| 11 | **中间态备份文件** | 闲置 | `output/eval_rewrite_cache.before_opt.json`、`output/ragas_eval_183_dashscope_partial.json.bak`、`frontend/templates/index.html.bak_20261022` |
| 12 | **`5-diagnose.py calib`** | 按需运行 | 阈值已标定回写 settings，日常不跑；改参前才复跑 |
| 13 | **单测/CI** | **缺失** | 全仓零单测（方向9 候选：纯函数优先集已列好）；无 CI（项目无 git 仓库，待版本库落地后接：pytest 本地绿 → git 化 → CI 门禁）。RAGAS 已部分落地后，该项收窄为 pytest 纯函数集 |
| 14 | **增量入库** | 未做 | 改一个 chunk 需全量重建索引（断点缓存缓解计费，不缓解时间） |
| 15 | **规模化路径** | 未做 | BM25 O(n) 全扫、媒体匹配全库扫描——490 槽无感，万级需 HNSW/倒排/Redis（方向6，明确保留） |

另需说明的两个"不是问题"：`/api/query` 非流式端点是前端的回退路径（非闲置）；`_CIRCUIT` 熔断状态进程内有效、重启清零（设计如此，充值后 10 分钟冷却期自动探测恢复）。

---

## 7. 新人上手路径

**读代码顺序**（约半天）：

1. `README.md` —— 全貌与命令（本文的速查 complement）
2. `lib/settings.py` —— 所有可调参数与默认值（30+ 个，每个都有注释说明为什么是这个值）
3. `lib/retriever.py` —— 按 docstring 的 R1–R6/S1–S8 编号读，每个编号对应一次真实优化
4. `frontend/app.py` 的 `retrieve()` → `prepare_turn()` → 两个 API 路由
5. `4-evaluate.py` 的档位定义 —— 理解"每个指标在回答什么问题"

**跑起来**：

```powershell
python -m venv .venv; .venv\Scripts\pip install -r requirements.txt
copy .env.example .env   # 填入密钥
.venv\Scripts\python frontend\app.py    # 服务起在 127.0.0.1:5001
.venv\Scripts\python smoke_test.py      # 烟测（需先起服务）
```

**改检索逻辑前必读**：`docs/02-全流程细节梳理-迪士尼RAG助手.md` + `docs/03-工程复盘-踩坑记录与排查流程.md`；改动纪律 = 先审核后落地，任何管线/索引改动后必须重跑全量基线（旧读数自动作废，答案缓存按 index_md5 判废）。

---

## 8. 附录

### 8.1 文件地图（8-金标扩充版本）

| 文件 | 角色 |
|---|---|
| `0-转换旧格式.ps1` | 旧格式 COM 转换（一次性） |
| `1-parse_and_chunk.py` | 解析 + 结构感知切片 → chunks.jsonl |
| `2-build_index.py` | embedding + FAISS/BM25 入库（含图片双向量） |
| `3-image_textify.py` | 图片 OCR + caption |
| `4-evaluate.py` | 主评估（检索/答案/意图/图片四类档位） |
| `5-diagnose.py` | 候选池/MMR/答案/标定四组诊断 |
| `6-expand_gold.py` / `8-validate_gold.py` | QA 金标扩充 / 校验（一次性） |
| `7-build_intent_gold.py` / `9-export_intent_eval.py` | 意图金标构建 / 评估导出 |
| `10-diagnose_evidence.py` | S2 证据丢失根因诊断 |
| `11-eval_selfcheck.py` | S3 自检 A/B |
| `12-build_image_gold.py` / `13-eval_image_gold.py` | 图片金标构建 / 评估 |
| `14-ragas_eval.py` | RAGAS 三指标交叉验证（隔离 venv） |
| `15/18-rerun_polluted_*` | 503 污染档位干净重跑 |
| `16-gen_colloquial.py` / `16-precheck_path_union.py` | 口语化集生成 / S7 零 API 预分析 |
| `17-compare_s7s8.py` | S7+S8 合跑对比 |
| `compare_baselines.py` / `compare_final.py` | 历史基线对比（一次性） |
| `lib/settings.py` / `lib/retriever.py` | 配置单一真源 / 检索核心 |
| `frontend/app.py` / `frontend/templates/index.html` | Flask 服务 / 双角色前端 |
| `smoke_test.py` | 线上链路烟测 |
| `requirements.txt` / `requirements-eval.txt` | 生产依赖 / RAGAS 隔离环境依赖 |

### 8.2 关键参数默认值（均可在 .env 覆盖）

| 参数 | 默认 | 备注 |
|---|---|---|
| `RERANK_TOPN` | 12 | rerank 候选池（S7 并集后实际更多） |
| `MAX_PER_SOURCE` | 3 | 同源限额（标定回写） |
| `MMR_LAMBDA` / `MMR_LAMBDA_MULTI` | 0.72 / 0.55 | 单意图 / 多意图 |
| `BM25_WEIGHT` | 0.5 | 加权 RRF 中 BM25 份额 |
| `FUSION_PATH_UNION_TOPK` | 9 | S7 并集深度（0=关闭） |
| `INTENT_TIMEOUT` / `INTENT_MAX_RETRY` / `INTENT_BACKOFF_BASE` / `INTENT_TOTAL_BUDGET` | 45s / 2 / 0.8 / 18s | 意图链路四件套 |
| `RERANK_MAX_RETRY` / `RERANK_BACKOFF_BASE` | 5 / 1.0 | rerank 瞬时失败重试 |
| `EVIDENCE_SELF_CHECK` / `EVIDENCE_MAX_RETRY` / `EVIDENCE_CHECK_TIMEOUT` | 0 / 1 / 20s | S3（默认关） |
| `SESSION_MAX` / `SESSION_TTL` | 200 / 7200s | 会话 LRU + TTL |
| `EMBED_CACHE_SIZE` / `RATE_LIMIT_PER_MIN` | 256 / 30 | embedding 缓存 / 限流 |
| `EVAL_WORKERS` | 8 | 评估并发（平台限流时调 4） |

---

*本文档随代码演进维护；指标读数以 `output/eval_results.json` 及其 `config.index_md5` 为准。*

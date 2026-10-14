# -*- coding: utf-8 -*-
"""
迪士尼完整知识库 - 解析与结构感知切片
=====================================
输入：迪士尼RAG知识库（完整）/ 下全部文件 + converted/ 中由 COM 转换好的旧格式文件
输出：output/chunks.jsonl（每行一个去重后的文本 chunk，含多来源 metadata）

策略（已与用户确认）：
- 目标 800 字 / 重叠 120 字；段落为最小聚合单位，不从句中硬切
- 表格整块保留 Markdown；超长表按行分组且每组重复表头
- PPT 每张幻灯片一个独立 chunk（标题+正文+备注），超长再递归切
- 短文档不切碎；每个 chunk 携带 文档标题>章节 标题路径
- PDF 按页提取文本；检测到扫描页（文本极少且含图）时用 RapidOCR
- 相同正文 hash 去重，sources 记录全部来源文件与章节
用法：
  python 1-parse_and_chunk.py --scan-pdf   # 只检测 PDF 扫描页
  python 1-parse_and_chunk.py              # 正式切片
"""
import os
import re
import sys
import json
import time
import argparse
import hashlib

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
MAX_MERGED_LEN = 1200   # P1-7: 短块并入前一块后的正文长度上限
SHORT_MERGE_LEN = 100   # P1-7: 正文短于此的非表格块并入前一块
SPARSE_COL_EMPTY_RATE = 0.80  # 列空值率高于此值视为稀疏列（网页按钮/视觉占位列）
SPARSE_COL_MIN_ROWS = 4       # 行数不足时不做稀疏列判定，避免误杀小表

HERE = os.path.dirname(os.path.abspath(__file__))
KB_ROOT = os.path.normpath(os.path.join(HERE, "..", "迪士尼RAG知识库（完整）"))
CONVERTED_DIR = os.path.join(HERE, "converted")
MANIFEST_FILE = os.path.join(CONVERTED_DIR, "manifest.json")
OUT_DIR = os.path.join(HERE, "output")
OUT_FILE = os.path.join(OUT_DIR, "chunks.jsonl")

SCANNED_TEXT_THRESHOLD = 20  # 页内去空白后字符数低于此值视为疑似扫描页

# ---------------------------------------------------------------- 基础工具

def clean(text):
    if not text:
        return ""
    # \x0b(vertical tab)/\x0c(form feed) 是 Office 常见软换行，统一为换行
    return (text.replace("\xa0", " ").replace("\r", "")
                .replace("\x0b", "\n").replace("\x0c", "\n").strip())


def norm(text):
    """归一化用于去重 hash：去掉所有空白。"""
    return "".join((text or "").split())


def chunk_hash(body):
    """文本 chunk 内容指纹：去空白后 sha1。
    切片去重、metadata、评估 QA 尺子共用同一指纹——索引重建后位置可变，内容指纹不变。"""
    return hashlib.sha1(norm(body).encode("utf-8")).hexdigest()


def image_chunk_hash(rel_path):
    """图片 chunk 指纹：以库内相对路径为身份（同一图片换索引位置后仍可被 remap）。"""
    return hashlib.sha1(norm(f"image:{rel_path}").encode("utf-8")).hexdigest()


def join_vertical_text(text):
    """I2: 合并 PPT 中竖排文字（如 '巍\\n峨\\n耸\\n立' -> '巍峨耸立'）。
    规则：连续多行均为单字符（中文/英文/数字）时合并。"""
    lines = text.split("\n")
    out, buf = [], []
    for ln in lines:
        s = ln.strip()
        if len(s) == 1 and s not in "·•◆■▪":
            buf.append(s)
        else:
            if buf:
                out.append("".join(buf))
                buf = []
            out.append(ln)
    if buf:
        out.append("".join(buf))
    return "\n".join(out)


def extract_tags(file_name, category, heading):
    """M2: 从文件名、分类、章节自动抽取标签（不新增资料）。"""
    text = f"{file_name} {category} {heading}"
    # 预设主题词库（仅从现有资料语义归纳）
    keyword_map = [
        ("门票", ["门票", "票价", "票种", "票券"]),
        ("退款", ["退款", "退票", "退订", "退钱"]),
        ("酒店", ["酒店", "客房", "入住", "退房"]),
        ("餐饮", ["餐饮", "餐厅", "美食", "菜单"]),
        ("邮轮", ["邮轮", "航线", "船只"]),
        ("游乐项目", ["游乐", "项目", "设施", "过山车"]),
        ("演艺", ["演艺", "巡游", "烟花", "演出"]),
        ("客诉", ["客诉", "投诉", "宾客关系"]),
        ("应急", ["紧急", "应急", "安全", "医疗"]),
        ("会员", ["会员", "年卡", "年票", "尊享"]),
        ("员工", ["员工", "培训", "操作手册", "岗位职责"]),
        ("攻略", ["攻略", "游玩", "路线"]),
    ]
    tags = []
    for tag, kws in keyword_map:
        if any(k in text for k in kws):
            tags.append(tag)
    return tags


def _fix_cjk_spaces(text):
    """清理跨文本框/单元格换行产生的 CJK 断词空格：'度 假村' -> '度假村'。
    仅当空格两侧都是 CJK 字符时才移除，保留中英文间正常空格（'Disney 频道' 保留）。"""
    return re.sub(r"([一-鿿])\s+([一-鿿])", r"\1\2", text or "")


def trim_sparse_cols(rows):
    """裁剪稀疏列（空值率 > SPARSE_COL_EMPTY_RATE）。全库实测仅两类命中：
    1. 仅表头有值的网页按钮列（如票价表"立即预订"）→ 整列连同表头名丢弃；
    2. 偶发值只落在"整行其余列全空"的分组标题行（如时间表中间列的
       "与迪士尼朋友见面/餐厅"）→ 标签提升到保留首列，信息不丢。
    数据行偶发备注则并入该行末列保留行归属（防御性，当前库 0 命中）。
    至少保留一列；行数不足 SPARSE_COL_MIN_ROWS 的小表原样返回。"""
    nrows = len(rows)
    if nrows < SPARSE_COL_MIN_ROWS:
        return rows
    ncols = len(rows[0])
    if ncols < 2:
        return rows
    sparse, dense, header_only = [], [], set()
    for j in range(ncols):
        filled = [i for i, r in enumerate(rows) if r[j].strip()]
        if not filled:
            dense.append(j)  # 理论上上游已裁全空列，防御
            continue
        if 1 - len(filled) / nrows > SPARSE_COL_EMPTY_RATE:
            sparse.append(j)
            if filled == [0]:
                header_only.add(j)  # 只有表头有值：按钮/链接伪列
        else:
            dense.append(j)
    if not sparse or not dense:
        return rows
    out = []
    for ri, r in enumerate(rows):
        sparse_j = [j for j in sparse if r[j].strip()]
        stexts = [r[j].strip() for j in sparse_j]
        dcells = [r[j] for j in dense]
        if not stexts:
            out.append(dcells)
        elif not any(c.strip() for c in dcells):
            # 整行仅稀疏列有值：分组标题行 → 标签落入保留首列
            nr = [""] * len(dense)
            nr[0] = " ".join(stexts)
            out.append(nr)
        elif ri == 0 and all(j in header_only for j in sparse_j):
            # 纯按钮/链接伪列的表头名：随列删除，不并入任何保留单元格
            out.append(dcells)
        else:
            # 数据行偶发稀疏值：并入末列，保留与同行数据的归属关系
            dcells[-1] = (dcells[-1].strip() + " " + " ".join(stexts)).strip()
            out.append(dcells)
    return out


def sanitize_table_rows(rows):
    """清洗表格行，返回 (rows, degenerate)：
    - 单元格：clean + 换行转空格 + 修复 CJK 断词
    - 删除全空行、裁剪全空列（PPT 中图片/logo 占位列取不到文字）
    - 裁剪稀疏列（空值率>80%）：按钮伪列整列删除，分组标题提升为首列标签行
    - degenerate=True：清洗后只剩 1 行且非空单元格 ≤2，
      多为 PDF 标题栏被 find_tables 误判，调用方应降级为文本段落而非表格。
    - giant=True：数据行 ≤2 且含 >300 字巨型单元格——Word 排版用的"文本容器"
      伪表（如 .doc 转换残留的双语长列表单格表），同样应降级为段落流。
    - unbalanced=True：列填充严重失衡（一列几乎全满、其余列≤50%），
      多为 SmartArt 结构图经 COM 转换后的"标签-明细"稀疏表。"""
    rows = [[_fix_cjk_spaces(clean(c)).replace("\n", " ") for c in r] for r in rows]
    if not rows:
        return [], True, False, False
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    rows = [r for r in rows if any(c.strip() for c in r)]  # 删除全空行
    if not rows:
        return [], True, False, False
    keep_cols = [j for j in range(width) if any(r[j].strip() for r in rows)]
    rows = [[r[j] for j in keep_cols] for r in rows]  # 裁剪全空列
    rows = trim_sparse_cols(rows)                    # 裁剪稀疏列/提升分组标题
    nonempty = sum(1 for r in rows for c in r if c.strip())
    degenerate = len(rows) == 1 and nonempty <= 2
    data_rows = rows[1:]
    max_cell = max((len(c) for r in data_rows for c in r), default=0)
    giant = bool(data_rows) and len(data_rows) <= 2 and max_cell > 300
    ncols = len(rows[0])
    rates = sorted((sum(1 for r in rows if r[j].strip()) / len(rows)
                    for j in range(ncols)), reverse=True)
    unbalanced = (ncols >= 2 and rates[0] >= 0.75 and rates[1] <= 0.5)
    return rows, degenerate, giant, unbalanced


def make_table_md(rows):
    """rows: List[List[str]] -> Markdown 表格字符串；空表/伪表格返回 ''。"""
    rows, degenerate, giant, unbalanced = sanitize_table_rows(rows)
    if not rows or degenerate or giant or unbalanced:
        return ""
    width = len(rows[0])
    lines = ["| " + " | ".join(rows[0]) + " |",
             "|" + "---|" * width]
    for r in rows[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def table_to_blocks(rows):
    """清洗后生成 block 列表：正常表格 -> [('table', md)]；
    伪表格（单行标题栏、巨型文本容器、失衡标签-明细表）降级为普通文本段落，
    让内容按句子/条目流切分，避免半句硬切与残缺空列。"""
    clean_rows, degenerate, giant, unbalanced = sanitize_table_rows(rows)
    if not clean_rows:
        return []
    if degenerate:
        txt = " ".join(c.strip() for r in clean_rows for c in r if c.strip())
        return [("p", txt)] if txt else []
    if giant or unbalanced:
        body_text = " ".join(c for r in clean_rows[1:] for c in r)
        out = []
        if giant and not unbalanced:
            # 单/双行巨型格：每个长格独立成段（如英文流、中文流各自完整）
            tail = []
            for r in clean_rows[1:]:
                for c in r:
                    c = c.strip()
                    if not c:
                        continue
                    if len(c) > 50:
                        if tail:
                            out.append(("p", " ".join(tail)))
                            tail = []
                        out.append(("p", c))
                    else:
                        tail.append(c)
            if tail:
                out.append(("p", " ".join(tail)))
        else:
            # 失衡标签-明细表：逐行成段（行内标签/明细换行连接）
            for ri, r in enumerate(clean_rows):
                cells, prev = [], None
                for c in r:
                    c = c.strip()
                    if c and c != prev:  # 去除合并单元格产生的横向重复文本
                        cells.append(c)
                    prev = c
                if not cells:
                    continue
                # 首行无列名语义：所有单元格同值（重复主题词，如文档名×N）→ 丢弃
                if ri == 0 and (len(set(cells)) == 1
                                or all(c in body_text for c in cells)):
                    continue
                out.append(("p", "\n".join(cells)))
        return out
    md = make_table_md(clean_rows)
    return [("table", md)] if md else []


# ---------------------------------------------------------------- DOCX 解析

def parse_docx(path):
    """按 body 顺序提取 标题/段落/表格 block。"""
    from docx import Document
    from docx.oxml.ns import qn

    doc = Document(path)
    p_map = {p._element: p for p in doc.paragraphs}
    t_map = {t._element: t for t in doc.tables}
    blocks = []

    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            p = p_map.get(child)
            if p is None:
                continue
            text = clean(p.text)
            if not text:
                continue
            level = heading_level(p)
            if level:
                blocks.append({"kind": "h", "text": text, "level": level})
            else:
                blocks.append({"kind": "p", "text": text})
        elif child.tag == qn("w:tbl"):
            t = t_map.get(child)
            if t is None:
                continue
            rows = [[c.text for c in row.cells] for row in t.rows]
            for kind, txt in table_to_blocks(rows):
                blocks.append({"kind": kind, "text": txt})
    return blocks


def heading_level(paragraph):
    """返回标题级别(1-9)，非标题返回 None。兼容英文/中文样式名与 outlineLvl。"""
    name = (paragraph.style.name or "") if paragraph.style else ""
    if "Heading" in name or "标题" in name or "heading" in name:
        m = re.search(r"(\d+)", name)
        return int(m.group(1)) if m else 1
    pPr = paragraph._p.pPr
    if pPr is not None and pPr.find(
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}outlineLvl") is not None:
        node = pPr.find(
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}outlineLvl")
        try:
            return int(node.get(
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val")) + 1
        except Exception:
            return 1
    return None


# ---------------------------------------------------------------- PPTX 解析

def _is_chart_data_table(rows):
    """判断表格是否为 PPT 图表内嵌的"编辑数据"表（数字密集型，非业务表格）。
    判定规则（满足任一即视为图表数据表）：
      1. 列数 >= 6：业务表格通常 2-5 列，图表数据表往往很宽
      2. 非空单元格中纯数字占比 > 40%：业务表格以文字为主"""
    if not rows:
        return True
    width = max(len(r) for r in rows)
    if width >= 6:
        return True
    cells = [c.strip() for r in rows for c in r if c and c.strip()]
    if not cells:
        return True
    import re
    num_cells = sum(1 for c in cells if re.fullmatch(r"-?\d+(\.\d+)?", c.strip()))
    return num_cells / len(cells) > 0.40


_PURE_NUM_RE = None


def _is_pure_number(text):
    """PPT 中独立成行的纯数字通常是图表坐标轴刻度/数据点，无独立阅读价值。"""
    global _PURE_NUM_RE
    if _PURE_NUM_RE is None:
        _PURE_NUM_RE = re.compile(r"-?\d+(\.\d+)?")
    return bool(_PURE_NUM_RE.fullmatch((text or "").strip()))


def _iter_shapes(shapes):
    """递归展开组合形状中的普通形状。"""
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    for shape in shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from _iter_shapes(shape.shapes)
        else:
            yield shape


def parse_pptx(path):
    """每张幻灯片产出：boundary + h1(幻灯片标题) + 正文段落/表格/备注。"""
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    prs = Presentation(path)
    blocks = []
    for idx, slide in enumerate(prs.slides, 1):
        blocks.append({"kind": "boundary"})
        title = ""
        try:
            if slide.shapes.title and slide.shapes.title.text:
                title = clean(slide.shapes.title.text)
        except Exception:
            pass
        blocks.append({"kind": "h", "text": f"幻灯片{idx}：{title}" if title else f"幻灯片{idx}",
                       "level": 1})
        title_shape = None
        try:
            title_shape = slide.shapes.title
        except Exception:
            pass
        for shape in _iter_shapes(slide.shapes):
            if shape is title_shape:
                continue
            # 跳过图表形状：其文本框是数据点/轴标签，逐行输出会产生数字垃圾
            if getattr(shape, "shape_type", None) == MSO_SHAPE_TYPE.CHART:
                continue
            if getattr(shape, "has_table", False) and shape.has_table:
                rows = [[c.text for c in row.cells] for row in shape.table.rows]
                # 过滤图表内嵌数据表（数字密集型），避免产生不可读的乱表
                if _is_chart_data_table(rows):
                    continue
                for kind, txt in table_to_blocks(rows):
                    blocks.append({"kind": kind, "text": txt})
            elif getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                joined = join_vertical_text(shape.text_frame.text)
                for line in joined.split("\n"):
                    line = _fix_cjk_spaces(clean(line))
                    # 跳过纯数字行（图表坐标轴刻度/数据点），保留带文字的有意义行
                    if not line or line == title or _is_pure_number(line):
                        continue
                    blocks.append({"kind": "p", "text": line})
        try:
            notes = clean(slide.notes_slide.notes_text_frame.text)
            if notes:
                blocks.append({"kind": "p", "text": f"演讲者备注：{notes}"})
        except Exception:
            pass
    return blocks


# ---------------------------------------------------------------- PDF 解析

_OCR_ENGINE = None
OCR_CACHE_FILE = os.path.join(OUT_DIR, "ocr_cache.json")
_OCR_META_KEY = "__cache_meta__"


def get_ocr_engine():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def _ocr_cache_meta():
    """OCR 缓存指纹：引擎名+安装版本+结构版本。引擎换代（包名变更）时缓存作废。"""
    ver = "unknown"
    try:
        from importlib.metadata import version
        ver = version("rapidocr_onnxruntime")
    except Exception:
        pass
    return {"engine": "rapidocr_onnxruntime", "engine_ver": ver,
            "schema": "ocr_cache_v1"}


def _load_ocr_cache():
    """加载 OCR 缓存。无指纹旧缓存：信任并零成本回写补指纹；
    指纹显示引擎换代（包名不同）：整份作废，避免跨引擎文本混用。"""
    try:
        with open(OCR_CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except Exception:
        return {}
    if not isinstance(cache, dict):
        return {}
    meta = cache.get(_OCR_META_KEY)
    if meta is None:
        if any(k != _OCR_META_KEY for k in cache):
            print("[ocr_cache] 无指纹旧缓存，信任并回写引擎指纹")
            _save_ocr_cache(cache)  # 内容不动，仅补 meta
        return cache
    if meta.get("engine") != _ocr_cache_meta()["engine"]:
        print(f"[ocr_cache] 引擎换代 {meta.get('engine')}，缓存作废")
        return {}
    return cache


def _save_ocr_cache(cache):
    os.makedirs(OUT_DIR, exist_ok=True)
    payload = dict(cache)
    payload[_OCR_META_KEY] = _ocr_cache_meta()
    tmp = OCR_CACHE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, OCR_CACHE_FILE)


def ocr_page(page, pdf_path, pno):
    import numpy as np
    cache = _load_ocr_cache()
    key = os.path.normcase(os.path.abspath(pdf_path))
    pages = cache.get(key, {})
    ck = str(pno)
    if ck in pages:
        return pages[ck]
    engine = get_ocr_engine()
    pix = page.get_pixmap(dpi=200)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = img[:, :, :3]
    result, _ = engine(img)
    text = "\n".join(item[1] for item in result if len(item) > 1) if result else ""
    pages[ck] = text
    cache[key] = pages
    _save_ocr_cache(cache)
    return text


def _dedup_tables(tables):
    """按 bbox 去重（find_tables 偶尔对同一区域重复检测）。"""
    seen, out = set(), []
    for t in tables:
        key = tuple(round(v) for v in t.bbox)
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


# ---------------------------------------------------------------- PDF 章节/噪声识别

def _is_heading_like(s, max_len=10):
    """短标题判定：单行、2-max_len 字、含中文、不含句读/冒号等标点。"""
    s = (s or "").strip()
    if "\n" in s or not (2 <= len(s) <= max_len):
        return False
    if re.search(r"[。！？，；、：:？?!]", s):
        return False
    return bool(re.search(r"[一-鿿]", s))


# 标题中通常不会出现的虚/功能字，用于形态启发，避免把半句正文当章节
_TITLE_STOP_CHARS = set("的了在于和与及或但是有我你他它们这那也都就还又被把让向从对为以之其此")


def _looks_like_section_title(s, vocab):
    """章节标题：词典命中直接通过；否则要求 3-10 字、纯中英数字、无标点、无虚词。"""
    s = (s or "").strip()
    if s in vocab:
        return True
    if not (3 <= len(s) <= 10) or "\n" in s:
        return False
    if not re.fullmatch(r"[一-鿿A-Za-z0-9·]+", s):
        return False
    if not re.search(r"[一-鿿]", s):
        return False
    return not any(ch in _TITLE_STOP_CHARS for ch in s)


def _is_pdf_footer(text, y0=None, page_height=None):
    """旅游攻略类 PDF 的页脚：'返回目录 我要纠错'，或页面底部的纯数字页码。"""
    s = (text or "").strip()
    if re.sub(r"\s+", "", s) in {"返回目录我要纠错"}:
        return True
    if re.fullmatch(r"\d{1,3}", s):
        # 有坐标时仅认定底部 12% 区域内的数字为页码，避免误杀正文
        if y0 is None or page_height is None:
            return True
        return y0 > page_height * 0.88
    return False


def _strip_footer_lines(text, drop_number=False):
    """行级剔除混入正文块的页脚行（'返回目录 我要纠错'，可选独立数字页码行）。"""
    kept = []
    for ln in (text or "").split("\n"):
        s = ln.strip()
        if re.sub(r"\s+", "", s) == "返回目录我要纠错":
            continue
        if drop_number and re.fullmatch(r"\d{1,3}", s):
            continue
        kept.append(ln)
    return "\n".join(kept).strip()


def _is_header_title(text, doc_name, name_key, y0):
    """顶部页眉中的文档名噪声块（如每页顶部的'香港迪士尼乐园旅游攻略'）。"""
    t = (text or "").strip()
    if y0 is None or y0 > 45 or not t:
        return False
    if t == doc_name:
        return True
    return bool(name_key) and name_key in t and len(t) <= len(doc_name) + 6


def _is_prose_mistable(rows):
    """PyMuPDF 把双栏并排正文误判为表格：存在 >60 字且含句末标点的单元格。
    全库实测：真实表格（门票/巴士/时刻表）最长单元格 186 字但只用冒号逗号，
    无双栏误判表之外的任何命中。"""
    for r in rows:
        for c in r:
            c = clean(c)
            if len(c) > 60 and re.search(r"[。；]", c):
                return True
    return False


def _collect_pdf_sections(doc, doc_name):
    """预扫描章节标题词典，两个来源：
    1) 页眉区：文档名块(y0<45)下方 45-140 高度处的短标题块（如'探险世界/住宿/乐园服务'）
    2) 目录页：按 '/' 与 2+ 空格切出的 2-10 字中文短条目（覆盖页内小标题）。"""
    vocab = set()
    name_key = re.sub(r"[（(].*?[)）]", "", doc_name).strip()[:4]
    for page in doc:
        text = clean(page.get_text("text"))
        if len(norm(text)) < SCANNED_TEXT_THRESHOLD:
            continue
        # 本页表格区域，词典收集时跳过，避免表头/单元格污染
        tboxes = []
        try:
            import pymupdf as fitz
            tboxes = [fitz.Rect(t.bbox) for t in _dedup_tables(page.find_tables().tables)]
        except Exception:
            pass

        def _in_table(bbox):
            cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            return any(r.x0 <= cx <= r.x1 and r.y0 <= cy <= r.y1 for r in tboxes)

        has_header = False
        for b in page.get_text("blocks"):
            if b[-1] != 0 or _in_table(b[:4]):
                continue
            t = clean(b[4])
            if not t:
                continue
            if _is_header_title(t, doc_name, name_key, b[1]):
                has_header = True
            elif has_header and 45 <= b[1] <= 140 and _is_heading_like(t):
                vocab.add(t)
        # 目录页：按 '/' 与 2+ 空格切出短条目；不收以数字开头的价格项
        if "目录" in text[:300] or "content" in text[:300].lower():
            for b in page.get_text("blocks"):
                if b[-1] != 0:
                    continue
                for seg in re.split(r"[/／•]|\s{2,}", clean(b[4])):
                    seg = seg.strip()
                    if not (2 <= len(seg) <= 10) or seg[0].isdigit():
                        continue
                    if seg in {"目录", "content", "Content", "CONTENT"}:
                        continue
                    if (re.search(r"[一-鿿]", seg)
                            and not re.search(r"[。！？，；、：:？]", seg)
                            and re.fullmatch(r"[一-鿿（）()A-Za-z0-9]+", seg)):
                        vocab.add(seg)
    return vocab, name_key


def _classify_pdf_text(text, vocab, name_key, doc_name="", y0=None,
                       page_height=None, is_toc=False):
    """把一个 PDF 文本块分类为 [(kind, payload), ...]；噪声/页码返回 []。
    - 坐标块（blocks 模式）：顶部文档名页眉、底部页码过滤
    - 拼接段（text 模式）：段首文档名行剥离，独立页码行按行剔除
    - 首行为章节标题（词典或形态启发）→ section(+ 剩余正文 p)"""
    t = (text or "").strip()
    if not t:
        return []
    if _is_pdf_footer(t, y0, page_height):
        return []
    if y0 is not None and _is_header_title(t, doc_name, name_key, y0):
        return []
    is_coord = y0 is not None
    t = _strip_footer_lines(t, drop_number=not is_coord)
    if not t:
        return []

    def _is_doc_name_line(s):
        s = s.strip()
        if s == doc_name:
            return True
        return bool(name_key) and name_key in s and len(s) <= len(doc_name) + 6

    lines = t.split("\n")
    if not is_coord:
        while lines and _is_doc_name_line(lines[0]):
            lines.pop(0)
        if not lines:
            return []

    title = lines[0].strip()
    # blocks 模式只认定独立单行短块；text 分段模式允许"首行标题+正文"同段拆分
    # 目录页条目不标 section（无正文归属的空标题）
    if (not is_toc and _looks_like_section_title(title, vocab)
            and (not is_coord or len(lines) == 1)):
        out = [("section", title)]
        rest = "\n".join(lines[1:]).strip()
        if rest:
            out.append(("p", rest))
        return out
    return [("p", "\n".join(lines).strip())]



def parse_pdf(path):
    """逐页提取；文本页用 find_tables 结构化表格为 Markdown，扫描页走 OCR。
    章节感知：预扫描页眉路径/目录构建章节词典，识别正文中的章节标题（section），
    使切片沿文档语义章节（如主题园区/交通/住宿）成块而非纯 800 字硬切。"""
    import pymupdf as fitz

    blocks = []
    doc = fitz.open(path)
    doc_name = os.path.splitext(os.path.basename(path))[0]
    section_vocab, name_key = _collect_pdf_sections(doc, doc_name)
    for pno in range(doc.page_count):
        page = doc[pno]
        text = clean(page.get_text("text"))
        page_blocks = []  # List[(kind, text)]
        is_toc = bool(re.search(
            r"目录.{0,6}content", text[:1500], re.I | re.S))
        if len(norm(text)) >= SCANNED_TEXT_THRESHOLD:
            # 1) 结构化表格提取（过滤：伪表格/双栏正文误判/空表）
            tables = []
            try:
                tables = _dedup_tables(page.find_tables().tables)
            except Exception as e:
                print(f"      [表格提取失败] 第{pno + 1}页: {e}")
            table_mds = []
            for t in tables:
                raw_rows = t.extract()
                if _is_prose_mistable(raw_rows):
                    continue  # 双栏并排正文误判：不进表格通道，文本按普通块保留
                md = make_table_md(raw_rows)
                if md and norm(md):
                    table_mds.append((fitz.Rect(t.bbox), md))

            if table_mds:
                # 2) 按纵向位置合并表格与表格外文本块
                tboxes = [b for b, _ in table_mds]
                items = []
                for blk in page.get_text("blocks"):
                    if blk[-1] != 0:  # 只要文本块
                        continue
                    r = fitz.Rect(blk[:4])
                    area = r.get_area()
                    inter = max(((r & tb).get_area() / area) if area else 0
                                for tb in tboxes)
                    if inter > 0.3:  # 位于表格区域内的文本块跳过
                        continue
                    bt = clean(blk[4])
                    for kind, payload in _classify_pdf_text(
                            bt, section_vocab, name_key, doc_name,
                            blk[1], page.rect.height, is_toc):
                        items.append((blk[1], kind, payload))
                for tb, md in table_mds:
                    items.append((tb.y0, "table", md))
                items.sort(key=lambda x: x[0])
                page_blocks = [(k, t) for _, k, t in items]
                print(f"      [表格] 第{pno + 1}页提取 {len(table_mds)} 个结构化表格")
            else:
                # 无表格：按空行分段（原逻辑），同样走章节/页脚分类
                for para in re.split(r"\n\s*\n", text):
                    para = clean("\n".join(ln.strip() for ln in para.splitlines() if ln.strip()))
                    page_blocks.extend(_classify_pdf_text(
                        para, section_vocab, name_key, doc_name,
                        is_toc=is_toc))
        else:
            # 疑似扫描页：有图像才 OCR
            if page.get_images(full=True) or page.get_drawings():
                try:
                    ocr_text = clean(ocr_page(page, path, pno + 1))
                    for para in re.split(r"\n+", ocr_text):
                        para = clean(para)
                        if para:
                            page_blocks.append(("p", para))
                    print(f"      [OCR] 第{pno + 1}页识别出 {len(norm(ocr_text))} 字")
                except Exception as e:
                    print(f"      [OCR失败] 第{pno + 1}页: {e}")
            if not page_blocks:
                print(f"      [警告] 第{pno + 1}页无可提取文本")
        if page_blocks:
            blocks.append({"kind": "h", "text": f"第{pno + 1}页", "level": 9})
            for kind, payload in page_blocks:
                blocks.append({"kind": kind, "text": payload})
    doc.close()
    return blocks


def scan_pdf_pages(path):
    """只检测扫描页，不做 OCR。返回 (总页数, 疑似扫描页数列表)。"""
    import pymupdf as fitz
    doc = fitz.open(path)
    scanned = []
    for pno in range(doc.page_count):
        page = doc[pno]
        text = clean(page.get_text("text"))
        if len(norm(text)) < SCANNED_TEXT_THRESHOLD:
            if page.get_images(full=True) or page.get_drawings():
                scanned.append(pno + 1)
    total = doc.page_count
    doc.close()
    return total, scanned


# ---------------------------------------------------------------- 切片器

def hard_split(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """对超长段落做递归边界切分，尽量在标点处断开。"""
    separators = ["\n", "。", "！", "？", "；", ". ", "，", "、", " ", ""]
    pieces = []

    def split_rec(s):
        if len(s) <= size:
            if s.strip():
                pieces.append(s.strip())
            return
        for sep in separators:
            if sep == "":
                window = s[:size]
                pieces.append(window.strip())
                nxt = s[size - overlap:]
                if nxt.strip():
                    split_rec(nxt)
                return
            # 在 [size*0.55, size] 窗口内找最后一个分隔符
            tail = s[:size]
            pos = tail.rfind(sep, int(size * 0.55))
            if pos != -1:
                cut = pos + len(sep)
                pieces.append(s[:cut].strip())
                nxt = s[max(0, cut - overlap):]
                if nxt.strip() and nxt != s:
                    split_rec(nxt)
                return

    split_rec(text)
    return pieces


def split_long_table(md, size=CHUNK_SIZE):
    """超长 Markdown 表按行分组，每组重复表头两行；单行超长时按字符硬切。"""
    lines = md.split("\n")
    if len(md) <= size:
        return [md]
    if len(lines) <= 2:  # 只有正文没有标准表头
        return hard_split(md, size, CHUNK_OVERLAP)
    header = lines[:2]
    header_text = "\n".join(header)
    data = lines[2:]
    groups = []
    cur, cur_len = [], 0

    def flush_group():
        if not cur:
            return
        group_text = header_text + "\n" + "\n".join(cur)
        if len(group_text) <= size:
            groups.append(group_text)
        else:
            # 组内存在超长行：正文按字符切，每片都重复表头
            budget = size - len(header_text) - 1
            for piece in hard_split("\n".join(cur), budget, CHUNK_OVERLAP):
                groups.append(header_text + "\n" + piece)

    for row in data:
        if cur and cur_len + len(row) + 1 > size:
            flush_group()
            cur, cur_len = [], 0
        cur.append(row)
        cur_len += len(row) + 1
    flush_group()
    return groups


class Chunker:
    MIN_CHUNK = 8  # M3: 正文短于此且非表格则丢弃

    def __init__(self):
        self.records = []          # 去重后的 chunk
        self._hash2idx = {}
        self.dup_hits = 0
        self.skipped = 0
        self.merged_short = 0      # P1-7: 短块合并计数
        # 单文档流状态
        self.stack = []
        self.buf = []
        self.buf_len = 0
        self.seq = 0
        self.doc_order = []        # I1: 当前文档内 chunk 的 records 下标顺序，用于父子上下文

    def breadcrumb(self):
        return " > ".join(t for _, t in self.stack)

    def begin_doc(self, title):
        self.stack = [(0, title)]
        self.buf, self.buf_len, self.seq = [], 0, 0
        self.doc_order = []

    def _overlap_seed(self, body):
        if CHUNK_OVERLAP <= 0 or len(body) <= CHUNK_OVERLAP:
            return None
        tail = body[-CHUNK_OVERLAP:]
        m = re.search(r"[。！？；\n]", tail)
        if m and m.end() < len(tail):
            tail = tail[m.end():]
        tail = tail.strip()
        return tail if tail else None

    def emit(self, body, source):
        body = body.strip()
        if not body:
            return
        # M3: 丢弃过短且非表格的无意义 chunk（如 PPT 标题页"乐园"）
        is_table = "|---|" in body or "| --- |" in body
        if not is_table and len(body) < self.MIN_CHUNK:
            self.skipped += 1
            return
        h = chunk_hash(body)
        if h in self._hash2idx:
            rec = self.records[self._hash2idx[h]]
            if source not in rec["sources"]:
                rec["sources"].append(source)
            self.dup_hits += 1
            return
        crumb = self.breadcrumb()
        source = dict(source, heading=crumb)
        # M2: 标签
        fname = os.path.splitext(os.path.basename(source.get("file", "")))[0]
        tags = extract_tags(fname, source.get("category", ""), crumb)
        source["tags"] = tags
        content = f"{crumb}\n{body}" if crumb else body
        idx = len(self.records)
        self._hash2idx[h] = idx
        self.records.append({
            "type": "text",
            "content": content,
            "body": body,          # I1: 纯正文，用于父子上下文拼接
            "body_chars": len(body),
            "hash": h,
            "sources": [source],
            "tags": tags,
        })
        self.doc_order.append(idx)

    def flush(self, source):
        if not self.buf:
            return
        body = "\n".join(self.buf)
        self.seq += 1
        src = dict(source, chunk_seq=self.seq)
        self.emit(body, src)
        seed = self._overlap_seed(body)
        self.buf = [seed] if seed else []
        self.buf_len = len(seed) if seed else 0

    def feed(self, blocks, source):
        for b in blocks:
            kind = b["kind"]
            if kind == "boundary":
                self.flush(source)
                title = self.stack[0][1] if self.stack else ""
                self.stack = [(0, title)]
                continue
            if kind == "h":
                lvl, text = b["level"], b["text"]
                while self.stack and self.stack[-1][0] >= lvl:
                    self.stack.pop()
                self.stack.append((lvl, text))
                continue
            if kind == "section":
                # PDF 语义章节（如主题园区"探险世界"、交通/住宿）：先收块再切边界
                self.flush(source)
                self.buf, self.buf_len = [], 0  # 章节边界不带重叠，防止跨章节串味
                while self.stack and self.stack[-1][0] >= 2:
                    self.stack.pop()
                self.stack.append((2, b["text"]))
                continue
            if kind == "table":
                self.flush(source)
                for piece in split_long_table(b["text"]):
                    self.seq += 1
                    self.emit(piece, dict(source, chunk_seq=self.seq))
                continue
            # 普通段落
            text = b["text"]
            if len(text) > CHUNK_SIZE:
                self.flush(source)
                for piece in hard_split(text):
                    self.seq += 1
                    self.emit(piece, dict(source, chunk_seq=self.seq))
                continue
            if self.buf_len + len(text) + 1 > CHUNK_SIZE:
                self.flush(source)
            self.buf.append(text)
            self.buf_len += len(text) + 1

    @staticmethod
    def _is_table_body(body):
        return "|---|" in body or "| --- |" in body

    @staticmethod
    def _is_empty_header(body):
        """N4: 纯标题/目录/序号等无正文短块（防御性，当前库 0 命中）。"""
        s = body.strip()
        if not s or len(s) > 80:
            return False
        if s in {"目录", "目  录", "Contents", "CONTENTS", "版权", "版权所有"}:
            return True
        if re.fullmatch(r"[一二三四五六七八九十]+[、.．]", s):
            return True
        if re.fullmatch(r"[，。、；：？！…—\s]+", s):
            return True
        return False

    @staticmethod
    def _is_watermark_notice(body):
        """N4: 独立成块的版权/URL 水印（严格条件，避免误杀正文中的引用）。
        实测库内 0 命中（CRM 手册中的内网地址、清单中的 © 标记均为合法内容）。"""
        s = body.strip()
        if not s or len(s) > 120:
            return False
        notice = re.search(r"(https?://\S+|www\.\S+|©\s?\d{4}|版权所有|All Rights Reserved|Copyright\s*\d{4})", s, re.I)
        if not notice:
            return False
        # 去掉通告片段后剩余的实质中文/字母不足 12 字符，才判定为纯水印块
        rest = re.sub(r"https?://\S+|www\.\S+|©\s?\d{4}|版权所有|All Rights Reserved|Copyright\s*\d{4}",
                      "", s, flags=re.I)
        return len(re.findall(r"[\u4e00-\u9fffA-Za-z]", rest)) < 12

    @staticmethod
    def _is_ocr_garbage(body):
        """N4: PDF 扫描 OCR 垃圾块——足量长度但有效字符（中日韩/字母数字）占比过低。
        阈值经全库扫描标定：0.35 时仅命中 1 块（纯符号乱码），零误杀。"""
        if len(body) < 40:
            return False
        meaningful = len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", body))
        return meaningful / len(body) < 0.35

    @staticmethod
    def _is_noise(body):
        """P0-4/N4: 解析噪声判定——docx 导出占位、空表头/标题、页脚、水印通告、OCR 乱码。"""
        s = body.strip()
        if not s:
            return True
        # 「表格\n复制」类导出占位
        if re.search(r"表格\s*\n\s*复制", body) and len(s) < 80:
            return True
        # 空表头碎片（仅分隔符/数字，无实质文字）
        if re.fullmatch(r"[\s|｜\-—:：\d.]+", s) and len(s) < 60:
            return True
        # 页脚模板
        if re.match(r"^第\s*\d+\s*页\s*(共|/)\s*\d+\s*页$", s) or re.match(r"^Page\s+\d+\s+of\s+\d+$", s, re.I):
            return True
        if Chunker._is_empty_header(body) or Chunker._is_watermark_notice(body) or Chunker._is_ocr_garbage(body):
            return True
        return False

    def _absorb(self, target_idx, src_idx):
        """把 src 块并入 target 块（target 保留自己的面包屑标题），同步 hash/来源。"""
        tgt, src = self.records[target_idx], self.records[src_idx]
        tgt["body"] = tgt["body"] + "\n" + src["body"]
        tgt["body_chars"] = len(tgt["body"])
        crumb_line = tgt["content"].split("\n", 1)[0] if "\n" in tgt["content"] else ""
        tgt["content"] = f"{crumb_line}\n{tgt['body']}" if crumb_line else tgt["body"]
        for s in src["sources"]:
            if s not in tgt["sources"]:
                tgt["sources"].append(s)
        self._hash2idx.pop(src["hash"], None)
        self._hash2idx.pop(tgt["hash"], None)
        tgt["hash"] = chunk_hash(tgt["body"])
        self._hash2idx[tgt["hash"]] = target_idx
        src["_merged_away"] = True
        self.merged_short += 1

    @staticmethod
    def _same_heading(a, b):
        ha = (a.get("sources") or [{}])[0].get("heading", "")
        hb = (b.get("sources") or [{}])[0].get("heading", "")
        return ha == hb

    def finish(self, source):
        self.flush(source)
        # P1-7/N4: 短块合并（仅非表格块；可后向并入前一块，或前向并入后一块）
        # 规则：表格永远整块保留，不与正文互并；前一块为表格或超长时，短块作为
        # pending 前向并入下一块；两侧都无法合并（如被表格夹）才保留独立短块。
        kept, pending = [], None
        for idx in self.doc_order:
            rec = self.records[idx]
            body = rec.get("body", "")
            # P0-4: 过滤解析噪声（表格/复制占位、空表头、页脚），不进入合并流程
            if self._is_noise(body):
                rec["_merged_away"] = True
                continue
            is_table = self._is_table_body(body)
            short = (not is_table) and rec.get("body_chars", 0) < SHORT_MERGE_LEN
            # 先处理挂起的前向短块
            if pending is not None:
                pend = self.records[pending]
                if (not is_table
                        and self._same_heading(pend, rec)
                        and pend["body_chars"] + rec["body_chars"] + 2 <= MAX_MERGED_LEN):
                    # pending 作为前缀并入当前块
                    pend_body, pend_rec = pend["body"], pend
                    rec["body"] = pend_body + "\n" + rec["body"]
                    rec["body_chars"] = len(rec["body"])
                    crumb_line = rec["content"].split("\n", 1)[0] if "\n" in rec["content"] else ""
                    rec["content"] = f"{crumb_line}\n{rec['body']}" if crumb_line else rec["body"]
                    for s in pend_rec["sources"]:
                        if s not in rec["sources"]:
                            rec["sources"].append(s)
                    self._hash2idx.pop(pend_rec["hash"], None)
                    self._hash2idx.pop(rec["hash"], None)
                    rec["hash"] = chunk_hash(rec["body"])
                    self._hash2idx[rec["hash"]] = idx
                    pend_rec["_merged_away"] = True
                    self.merged_short += 1
                    pending = None
                else:
                    kept.append(pending)  # 下一块是表格/超长，挂起块无法前向合并
                    pending = None
            if is_table or not short:
                kept.append(idx)
                continue
            # 当前为短块：优先后向并入前一保留块（必须同为非表格）
            if kept:
                prev_idx = kept[-1]
                prev = self.records[prev_idx]
                if (not self._is_table_body(prev.get("body", ""))
                        and self._same_heading(prev, rec)
                        and prev["body_chars"] + rec["body_chars"] + 2 <= MAX_MERGED_LEN):
                    self._absorb(prev_idx, idx)
                    continue
            pending = idx  # 后向不可行（文档首块/前为表格/超长）→ 等待前向合并
        if pending is not None:
            kept.append(pending)
        self.doc_order = kept
        # I1: 父子 chunk —— 为当前文档每个 chunk 填充 parent_context（前一块 + 本块 + 后一块）
        for pos, idx in enumerate(self.doc_order):
            rec = self.records[idx]
            parts = []
            if pos > 0:
                prev = self.records[self.doc_order[pos - 1]]
                if prev.get("body"):
                    parts.append(prev["body"][-300:])  # 上一块尾部
            parts.append(rec.get("body", ""))
            if pos < len(self.doc_order) - 1:
                nxt = self.records[self.doc_order[pos + 1]]
                if nxt.get("body"):
                    parts.append(nxt["body"][:300])   # 下一块头部
            rec["parent_context"] = "\n\n".join(p for p in parts if p)


# ---------------------------------------------------------------- 文件调度

def build_inventory():
    """构造 (原始相对路径, 实际解析路径, 格式) 列表。"""
    with open(MANIFEST_FILE, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    # normcase 归一化，规避盘符大小写(D: vs d:)不一致导致的映射失败
    manifest = {os.path.normcase(m["original"]): m["converted"] for m in raw}

    items, images = [], []
    for dirpath, _, files in os.walk(KB_ROOT):
        for fn in sorted(files):
            if fn.startswith("."):
                continue
            fp = os.path.join(dirpath, fn)
            rel = os.path.relpath(fp, KB_ROOT)
            ext = os.path.splitext(fn)[1].lower()
            category = rel.split(os.sep)[0] if os.sep in rel else "根目录（总览）"
            if ext == ".docx":
                items.append((rel, fp, "docx", category))
            elif ext == ".pptx":
                items.append((rel, fp, "pptx", category))
            elif ext == ".pdf":
                items.append((rel, fp, "pdf", category))
            elif ext in (".jpeg", ".jpg", ".png"):
                images.append((rel, fp, category))
            elif ext in (".doc", ".ppt"):
                converted = manifest.get(os.path.normcase(fp))
                if converted and os.path.exists(converted):
                    fmt = "docx" if ext == ".doc" else "pptx"
                    items.append((rel, converted, fmt, category))
                else:
                    print(f"[缺失转换] {rel}")
    return items, images


# ---------------------------------------------------------------- 构建完成度报告
BUILD_REPORT_FILE = os.path.join(OUT_DIR, "build_report.json")


def file_md5(path, limit_prefix=12):
    if not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:limit_prefix]


def update_build_report(stage, payload):
    """机器可读构建报告（各阶段脚本读-改-写，原子替换）。
    stage ∈ stage1_parse / stage3_image_textify / stage2_index；
    stage2 同时汇总 overall 一致性裁决。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    doc = {}
    if os.path.exists(BUILD_REPORT_FILE):
        try:
            with open(BUILD_REPORT_FILE, encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            doc = {}
    doc.setdefault("stages", {})[stage] = {"ts": int(time.time()), **payload}
    tmp = BUILD_REPORT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    os.replace(tmp, BUILD_REPORT_FILE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-pdf", action="store_true", help="只检测 PDF 扫描页")
    args = parser.parse_args()

    items, images = build_inventory()
    pdf_items = [it for it in items if it[2] == "pdf"]

    if args.scan_pdf:
        print("=== PDF 扫描页检测 ===")
        need_ocr = False
        for rel, fp, fmt, cat in pdf_items:
            total, scanned = scan_pdf_pages(fp)
            flag = f"，疑似扫描页: {scanned}" if scanned else "，全部为文本页"
            print(f"  {rel}（共{total}页）{flag}")
            if scanned:
                need_ocr = True
        if need_ocr:
            print("\n=> 存在扫描页，正式运行时将自动调用 OCR（需 rapidocr-onnxruntime）")
        else:
            print("\n=> 无需 OCR")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    chunker = Chunker()
    parse_errors = []
    counts = {"docx": 0, "pptx": 0, "pdf": 0}

    for rel, fp, fmt, category in items:
        title = os.path.splitext(os.path.basename(rel))[0]
        print(f"[{fmt}] {rel}")
        try:
            if fmt == "docx":
                blocks = parse_docx(fp)
            elif fmt == "pptx":
                blocks = parse_pptx(fp)
            else:
                blocks = parse_pdf(fp)
            counts[fmt] += 1
        except Exception as e:
            parse_errors.append((rel, str(e)))
            print(f"  [解析失败] {e}")
            continue

        chunker.begin_doc(title)
        source = {"file": rel, "category": category, "orig_format": fmt}
        chunker.feed(blocks, source)
        chunker.finish(source)
        print(f"    累计 {len(chunker.records)} chunk（去重命中 {chunker.dup_hits}）")

    # 赋 id 并写出（过滤被合并掉的短块）
    kept = [r for r in chunker.records if not r.get("_merged_away")]
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for cid, rec in enumerate(kept):
            rec["id"] = cid
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    lens = sorted(r["body_chars"] for r in kept)
    print("\n=== 切片统计 ===")
    print(f"解析成功文档: docx={counts['docx']}, pptx={counts['pptx']}, pdf={counts['pdf']}")
    print(f"图片(在入库阶段处理): {len(images)} 张")
    print(f"去重后 chunk 总数: {len(kept)}，重复命中: {chunker.dup_hits}，短块合并: {chunker.merged_short}")
    small = sum(1 for L in lens if L < 100)
    print(f"剩余 <100 字短块: {small}（多为整块保留的表格）")
    if lens:
        n = len(lens)
        print(f"chunk 长度(正文字符): min={lens[0]}, 中位={lens[n // 2]}, max={lens[-1]}")
    cat_count = {}
    for r in chunker.records:
        for s in r["sources"]:
            cat_count[s["category"]] = cat_count.get(s["category"], 0) + 1
    print("各分类来源引用次数:", json.dumps(cat_count, ensure_ascii=False))
    if parse_errors:
        print(f"\n解析失败 {len(parse_errors)} 个:")
        for rel, e in parse_errors:
            print(f"  - {rel}: {e}")
    print(f"\n已写出: {os.path.relpath(OUT_FILE, HERE)}")

    # 机器可读构建报告（stage1）
    n_sources = len({s["file"] for r in kept for s in r.get("sources", [])})
    update_build_report("stage1_parse", {
        "chunks_n": len(kept), "sources_n": n_sources,
        "images_discovered": len(images),
        "parsed_docs": counts, "parse_errors_n": len(parse_errors),
        "parse_errors": [{"file": rel, "error": str(e)[:200]}
                         for rel, e in parse_errors],
        "dup_hits": chunker.dup_hits, "merged_short": chunker.merged_short,
        "chunks_md5": file_md5(OUT_FILE),
        "ok": len(kept) > 0 and not parse_errors,
    })


if __name__ == "__main__":
    main()

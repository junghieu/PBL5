import io
import re
import unicodedata
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# Vùng in vật lý của tờ A4 trong Word (twips)
_A4_PAGE_WIDTH_TWIPS = 9026

# Indent ngữ nghĩa theo loại nội dung (inch)
_INDENT_PARAGRAPH = 0.5
_INDENT_PLAIN     = 0.5

# Ngưỡng x/page_w để fallback detect 2 cột
_TWO_COL_RIGHT_THRESHOLD = 0.40

# Patterns
_QUESTION_RE      = re.compile(r'^\s*(Câu\s*\d+|[IVX]+\.|\d+\.)', re.UNICODE)
_NUMBERED_PARA_RE = re.compile(r'^\s*\(\d+\)')
_CITATION_RE      = re.compile(r'^\s*\(.*[Tt]rích')
_TIME_RE          = re.compile(r'^\s*Thời gian', re.UNICODE)

# Labels có thể merge nhiều dòng cùng block_id thành 1 paragraph
_MERGEABLE_LABELS = {"plain text", "plaintext", "paragraph",
                     "list", "footnote", "formula", "text"}


def _normalize_joined_text(text: str) -> str:
    """
    Chuẩn hóa text sau khi ghép mảnh OCR.

    Xử lý toàn diện các lỗi dấu ngoặc kép phát sinh khi merge nhiều dòng:

    Lỗi 1 — Lặp dấu mở: "..." + "..." → ""..."  (2 dấu " liên tiếp)
      Fix: collapse "" → "

    Lỗi 2 — Mất space sau dấu đóng: ..."Một → ..." Một
      Fix: chèn space sau " khi theo sau là chữ cái

    Lỗi 3 — Dấu câu bị tách khỏi ngoặc đóng: !" → !",  ." → ."
      Fix: kéo dấu câu vào trong ngoặc đóng nếu phù hợp

    Lỗi 4 — Thiếu space trước dấu mở: chữ" → chữ "
      Fix: chèn space trước " khi ngay trước là chữ cái

    Lỗi 5 — Khoảng trắng thừa bên trong ngoặc: " text " → "text"
      Fix: trim space sát dấu ngoặc

    Thứ tự fix quan trọng: phải chuẩn hóa quote trước, rồi mới fix spacing.
    """
    if not text:
        return text

    text = unicodedata.normalize("NFC", text)

    # Bước 0: Chuẩn hóa tất cả loại ngoặc kép fancy → ASCII "
    text = (text
            .replace('\u201c', '"').replace('\u201d', '"')
            .replace('\u201e', '"').replace('\u201f', '"')
            .replace('\u00ab', '"').replace('\u00bb', '"')
            .replace("''", '"').replace('``', '"'))

    # Bước 1: Collapse 2+ dấu " liên tiếp → 1 dấu
    # Lỗi phát sinh khi: dòng A kết thúc bằng " đóng, dòng B bắt đầu bằng " mở
    # merge: "...quá!" + "Một người..." → "...quá!""Một người..."
    text = re.sub(r'"{2,}', '"', text)

    # Bước 2: Dấu câu bị lạc ra ngoài ngoặc đóng
    # Trường hợp: !" → !" (kéo ! vào trong), tương tự .?" ...
    # KHÔNG làm điều này tự động vì convention tiếng Việt đặt dấu câu NGOÀI ngoặc
    # → chỉ fix trường hợp có khoảng trắng thừa: ! " → !"
    text = re.sub(r'([!?\.,:;])\s+"', r'\1"', text)

    # Bước 3: Thêm space giữa dấu câu + ngoặc đóng và chữ kế tiếp
    # "quá!"Một → "quá!" Một   (phổ biến nhất)
    text = re.sub(r'([!?\.,:;])"([A-Za-z\u00C0-\u024F\u1E00-\u1EFF])', r'\1" \2', text)

    # Bước 4: Thêm space giữa ngoặc đóng đơn thuần và chữ kế tiếp
    # text"Chữ → text" Chữ  (khi không có dấu câu trước ngoặc)
    text = re.sub(
        r'([A-Za-z\u00C0-\u024F\u1E00-\u1EFF])"([A-Za-z\u00C0-\u024F\u1E00-\u1EFF])',
        r'\1" \2', text
    )

    # Bước 5: Thêm space trước ngoặc mở khi trước là chữ cái và trong ngoặc là chữ
    # nói"Điều → nói "Điều
    # Dùng lookbehind để không nhầm với ngoặc đóng (đã xử lý ở bước 3 và 4)
    text = re.sub(
        r'(?<=[A-Za-z\u00C0-\u024F\u1E00-\u1EFF])\s*"(?=[A-Za-z\u00C0-\u024F\u1E00-\u1EFF])',
        r' "', text
    )

    # Bước 6: Trim khoảng trắng thừa bên trong ngoặc
    # " text " → "text"  — chỉ trim space ngay sau " mở (trước chữ đầu tiên trong ngoặc)
    # và space ngay trước " đóng, KHÔNG xóa space ở ngoài ngoặc
    text = re.sub(r'"\s{2,}', '" ', text)   # nhiều space → 1 space sau "
    text = re.sub(r'\s{2,}"', ' "', text)   # nhiều space → 1 space trước "

    # Bước 6: Dọn khoảng trắng thừa tổng quát
    return " ".join(text.split()).strip()


def _fix_punct_spacing(text: str) -> str:
    """
    Chuẩn hóa khoảng trắng quanh dấu câu.

    Các rule:
    - Không có space TRƯỚC dấu câu: "câu , từ" → "câu, từ"
    - Có space SAU dấu câu nếu theo sau là chữ cái: "câu.từ" → "câu. từ"
    - Không áp dụng cho số thập phân: "3.14" giữ nguyên
    - Không áp dụng cho dấu chấm cuối câu (không theo sau chữ nào)

    Gọi sau _normalize_joined_text để dọn dẹp lần cuối.
    """
    # Xóa space trước dấu câu
    text = re.sub(r'\s+([,;:!?])', r'\1', text)
    text = re.sub(r'\s+\.',        '.',    text)

    # Thêm space sau dấu câu nếu không có, trước chữ cái
    # Ngoại lệ: không thêm nếu trước dấu chấm là số (số thập phân)
    text = re.sub(
        r'([,;:!?])([A-Za-z\u00C0-\u024F\u1E00-\u1EFF])',
        r'\1 \2',
        text
    )
    text = re.sub(
        r'([A-Za-z\u00C0-\u024F\u1E00-\u1EFF])\.([A-Za-z\u00C0-\u024F\u1E00-\u1EFF])',
        r'\1. \2',
        text
    )

    return text


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def build_docx_mem(ocr_results: list):
    """
    Tạo file Word từ kết quả OCR.

    Args:
        ocr_results: list[dict] — mỗi dict có key "content_with_labels"

    Returns:
        io.BytesIO — stream file .docx
    """
    doc   = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(13)

    for res in ocr_results:
        items = res.get("content_with_labels", [])
        if not items:
            continue
        paragraphs = _group_into_paragraphs(items)
        _write_all(doc, paragraphs)

    file_stream = io.BytesIO()
    doc.save(file_stream)
    file_stream.seek(0)
    return file_stream


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 1 — GOM ITEMS THÀNH PARAGRAPHS
# ══════════════════════════════════════════════════════════════════════════════

def _group_into_paragraphs(items):
    """
    Chuyển danh sách items OCR thành list paragraph để xuất Word.

    Pipeline:
      1. Sắp xếp items theo (y, x).
      2. Tách items có col_index >= 0 (YOLO đã xác nhận 2 cột) ra riêng.
      3. Xử lý normal_items: gom → phát hiện 2 cột fallback → merge block_id.
      4. Gộp các two_col_line liên tiếp thành two_col_block.
      5. Chèn yolo_two_col_block vào đúng vị trí y.
    """
    items = sorted(items, key=lambda it: (it.get("y", 0), it.get("x", 0)))

    two_col_items = [it for it in items if it.get("col_index", -1) >= 0]
    normal_items  = [it for it in items if it.get("col_index", -1) <  0]

    yolo_two_col_block = None
    if two_col_items:
        col0 = sorted([it for it in two_col_items if it.get("col_index") == 0],
                      key=lambda it: it.get("y", 0))
        col1 = sorted([it for it in two_col_items if it.get("col_index") == 1],
                      key=lambda it: it.get("y", 0))

        left_lines  = _group_col_into_lines(col0)
        right_lines = _group_col_into_lines(col1)

        yolo_two_col_block = {
            "type":        "two_col_block",
            "anchor_cy":   min(it.get("y", 0) for it in two_col_items),
            "left_lines":  left_lines,
            "right_lines": right_lines,
            "label":       two_col_items[0].get("label", "header"),
            "page_w":      two_col_items[0].get("page_w", 1),
        }

    raw_lines = []
    for it in normal_items:
        h  = it.get("h", 20)
        cy = it.get("y", 0) + h / 2.0
        dy = max(8, h * 0.30)
        placed = False
        for ln in raw_lines:
            if abs(cy - ln["anchor_cy"]) < dy:
                ln["items"].append(it)
                placed = True
                break
        if not placed:
            raw_lines.append({"anchor_cy": cy, "items": [it]})

    raw_lines.sort(key=lambda ln: ln["anchor_cy"])

    line_dicts = []
    for ln in raw_lines:
        ln["items"].sort(key=lambda it: it.get("x", 0))
        first = ln["items"][0]
        pw    = first.get("page_w") or max(
            it.get("x", 0) + it.get("w", 0) for it in ln["items"])

        is_two_col = False
        if len(ln["items"]) == 2 and pw > 0:
            col_indices = [it.get("col_index", -1) for it in ln["items"]]
            if col_indices[0] == 0 and col_indices[1] == 1:
                is_two_col = True
            elif ln["items"][1].get("x", 0) / pw > _TWO_COL_RIGHT_THRESHOLD:
                is_two_col = True

        # FIX: áp _normalize_joined_text + _fix_punct_spacing khi ghép text
        joined_text = _fix_punct_spacing(
            _normalize_joined_text(
                " ".join(it["text"] for it in ln["items"])
            )
        ) if not is_two_col else ""

        line_dicts.append({
            "type":      "two_col_line" if is_two_col else "normal",
            "anchor_cy": ln["anchor_cy"],
            "first":     first,
            "page_w":    pw,
            "x_start":   first.get("block_x", first.get("x", 0)),
            "label":     first.get("label", "plain text").lower(),
            "block_id":  first.get("block_id", -1),
            "col_left":  ln["items"][0]["text"] if is_two_col else "",
            "col_right": ln["items"][1]["text"] if is_two_col else "",
            "text":      joined_text,
        })

    # Merge dòng cùng block_id
    merged_line_dicts = []
    i = 0
    while i < len(line_dicts):
        cur = line_dicts[i]
        if (cur["type"] == "two_col_line"
                or cur["block_id"] < 0
                or cur["label"] not in _MERGEABLE_LABELS
                or _CITATION_RE.match(cur["text"])):
            merged_line_dicts.append(cur)
            i += 1
            continue

        merged_texts = [cur["text"]]
        j = i + 1
        while j < len(line_dicts):
            nxt = line_dicts[j]
            if (nxt["block_id"] == cur["block_id"]
                    and nxt["type"] == "normal"
                    and nxt["label"] == cur["label"]):
                merged_texts.append(nxt["text"])
                j += 1
            else:
                break

        merged = dict(line_dicts[i])
        # FIX: normalize lại sau khi merge nhiều dòng (quote có thể lặp ở ranh giới)
        merged["text"] = _fix_punct_spacing(
            _normalize_joined_text(" ".join(merged_texts))
        )
        merged_line_dicts.append(merged)
        i = j

    # Gộp các two_col_line liên tiếp thành two_col_block
    paragraphs = _merge_two_col_lines(merged_line_dicts)

    # Chèn yolo_two_col_block vào đúng vị trí y
    if yolo_two_col_block:
        tcb_y = yolo_two_col_block["anchor_cy"]
        insert_idx = 0
        for k, p in enumerate(paragraphs):
            if p.get("anchor_cy", 0) < tcb_y:
                insert_idx = k + 1
        paragraphs.insert(insert_idx, yolo_two_col_block)

    return paragraphs


def _merge_two_col_lines(line_dicts):
    """
    Gộp các two_col_line liên tiếp thành 1 two_col_block duy nhất.

    Ví dụ header đề thi có 3 dòng 2 cột:
        "SỞ GD & ĐT..."  |  "ĐỀ THI THỬ..."
        "TRƯỜNG THPT"    |  "NĂM 2020-2021"
        "CHUYÊN HVT"     |  "MÔN: NGỮ VĂN"
    → 3 two_col_line → 1 two_col_block với left_lines=[3 dòng], right_lines=[3 dòng]
    """
    result = []
    i = 0
    while i < len(line_dicts):
        cur = line_dicts[i]

        if cur["type"] != "two_col_line":
            result.append(cur)
            i += 1
            continue

        left_lines  = []
        right_lines = []
        block_start = cur

        while i < len(line_dicts) and line_dicts[i]["type"] == "two_col_line":
            left_lines.append(line_dicts[i]["col_left"])
            right_lines.append(line_dicts[i]["col_right"])
            i += 1

        result.append({
            "type":        "two_col_block",
            "anchor_cy":   block_start["anchor_cy"],
            "left_lines":  left_lines,
            "right_lines": right_lines,
            "label":       block_start.get("label", "header"),
            "page_w":      block_start.get("page_w", 1),
        })

    return result


def _group_col_into_lines(col_items):
    """
    Gom items trong 1 cột thành danh sách chuỗi text (1 phần tử = 1 dòng vật lý).
    FIX: áp _normalize_joined_text + _fix_punct_spacing khi ghép text trong cột.
    """
    lines = []
    for it in col_items:
        h  = it.get("h", 20)
        cy = it.get("y", 0) + h / 2.0
        dy = max(8, h * 0.40)
        placed = False
        for ln in lines:
            if abs(cy - ln["anchor_cy"]) < dy:
                ln["items"].append(it)
                placed = True
                break
        if not placed:
            lines.append({"anchor_cy": cy, "items": [it]})

    result = []
    for ln in sorted(lines, key=lambda l: l["anchor_cy"]):
        ln["items"].sort(key=lambda it: it.get("x", 0))
        joined = _normalize_joined_text(" ".join(it["text"] for it in ln["items"]))
        joined = _fix_punct_spacing(joined)
        result.append(joined)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 2 — XUẤT RA WORD
# ══════════════════════════════════════════════════════════════════════════════

def _write_all(doc, paragraphs):
    """Xuất toàn bộ paragraphs ra doc."""
    for para in paragraphs:
        ptype = para.get("type", "normal")
        if ptype == "two_col_block":
            _write_two_col_block(doc, para)
        else:
            _write_paragraph(doc, para)


def _write_two_col_block(doc, block):
    """
    Xuất vùng 2 cột dưới dạng BẢNG ẨN VIỀN 2 Ô.

    Mỗi ô chứa toàn bộ các dòng của cột đó, căn giữa.
    In đậm nếu label là header/title.
    Viền hoàn toàn ẩn — chỉ dùng để căn layout.
    """
    left_lines  = block.get("left_lines",  [])
    right_lines = block.get("right_lines", [])
    label       = block.get("label", "header")
    is_bold     = label in ("title", "header")

    col_w = _A4_PAGE_WIDTH_TWIPS // 2

    tbl = doc.add_table(rows=1, cols=2)
    tbl.style = "Table Grid"

    _set_tbl_width(tbl, _A4_PAGE_WIDTH_TWIPS)
    _hide_tbl_borders(tbl)

    cells = tbl.rows[0].cells
    _fill_two_col_cell(cells[0], left_lines,  col_w, is_bold)
    _fill_two_col_cell(cells[1], right_lines, col_w, is_bold)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def _get_or_add_tblPr(tbl_el):
    tbl_pr = tbl_el.find(qn("w:tblPr"))
    if tbl_pr is None:
        tbl_pr = OxmlElement("w:tblPr")
        tbl_el.insert(0, tbl_pr)
    return tbl_pr


def _set_tbl_width(tbl, width_twips):
    tbl_pr = _get_or_add_tblPr(tbl._tbl)
    tbl_w  = OxmlElement("w:tblW")
    tbl_w.set(qn("w:w"), str(width_twips))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_pr.append(tbl_w)


def _hide_tbl_borders(tbl):
    tbl_pr      = _get_or_add_tblPr(tbl._tbl)
    tbl_borders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"), "none")
        el.set(qn("w:sz"), "0")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "auto")
        tbl_borders.append(el)
    tbl_pr.append(tbl_borders)


def _make_tc_borders_hidden():
    b = OxmlElement("w:tcBorders")
    for side in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"), "none")
        el.set(qn("w:sz"), "0")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "auto")
        b.append(el)
    return b


def _get_or_add_tcPr(tc_el):
    tc_pr = tc_el.find(qn("w:tcPr"))
    if tc_pr is None:
        tc_pr = OxmlElement("w:tcPr")
        tc_el.insert(0, tc_pr)
    return tc_pr


def _fill_two_col_cell(cell, lines, col_w_twips, bold):
    """Điền nội dung vào 1 ô của bảng 2 cột."""
    tc_pr = _get_or_add_tcPr(cell._tc)

    tc_w = OxmlElement("w:tcW")
    tc_w.set(qn("w:w"), str(col_w_twips))
    tc_w.set(qn("w:type"), "dxa")
    tc_pr.append(tc_w)

    tc_pr.append(_make_tc_borders_hidden())

    for p_el in list(cell._tc.findall(qn("w:p"))):
        cell._tc.remove(p_el)

    for line_text in lines:
        p   = cell.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after  = Pt(1)
        p.paragraph_format.space_before = Pt(1)
        run = p.add_run(line_text)
        run.bold      = bold
        run.font.name = "Times New Roman"
        run.font.size = Pt(13)

    if not lines:
        cell.add_paragraph()


def _write_paragraph(doc, line):
    """
    Xuất 1 paragraph thường.

    Thứ tự xử lý:
      1. Trích dẫn  → căn phải, in nghiêng
      2. Thời gian  → căn phải
      3. Title/Header → căn giữa (nếu x_start > 25% trang), in đậm
      4. Table      → dấu phân cách + căn giữa
      5. Caption    → căn giữa, in nghiêng, nhỏ hơn
      6. Footer     → căn giữa, in nghiêng, mờ
      7. Plain text → thụt đầu dòng theo ngữ nghĩa
    """
    label   = line.get("label", "plain text")
    page_w  = line.get("page_w", 1)
    x_start = line.get("x_start", 0)
    text    = line.get("text", "")

    # Lần normalize cuối trước khi xuất ra Word
    # (đề phòng có text không đi qua _group_into_paragraphs)
    text = _fix_punct_spacing(_normalize_joined_text(text))

    # ── 1. Trích dẫn ──────────────────────────────────────────────────────────
    if _CITATION_RE.match(text):
        p = doc.add_paragraph(text)
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p.paragraph_format.space_after  = Pt(6)
        p.paragraph_format.space_before = Pt(4)
        if p.runs:
            p.runs[0].italic = True
        return

    # ── 2. Thời gian ──────────────────────────────────────────────────────────
    if _TIME_RE.match(text):
        p = doc.add_paragraph(text)
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p.paragraph_format.space_after = Pt(4)
        return

    # ── 3. Title / Header ─────────────────────────────────────────────────────
    if label in ("title", "header"):
        p = doc.add_paragraph(text)
        p.paragraph_format.space_after  = Pt(6)
        p.paragraph_format.space_before = Pt(4)
        if page_w > 0 and x_start / page_w > 0.25:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if p.runs:
            p.runs[0].bold = True
        return

    # ── 4. Table ──────────────────────────────────────────────────────────────
    if label == "table":
        _add_separator(doc)
        p = doc.add_paragraph(text)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(4)
        if p.runs:
            p.runs[0].font.color.rgb = RGBColor(0x44, 0x44, 0x44)
        _add_separator(doc)
        return

    # ── 5. Caption ────────────────────────────────────────────────────────────
    if label == "caption":
        p = doc.add_paragraph(text)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(6)
        if p.runs:
            p.runs[0].italic    = True
            p.runs[0].font.size = Pt(11)
        return

    # ── 6. Footer ─────────────────────────────────────────────────────────────
    if label == "footer":
        p = doc.add_paragraph(text)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after  = Pt(2)
        p.paragraph_format.space_before = Pt(2)
        if p.runs:
            p.runs[0].italic         = True
            p.runs[0].font.size      = Pt(10)
            p.runs[0].font.color.rgb = RGBColor(0x88, 0x88, 0x88)
        return

    # ── 7. Plain text ─────────────────────────────────────────────────────────
    p = doc.add_paragraph(text)
    p.paragraph_format.space_after = Pt(4)

    if _QUESTION_RE.match(text):
        if len(text) > 60:
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    elif _NUMBERED_PARA_RE.match(text):
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.first_line_indent = Inches(_INDENT_PARAGRAPH)
    elif page_w > 0 and x_start / page_w > 0.08:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.first_line_indent = Inches(_INDENT_PLAIN)
    else:
        if len(text) > 60:
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


# ══════════════════════════════════════════════════════════════════════════════
# TIỆN ÍCH
# ══════════════════════════════════════════════════════════════════════════════

def _add_separator(doc):
    """Thêm dòng kẻ ngang mờ (dùng cho label table)."""
    p = doc.add_paragraph("─" * 40)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after  = Pt(2)
    p.paragraph_format.space_before = Pt(2)
    if p.runs:
        p.runs[0].font.color.rgb = RGBColor(0xBB, 0xBB, 0xBB)
        p.runs[0].font.size      = Pt(9)
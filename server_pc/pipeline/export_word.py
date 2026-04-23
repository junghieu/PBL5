import io
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

# Vùng in vật lý của tờ A4 trong Word (inch) — hằng số, không thay đổi
_A4_PRINT_WIDTH_INCHES = 6.5

# Lề trái thông thường của ảnh văn bản ≈ 4% chiều rộng ảnh
_LEFT_MARGIN_RATIO = 0.04

# Ngưỡng tối thiểu để áp dụng thụt đầu dòng (tránh thụt lắt nhắt do méo ảnh)
_MIN_INDENT_INCHES = 0.2


def build_docx_mem(ocr_results: list):
    """
    Tạo file Word từ kết quả OCR.

    Args:
        ocr_results: list[dict] — mỗi dict là kết quả 1 trang,
                     có key "content_with_labels" từ run_ocr_mem()

    Returns:
        io.BytesIO — stream file .docx sẵn sàng để trả về HTTP response
    """
    doc   = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(13)

    for res in ocr_results:
        items = res.get("content_with_labels", [])
        if not items:
            continue

        physical_lines = _group_into_physical_lines(items)

        for line in physical_lines:
            _write_paragraph(doc, line)

    file_stream = io.BytesIO()
    doc.save(file_stream)
    file_stream.seek(0)
    return file_stream


# ── NỘI BỘ ───────────────────────────────────────────────────────────────────

def _group_into_physical_lines(items):
    """
    Gom các item OCR cùng dòng vật lý thành 1 dòng duy nhất.

    Dùng anchor_cy cố định (cy của item đầu tiên trong mỗi dòng) làm mốc,
    tránh running average kéo threshold và merge nhầm dòng tiêu đề với nội dung.
    """
    items = sorted(items, key=lambda it: (it.get("y", 0), it.get("x", 0)))

    raw_lines = []
    for it in items:
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

    physical_lines = []
    for ln in raw_lines:
        ln["items"].sort(key=lambda it: it.get("x", 0))

        first  = ln["items"][0]
        page_w = first.get("page_w") or max(
            it.get("x", 0) + it.get("w", 0) for it in ln["items"]
        )

        text = " ".join(it["text"] for it in ln["items"]).strip()

        physical_lines.append({
            "text":    text,
            "label":   first.get("label", "plain text").lower(),
            "x_start": first.get("x", 0),
            "page_w":  page_w,
        })

    return physical_lines


def _write_paragraph(doc, line):
    """
    Thêm 1 đoạn văn vào doc với style phù hợp theo label.

    Label → style áp dụng:
      title / header  : căn giữa, in đậm, space_after lớn hơn
      table           : separator trước + sau, căn giữa, chữ xám
      caption         : căn giữa, in nghiêng, cỡ nhỏ hơn
      footer          : căn giữa, in nghiêng, cỡ nhỏ hơn, màu xám
      plain text /... : tính thụt đầu dòng từ tọa độ x
    """
    label   = line["label"]
    text    = line["text"]
    x_start = line["x_start"]
    page_w  = line["page_w"]

    if label in ("title", "header"):
        p = doc.add_paragraph(text)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after  = Pt(8)
        p.paragraph_format.space_before = Pt(4)
        if p.runs:
            p.runs[0].bold = True

    elif label == "table":
        _add_separator(doc)
        p = doc.add_paragraph(text)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(4)
        if p.runs:
            p.runs[0].font.color.rgb = RGBColor(0x44, 0x44, 0x44)
        _add_separator(doc)

    elif label == "caption":
        p = doc.add_paragraph(text)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(6)
        if p.runs:
            p.runs[0].italic    = True
            p.runs[0].font.size = Pt(11)

    elif label == "footer":
        p = doc.add_paragraph(text)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after  = Pt(2)
        p.paragraph_format.space_before = Pt(2)
        if p.runs:
            p.runs[0].italic          = True
            p.runs[0].font.size       = Pt(10)
            p.runs[0].font.color.rgb  = RGBColor(0x88, 0x88, 0x88)

    else:
        # plain text, paragraph, list, footnote, formula, ...
        p = doc.add_paragraph(text)
        p.paragraph_format.space_after = Pt(4)

        if page_w > 0:
            left_margin_px = _LEFT_MARGIN_RATIO * page_w
            indent_inches  = max(0, ((x_start - left_margin_px) / page_w) * _A4_PRINT_WIDTH_INCHES)
            if indent_inches > _MIN_INDENT_INCHES:
                p.paragraph_format.left_indent = Inches(indent_inches)


def _add_separator(doc):
    """Thêm dòng kẻ ngăn cách dạng text — dùng trước/sau vùng bảng."""
    p = doc.add_paragraph("─" * 40)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after  = Pt(2)
    p.paragraph_format.space_before = Pt(2)
    if p.runs:
        p.runs[0].font.color.rgb = RGBColor(0xBB, 0xBB, 0xBB)
        p.runs[0].font.size      = Pt(9)
from pathlib import Path
import io
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

def build_docx_mem(ocr_results: list):
    doc = Document()
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(13)

    for res in ocr_results:
        items = res.get("content_with_labels", [])
        if not items: continue

        # Sort toàn cục theo y rồi x (thay cho “last_line” bias)
        items.sort(key=lambda it: (it.get("y",0), it.get("x",0)))

        # Gom dòng: quét lần lượt, nếu lệch y < 0.25*h thì cùng dòng
        lines = []
        for it in items:
            cy = it.get("y",0) + it.get("h",20)/2.0
            h  = it.get("h",20)
            dy = max(6, h * 0.25)
            if lines and abs(cy - lines[-1]["cy"]) < dy:
                lines[-1]["items"].append(it)
                # cập nhật cy trung bình
                n = len(lines[-1]["items"])
                lines[-1]["cy"] = (lines[-1]["cy"]*(n-1) + cy)/n
            else:
                lines.append({"cy": cy, "items": [it]})

        # Sort dòng theo cy để không đảo a/b
        lines.sort(key=lambda ln: ln["cy"])

        for line in lines:
            line["items"].sort(key=lambda it: it.get("x",0))
            ...
            page_w = line["items"][0].get("page_w", 1400)
            gap_thr = max(0.04*page_w, 40)  # tỉ lệ thay vì hằng số
            text_acc = line["items"][0]["text"]
            for i in range(1, len(line["items"])):
                prev, cur = line["items"][i-1], line["items"][i]
                gap = cur.get("x", 0) - (prev.get("x", 0) + prev.get("w", 20))
                if gap > gap_thr:
                    text_acc += "\t" + cur["text"]
                else:
                    text_acc += " " + cur["text"]

            label = line["items"][0].get("label", "plain text").lower()
            x_start = line["items"][0].get("x", 0)

            p = doc.add_paragraph(text_acc)
            p.paragraph_format.space_after = Pt(4 if label == "plain text" else 8)

            if label in ["title", "header"]:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                if p.runs: p.runs[0].bold = True
            elif label == "table":
                p.insert_paragraph_before("--- [Bảng biểu] ---")
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                indent_inches = max(0, (x_start / page_w) * 6.5)
                if indent_inches > 0.3:
                    p.paragraph_format.left_indent = Inches(indent_inches)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf
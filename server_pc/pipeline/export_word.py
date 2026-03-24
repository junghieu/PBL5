from pathlib import Path
import io
import re
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

        # 1. Gom dòng vật lý an toàn (Giữ nguyên logic chuẩn của bạn)
        items.sort(key=lambda it: (it.get("y",0), it.get("x",0)))
        lines = []
        for it in items:
            cy = it.get("y",0) + it.get("h",20)/2.0
            h  = it.get("h",20)
            # Ngưỡng 25% an toàn
            dy = max(6, h * 0.25)
            if lines and abs(cy - lines[-1]["cy"]) < dy:
                lines[-1]["items"].append(it)
                n = len(lines[-1]["items"])
                lines[-1]["cy"] = (lines[-1]["cy"]*(n-1) + cy)/n
            else:
                lines.append({"cy": cy, "items": [it]})

        lines.sort(key=lambda ln: ln["cy"])

        # 2. Xử lý từng dòng vật lý
        physical_lines = []
        for line in lines:
            line["items"].sort(key=lambda it: it.get("x",0))
            page_w = line["items"][0].get("page_w", 1400)
            gap_thr = max(0.04*page_w, 40)
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
            physical_lines.append({
                "text": text_acc.strip(), 
                "label": label, 
                "x_start": x_start
            })

        # 3. Xuất Word từ các dòng
        for para in physical_lines:
            p = doc.add_paragraph(para["text"])
            label = para["label"]
            x_start = para["x_start"]

            p.paragraph_format.space_after = Pt(4 if label == "plain text" else 8)

            if label in ["title", "header"]:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                if p.runs: p.runs[0].bold = True
            elif label == "table":
                p.insert_paragraph_before("--- [Bảng biểu] ---")
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                indent_inches = max(0, ((x_start - 60) / 1400.0) * 6.5)
                if indent_inches > 0.3:
                    p.paragraph_format.left_indent = Inches(indent_inches)

    file_stream = io.BytesIO()
    doc.save(file_stream)
    file_stream.seek(0)
    return file_stream
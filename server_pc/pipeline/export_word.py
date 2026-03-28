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

        # 1. Gom dòng vật lý an toàn
        items.sort(key=lambda it: (it.get("y",0), it.get("x",0)))
        lines = []
        for it in items:
            h = it.get("h", 20) # Giữ 20 làm giá trị dự phòng nếu dict rỗng
            cy = it.get("y",0) + h / 2.0
            
            # SỬA LỖI 1: Bỏ số 6 fix cứng, dùng 30% chiều cao của chính box text đó
            dy = h * 0.30 
            
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
            
            # Lấy page_w từ OCR truyền sang. Nếu rủi ro bị mất key,
            # tự động lấy điểm kết thúc của chữ nằm xa nhất bên phải làm chiều rộng dự phòng.
            page_w = line["items"][0].get("page_w")
            if not page_w:
                page_w = max(it.get("x", 0) + it.get("w", 0) for it in line["items"])
            
            gap_thr = 0.04 * page_w
            text_acc = line["items"][0]["text"]
            for i in range(1, len(line["items"])):
                prev, cur = line["items"][i-1], line["items"][i]
                
                prev_w = prev.get("w", page_w * 0.01) 
                
                gap = cur.get("x", 0) - (prev.get("x", 0) + prev_w)
                if gap > gap_thr:
                    text_acc += "\t" + cur["text"]
                else:
                    text_acc += " " + cur["text"]
            
            label = line["items"][0].get("label", "plain text").lower()
            x_start = line["items"][0].get("x", 0)
            
            physical_lines.append({
                "text": text_acc.strip(), 
                "label": label, 
                "x_start": x_start,
                "page_w": page_w 
            })

        # 3. Xuất Word từ các dòng
        for para in physical_lines:
            p = doc.add_paragraph(para["text"])
            label = para["label"]
            x_start = para["x_start"]
            page_w = para["page_w"]

            p.paragraph_format.space_after = Pt(4 if label == "plain text" else 8)

            if label in ["title", "header"]:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                if p.runs: p.runs[0].bold = True
            elif label == "table":
                p.insert_paragraph_before("--- [Bảng biểu] ---")
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                # SỬA LỖI 3: Loại bỏ magic number 60. 
                # Lề trái thông thường của ảnh văn bản chiếm khoảng 4% chiều rộng
                left_margin_px = 0.04 * page_w 
                
                # 6.5 là số Inch vùng in vật lý của tờ A4 (hằng số Word, giữ nguyên)
                indent_inches = max(0, ((x_start - left_margin_px) / page_w) * 6.5)
                
                # Thụt đầu dòng tối thiểu 0.2 inch mới tác dụng (tránh thụt lắt nhắt do méo ảnh)
                if indent_inches > 0.2:
                    p.paragraph_format.left_indent = Inches(indent_inches)

    file_stream = io.BytesIO()
    doc.save(file_stream)
    file_stream.seek(0)
    return file_stream
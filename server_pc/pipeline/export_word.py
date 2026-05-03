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
            
            # Dùng 30% chiều cao của chính box text đó
            dy = max(8, h * 0.30)  # Tối thiểu 8px dù box nhỏ đến đâu
            
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
            
            text_acc = line["items"][0]["text"]
            for i in range(1, len(line["items"])):
                text_acc += " " + line["items"][i]["text"]
            
            text_acc = text_acc.strip()
            
            label = line["items"][0].get("label", "plain text").lower()
            x_start = line["items"][0].get("x", 0)
            
            physical_lines.append({
                "text": text_acc, 
                "label": label, 
                "x_start": x_start,
                "page_w": page_w 
            })

        # 3. Gom đoạn (Heuristic Level 1)
        logical_paragraphs = []
        current_para = None

        for line in physical_lines:
            text = line.get("text", "").strip()
            if not text:
                continue
            
            if current_para is None:
                current_para = {**line, "text": text}
            else:
                prev_text = current_para["text"].strip()
                curr_label = line.get("label", "plain text").lower()
                prev_label = current_para.get("label", "plain text").lower()

                ends_with_punct = prev_text.endswith(('.', ':', '?', '!', ';'))
                starts_lower = text[0].islower()
                
                # Cập nhật Regex để bắt được cả dấu gạch ngang dài (–, —) thay vì chỉ trừ (-)
                is_list_item = bool(re.match(r'^([\-\–\—]|\+|\d+\.\d+|\d+\.)', text))
                is_heading = text.startswith("CHƯƠNG") or text.isupper()

                # Cắt đoạn mới nếu:
                # 1. Label khác plain text (chuyển sang/đi từ title, table, etc.)
                # 2. Dòng hiện tại là list item hoặc heading
                # 3. Dòng trước kết thúc bằng dấu chấm VÀ dòng này bắt đầu bằng chữ hoa
                if curr_label != "plain text" or prev_label != "plain text":
                    flush = True
                elif is_list_item or is_heading:
                    flush = True
                elif ends_with_punct and not starts_lower:
                    flush = True
                else:
                    flush = False

                if flush:
                    logical_paragraphs.append(current_para)
                    current_para = {**line, "text": text}
                else:
                    current_para["text"] += " " + text

        if current_para:
            logical_paragraphs.append(current_para)

        # 4. Xuất Word từ các đoạn văn
        for para in logical_paragraphs:
            p = doc.add_paragraph(para["text"])
            label = para.get("label", "plain text")
            x_start = para.get("x_start", 0)
            page_w = para.get("page_w", 1)

            p.paragraph_format.space_after = Pt(4 if label == "plain text" else 8)

            # Tính tỷ lệ khoảng lùi của lề so với chiều rộng trang
            left_margin_px = 0.04 * page_w 
            ratio = (x_start - left_margin_px) / page_w

            if label in ["title", "header"]:
                if p.runs: p.runs[0].bold = True
                
                # Chỉ căn giữa nếu nó thực sự nằm sâu vào trong giữa trang giấy (> 20%)
                if ratio > 0.2:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                else:
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    # Có thể thiết lập mức thụt nhẹ nếu nó không nằm sát lề trái
                    if ratio > 0.03:
                        p.paragraph_format.left_indent = Inches(0.2)

            elif label == "table":
                p.insert_paragraph_before("--- [Bảng biểu] ---")
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                
                # Xác định xem đoạn này có phải list item không
                clean_text = para["text"].strip()
                is_list_item = bool(re.match(r'^([\-\–\—]|\+|\d+\.\d+|\d+\.)', clean_text))
                
                if is_list_item:
                    # Xử lý các dòng dạng danh sách (-, +, 1.1)
                    if clean_text.startswith("+"):
                        p.paragraph_format.left_indent = Inches(0.5)  # Thụt vô sâu hơn chút
                        p.paragraph_format.first_line_indent = Inches(-0.15) # Treo
                    else:
                        p.paragraph_format.left_indent = Inches(0.25) # Thụt cỡ 2-3 chữ
                        p.paragraph_format.first_line_indent = Inches(-0.15) # Treo
                else:
                    # Đoạn văn bản bình thường: chỉ thụt dòng đầu tiên nếu tọa độ x_start lùi vào trong
                    if ratio > 0.03: 
                        p.paragraph_format.first_line_indent = Inches(0.3) # Cố định thụt dòng đầu (~3 chữ)

    file_stream = io.BytesIO()
    doc.save(file_stream)
    file_stream.seek(0)
    return file_stream
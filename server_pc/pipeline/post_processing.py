import re

try:
    from underthesea import text_normalize
except ImportError:
    text_normalize = lambda x: x

def remove_repetitions(text):
    # Xóa lặp từ quá 2 lần (vd: tranh tranh tranh -> tranh)
    pattern = r'(\b\w+\b)(?:\s+\1){2,}'
    return re.sub(pattern, r'\1', text, flags=re.IGNORECASE)

def fix_question_keyword(text):
    # Sửa lỗi Câu dạng tiếng Việt
    variants = ["chu", "cậu", "cầu", "chú", "cán"]
    variants_str = "|".join(variants)
    pattern = rf'^({variants_str})\s+(\d+)'
    
    def replace_func(match):
        return f"Câu {match.group(2)}"
        
    text = re.sub(pattern, replace_func, text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Bước 3: Sửa lỗi nhận diện số "19xx" thành "Question xx" trong đề tiếng Anh
    text = re.sub(r'^19(\d+)\b', r'Question \1', text, flags=re.MULTILINE)
    
    return text

def fix_spacing_and_punctuation(text):
    # Bước 1: Xóa dấu cách dư thừa sau dấu mở ngoặc và trước dấu đóng ngoặc
    text = re.sub(r'\(\s+([^\)]+?)\s+\)', r'(\1)', text)
    # Phòng hờ trường hợp chỉ có 1 bên khoảng trắng
    text = re.sub(r'\(\s+', r'(', text)
    text = re.sub(r'\s+\)', r')', text)
    
    # Bước 1: Xóa dấu cách dư thừa quanh dấu gạch ngang (ngày tháng/mã đề)
    text = re.sub(r'(\d)\s*-\s*(\d)', r'\1-\2', text)
    
    # Bước 1 & Bước 4: Sửa lỗi dính phương án (1B ) -> B.) và bọc phương án trắc nghiệm ([A-D] ) -> [A-D].)
    # \d* cho phép khớp "1B" thành "B"
    text = re.sub(r'\b\d*([A-D])\s*\)', r'\1.', text)
    
    return text

def remove_tail_hallucinations(text):
    # Bước 2: Xóa Tail-Hallucinations (Bộ lọc đuôi câu)
    # Vì ở bước post-processing text thô ta không có tham số Confidence < 0.3
    # Nên ta dùng Regex để gọt các từ rác có dấu hiệu là ảo giác (ký tự lặp, từ cụt lặp lại)
    # Cắt các chuỗi ký tự đơn lẻ vô nghĩa ở cuối câu (ví dụ: a c b d)
    text = re.sub(r'(?:\s+\b\w\b){3,}$', '', text)
    # Xóa khoảng trắng thừa ở cuối
    return text.strip()

def process_ocr_text(text):
    if not isinstance(text, str) or not text.strip():
        return text
        
    text = text.strip()
    
    # Bước 1 & 4 (Space, Multiple Choice, Hyphen)
    text = fix_spacing_and_punctuation(text)
    
    # Bước 3 (Fix Question vs 19xx, Cậu -> Câu)
    text = fix_question_keyword(text)
    
    # Bước 2 (Loop / Hallucination đuôi)
    text = remove_repetitions(text)
    text = remove_tail_hallucinations(text)
    
    # Normalize dấu tiếng Việt
    text = text_normalize(text)
    
    return text

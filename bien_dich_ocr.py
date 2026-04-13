from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

def thiet_lap_bo_nhan_dang(duong_dan_cau_hinh, duong_dan_trong_so):
    # Nạp cấu hình từ tệp yml (đảm bảo file yml đã đổi sang seq2seq)
    cau_hinh_he_thong = Cfg.load_config_from_file(duong_dan_cau_hinh)
    
    # Gán đường dẫn trọng số .pth mới (85MB)
    cau_hinh_he_thong['weights'] = duong_dan_trong_so
    
    # Ép chạy trên CPU để tương thích hoàn toàn với Raspberry Pi
    cau_hinh_he_thong['device'] = 'cpu'
    
    return Predictor(cau_hinh_he_thong)

def thuc_hien_nhan_dien(anh_dau_vao, may_ocr, nguong_tin_cay=0.5):
    """
    anh_dau_vao: Ảnh đã được cắt dòng (PIL Image)
    may_ocr: Đối tượng Predictor đã khởi tạo
    nguong_tin_cay: Mức độ tin tưởng tối thiểu để lấy kết quả (mặc định 50%)
    """
    # return_prob=True giúp lấy thêm điểm số tin cậy của AI
    ket_qua_chu_viet, diem_so = may_ocr.predict(anh_dau_vao, return_prob=True)
    
    # Nếu điểm số quá thấp, có thể đây là vùng nhiễu, trả về chuỗi rỗng
    if diem_so < nguong_tin_cay:
        return ""
        
    return ket_qua_chu_viet
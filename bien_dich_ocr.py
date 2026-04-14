from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

def thiet_lap_bo_nhan_dang(duong_dan_cau_hinh, duong_dan_trong_so):
    # QUAN TRỌNG: Nạp cấu hình trực tiếp từ file .yml cục bộ của nhóm
    # Điều này đảm bảo bảng mã (vocab) khớp hoàn toàn với file .pth 85MB
    cau_hinh_he_thong = Cfg.load_config_from_file(duong_dan_cau_hinh)
    
    # Chỉ định file trọng số mới đã được train lại với vocab chuẩn
    cau_hinh_he_thong['weights'] = duong_dan_trong_so
    
    # Ép chạy trên CPU để ổn định trên Raspberry Pi
    cau_hinh_he_thong['device'] = 'cpu'
    
    return Predictor(cau_hinh_he_thong)

def thuc_hien_nhan_dien(anh_dau_vao, may_ocr, nguong_tin_cay=0.5):
    """
    anh_dau_vao: Ảnh dòng chữ (PIL Image)
    may_ocr: Đối tượng Predictor
    nguong_tin_cay: Ngưỡng lọc nhiễu (mặc định 0.5)
    """
    # Lấy thêm điểm tin cậy (prob) để lọc bỏ các dòng "ma"
    ket_qua_chu_viet, diem_so = may_ocr.predict(anh_dau_vao, return_prob=True)
    
    # Nếu AI đoán với độ tự tin thấp (dưới ngưỡng), trả về rỗng
    # Cách này giúp loại bỏ các ký tự rác do nhiễu ảnh
    if diem_so < nguong_tin_cay:
        return ""
        
    return ket_qua_chu_viet
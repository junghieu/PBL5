from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

def thiet_lap_bo_nhan_dang(duong_dan_cau_hinh, duong_dan_trong_so):
    cau_hinh_he_thong = Cfg.load_config_from_file(duong_dan_cau_hinh)
    cau_hinh_he_thong['weights'] = duong_dan_trong_so
    cau_hinh_he_thong['device'] = 'cpu'
    return Predictor(cau_hinh_he_thong)

def thuc_hien_nhan_dien(anh_dau_vao, may_ocr):
    ket_qua_chu_viet = may_ocr.predict(anh_dau_vao)
    return ket_qua_chu_viet
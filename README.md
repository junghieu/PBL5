Hệ Thống Nhận Diện Chữ Viết Tiếng Việt - Nhóm 11
Kho lưu trữ này chứa mô hình AI nhận diện chữ viết (OCR) sử dụng kiến trúc VGG-Seq2Seq, được tối ưu hóa đặc biệt về tốc độ và độ chính xác để chạy trên Raspberry Pi.

1. Thành phần tệp tin
seq2seq_ocr_nhom11.pth: Trọng số mô hình đã huấn luyện (85MB). Sử dụng kiến trúc LSTM giúp nhận diện ổn định và nhẹ hơn bản Transformer cũ.

cau_hinh_ocr.yml: Chứa tham số cấu hình mạng Seq2Seq, bộ từ vựng (vocab) và thiết lập chạy trên CPU.

bien_dich_ocr.py: Module xử lý trung gian, đóng gói các hàm khởi tạo và nhận diện có tích hợp bộ lọc độ tin cậy.

requirements.txt: Danh sách thư viện cần thiết (VietOCR, PyTorch, OpenCV...).

2. Cài đặt môi trường
Mở Terminal trên Raspberry Pi và thực hiện các lệnh sau:

Bash
# Cài đặt trình quản lý tệp lớn
sudo apt-get install git-lfs
git lfs install

# Tải mã nguồn và trọng số
git pull
git lfs pull

# Cài đặt thư viện Python
pip install -r requirements.txt
3. Hướng dẫn tích hợp
Đoạn mã mẫu để tích hợp vào luồng xử lý chính của Camera:

Python
from bien_dich_ocr import thiet_lap_bo_nhan_dang, thuc_hien_nhan_dien
from PIL import Image

# 1. Khởi tạo mô hình
may_ocr = thiet_lap_bo_nhan_dang('cau_hinh_ocr.yml', 'seq2seq_ocr_nhom11.pth')

# 2. Đọc ảnh (Đã qua xử lý cắt dòng từ OpenCV/YOLO)
anh_dong_chu = Image.open('line_image.jpg')

# 3. Nhận diện với ngưỡng tin cậy 0.5 (Tránh nhiễu)
van_ban = thuc_hien_nhan_dien(anh_dong_chu, may_ocr, nguong_tin_cay=0.5)

print(f"Kết quả: {van_ban}")
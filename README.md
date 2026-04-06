# Hệ Thống Nhận Diện Chữ Viết Tiếng Việt - Nhóm 11

Kho lưu trữ này chứa mô hình AI nhận diện chữ viết (OCR) và mã nguồn tích hợp, được tối ưu hóa để chạy trên môi trường phần cứng nhúng (Raspberry Pi).

## 1. Thành phần tệp tin

* `transformerocr.pth`: Trọng số mô hình đã được huấn luyện (Quản lý qua Git LFS).
* `cau_hinh_ocr.yml`: Tập tin chứa cấu hình mạng nơ-ron và bộ từ vựng tiếng Việt.
* `bien_dich_ocr.py`: Mã nguồn chứa các hàm khởi tạo và chạy mô hình.
* `requirements.txt`: Danh sách các thư viện cần thiết.

## 2. Cài đặt môi trường

Thực hiện các lệnh sau trên Terminal của thiết bị để tải mã nguồn và cài đặt các thư viện:

```bash
sudo apt-get install git-lfs
git lfs install
git pull
pip install -r requirements.txt
import cv2
import imutils
import numpy as np

# 1) Đọc ảnh
img = cv2.imread("demo_images/notecard.png")
if img is None:
    raise FileNotFoundError("Không tìm thấy file demo_images/notecard.png")

# 2) Resize giữ tỉ lệ
img = imutils.resize(img, width=1200)

# 3) Chuyển xám + blur nhẹ
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# 4) Tự động tìm biên (theo README)
edged = imutils.auto_canny(gray)

# 5) Morphology để nối nét chữ/khung
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=1)

# 6) Lưu kết quả trung gian
cv2.imwrite("01_resized.jpg", img)
cv2.imwrite("02_gray.jpg", gray)
cv2.imwrite("03_edges.jpg", edged)

print("Done! Đã lưu: 01_resized.jpg, 02_gray.jpg, 03_edges.jpg")
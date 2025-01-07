import cv2
import numpy as np

# Tạo hai ảnh nhị phân (grayscale)
img1 = np.zeros((200, 200), dtype=np.uint8)
img2 = np.zeros((200, 200), dtype=np.uint8)

# Vẽ hình chữ nhật và hình tròn
cv2.rectangle(img1, (50, 50), (150, 150), 255, -1)  # Hình chữ nhật trắng
cv2.circle(img2, (100, 100), 50, 255, -1)          # Hình tròn trắng

# Thực hiện phép toán AND
result = cv2.bitwise_and(img1, img2)

# Hiển thị kết quả
cv2.imshow("Image 1", img1)
cv2.imshow("Image 2", img2)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

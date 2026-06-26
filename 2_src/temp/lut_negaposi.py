import cv2
import numpy as np

cv2.namedWindow('src')
cv2.namedWindow('dst')

file_src = 'src.png'
file_dst = 'dst.png'

img_src = cv2.imread(file_src, 0)

# LUTを作る
lut = np.zeros(256, dtype=np.uint8)

for i in range(256):
    lut[i] = 255 - i

# LUTを画像に適用
img_dst = cv2.LUT(img_src, lut)

cv2.imshow('src', img_src)
cv2.imshow('dst', img_dst)

cv2.imwrite(file_dst, img_dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
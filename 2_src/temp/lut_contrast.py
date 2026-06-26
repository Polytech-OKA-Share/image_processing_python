import cv2
import numpy as np

cv2.namedWindow('src')
cv2.namedWindow('dst')

file_src = 'src.png'
file_dst = 'dst.png'

img_src = cv2.imread(file_src, 1)
img_src = cv2.cvtColor(img_src,cv2. COLOR_BGR2GRAY)

# ★ コントラスト変換（区間引き伸ばし）★
MIN = 100
MAX = 150

# LUT（256要素の変換テーブル）を作る
# lut[入力値] = 出力値 になる配列
lut = np.zeros(256, dtype=np.uint8)

for i in range(256):
    if i <= MIN:
        lut[i] = 0                              # MIN以下は黒に潰す
    elif i >= MAX:
        lut[i] = 255                            # MAX以上は白に飛ばす
    else:
        lut[i] = (i - MIN) * 255 // (MAX - MIN) # 線形に引き伸ばす

# LUTを画像に適用（全ピクセルの輝度値を変換テーブルで一括変換）
img_dst = cv2.LUT(img_src, lut)

cv2.imshow('src', img_src)
cv2.imshow('dst', img_dst)

cv2.imwrite(file_dst, img_dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
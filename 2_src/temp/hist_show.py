import cv2
import numpy as np
from picamera2 import Picamera2

cv2.namedWindow('src')
cv2.namedWindow('dst')
cv2.namedWindow('histogram')  # ヒストグラム表示用ウィンドウを追加


def draw_histogram(gray_img, height=200, width=256):
    """
    グレー画像からヒストグラム画像を生成して返す関数

    ・横軸 = 輝度値（0〜255）、縦軸 = ピクセル数
    ・最大値を画面の高さに合わせて正規化する
    """
    # ヒストグラムを計算（256段階、輝度値ごとのピクセル数）
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

    # 最大値で正規化（一番多い輝度値が height ピクセルの高さになるようにスケール）
    cv2.normalize(hist, hist, alpha=0, beta=height, norm_type=cv2.NORM_MINMAX)

    # 描画先（黒いキャンバス）を用意
    canvas = np.zeros((height, width), dtype=np.uint8)

    # 輝度値ごとに縦棒を1本ずつ描く
    for x in range(256):
        bar_height = int(hist[x, 0])
        # 下から積み上げる（y座標は上が0なので、下端 = height-1 から積む）
        cv2.line(canvas,
            (x, height - 1),           # 棒の下端
            (x, height - bar_height),  # 棒の上端
            255,                        # 白で描画
            1)

    return canvas


picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()

while True:
    img_src = picam2.capture_array()
    img_src = cv2.cvtColor(img_src, cv2.COLOR_RGB2BGR)
    img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    # ヒストグラム平坦化
    img_dst = cv2.equalizeHist(img_src)

    # ヒストグラムを画像として生成（src と dst の両方を重ねて比較）
    hist_src = draw_histogram(img_src)   # 元画像のヒストグラム（暗め）
    hist_dst = draw_histogram(img_dst)   # 平坦化後のヒストグラム（広がる）

    # 2つを横に並べて1枚の画像にする
    # ・左半分 = 元画像のヒストグラム（暗い線）
    # ・右半分 = 平坦化後のヒストグラム（明るい線）
    hist_combined = np.hstack([hist_src, hist_dst])

    # 区切り線（中央に白い縦線）
    cv2.line(hist_combined, (255, 0), (255, 199), 180, 1)

    # ラベルを書き込む
    cv2.putText(hist_combined, 'src', (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, 200, 1)
    cv2.putText(hist_combined, 'dst', (261, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, 200, 1)

    cv2.imshow('src', img_src)
    cv2.imshow('dst', img_dst)
    cv2.imshow('histogram', hist_combined)  # ヒストグラムを表示

    ch = cv2.waitKey(1)
    if ch == ord('q'):
        break

picam2.stop()
picam2.close()
cv2.destroyAllWindows()
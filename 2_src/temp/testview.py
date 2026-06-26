import cv2
import numpy as np
from picamera2 import Picamera2

cv2.namedWindow('src')
cv2.namedWindow('dst')

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()

# ============================================================
# ★ ループの前に射影変換行列を1回だけ計算しておく
#
# カメラ画像のサイズに合わせて src_pts を調整してください。
# 座標を調べるには、先にカメラ画像を1枚保存して
# click_points.py（前回のコード）でクリックするのが便利です。
# ============================================================

# カメラ画像のサイズ（Picamera2のデフォルトに合わせる）
img_w, img_h = 640, 480

# 元画像で「地面の四角形」の4隅（左上→右上→右下→左下）
# ★ ここを自分のカメラ映像に合わせて調整する ★
src_pts = np.float32([
    [img_w * 0.35, img_h * 0.5],   # 左上
    [img_w * 0.65, img_h * 0.5],   # 右上
    [img_w * 0.9,  img_h * 0.9],   # 右下
    [img_w * 0.1,  img_h * 0.9],   # 左下
])

# 出力画像のサイズ
out_w, out_h = 400, 600

dst_pts = np.float32([
    [0,     0    ],   # 左上
    [out_w, 0    ],   # 右上
    [out_w, out_h],   # 右下
    [0,     out_h],   # 左下
])

# 変換行列をここで1回だけ計算（重い処理なのでループ外に置く）
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

while True:
    img_src = picam2.capture_array()
    img_src = cv2.cvtColor(img_src, cv2.COLOR_RGB2BGR)

    # ★ 画像処理の内容を書く ★
    # 射影変換を適用（変換行列 M は計算済みなので一瞬で終わる）
    img_dst = cv2.warpPerspective(img_src, M, (out_w, out_h))

    # -------- src に4点の枠を重ねて表示（調整用・確認できたら消してOK）--------
    img_show = img_src.copy()
    cv2.polylines(img_show, [src_pts.astype(int)], isClosed=True,
                  color=(0, 255, 255), thickness=2)   # 黄色い枠
    for i, pt in enumerate(src_pts.astype(int)):
        cv2.circle(img_show, tuple(pt), 6, (0, 0, 255), -1)          # 赤丸
        cv2.putText(img_show, str(i), tuple(pt + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)   # 番号

    cv2.imshow('src', img_show)
    cv2.imshow('dst', img_dst)

    ch = cv2.waitKey(1)
    if ch == ord('q'):
        break

picam2.stop()
picam2.close()
cv2.destroyAllWindows()
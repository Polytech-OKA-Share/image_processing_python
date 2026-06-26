import cv2
import numpy as np

cv2.namedWindow('src')
cv2.namedWindow('dst')

file_src = 'src.png'
file_dst = 'dst.png'

img_src = cv2.imread(file_src, 1)

# ============================================================
# ★ 使いたいフィルタの行だけコメントアウトを外して使う ★
#
# 【各フィルタの特徴まとめ】
#
#  フィルタ名       縦エッジ  横エッジ  斜め  ノイズ耐性  特徴
#  ────────────────────────────────────────────────────────
#  Sobel X         ◎        ✕        △    ○          縦線（横方向の変化）を検出
#  Sobel Y         ✕        ◎        △    ○          横線（縦方向の変化）を検出
#  Sobel 合成       ◎        ◎        ○    ○          X・Y を合わせた総合エッジ
#  Laplacian       ◎        ◎        ◎    ✕          全方向・細いエッジ・ノイズに弱い
#  Canny           ◎        ◎        ◎    ◎          最もきれいな2値エッジ線
#  Prewitt         ◎        ◎        △    ○          Sobelの単純版
#  Scharr          ◎        ◎        △    ◎          Sobelより高精度な方向微分
# ============================================================

# エッジ検出はグレースケールで行うのが基本
img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

# ノイズを先に除去しておくとエッジがきれいになる（お好みで）
# img_gray = cv2.GaussianBlur(img_gray, ksize=(3, 3), sigmaX=1.0)


# ① Sobel フィルタ（X方向）
#    横方向の輝度変化を検出 → 縦のエッジが強調される
#    dx=1, dy=0 → X方向の1次微分
#    ksize: カーネルサイズ（1, 3, 5, 7）大きいほど広い範囲を参照
#img_dst = cv2.Sobel(img_gray, ddepth=cv2.CV_8U, dx=1, dy=0, ksize=3)

# ② Sobel フィルタ（Y方向）
#    縦方向の輝度変化を検出 → 横のエッジが強調される
# img_dst = cv2.Sobel(img_gray, ddepth=cv2.CV_8U, dx=0, dy=1, ksize=3)

# ③ Sobel フィルタ（X・Y合成）
#    X方向とY方向を合わせて全方向のエッジを検出する
#    合成式: sqrt(Gx^2 + Gy^2) ← 本来の勾配の大きさ
#    近似式: |Gx| + |Gy|       ← 計算が軽い
sobel_x = cv2.Sobel(img_gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
sobel_y = cv2.Sobel(img_gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
img_dst = cv2.convertScaleAbs(np.sqrt(sobel_x**2 + sobel_y**2))  # 本来の合成
# img_dst = cv2.addWeighted(                                         # 近似の合成
#     cv2.convertScaleAbs(sobel_x), 0.5,
#     cv2.convertScaleAbs(sobel_y), 0.5, 0)

# ④ Laplacian フィルタ
#    2次微分でエッジを検出。全方向に反応するが、ノイズにも敏感。
#    事前にガウシアンをかけると安定する（LoG: Laplacian of Gaussian）
#    ksize: カーネルサイズ（1 or 3 or 5...）
# img_dst = cv2.Laplacian(img_gray, ddepth=cv2.CV_8U, ksize=3)

# ⑤ Canny フィルタ
#    最もよく使われる高品質エッジ検出。出力は2値（白=エッジ、黒=背景）。
#    内部で ガウシアン → Sobel → 非最大抑制 → ヒステリシス閾値 を自動処理。
#    threshold1: 弱いエッジを捨てる下限（小さいほどエッジが増える）
#    threshold2: 確実なエッジと判断する上限（小さいほどエッジが増える）
#    ※ threshold2 : threshold1 = 2:1 ～ 3:1 が目安
# img_dst = cv2.Canny(img_gray, threshold1=50, threshold2=150)

# ⑥ Prewitt フィルタ（手動カーネル）
#    Sobel の単純版。重みが均一なので計算は軽いがノイズに少し弱い。
#    X・Y それぞれのカーネルを自分で定義して filter2D で適用する。
# kernel_prewitt_x = np.array([
#     [-1,  0,  1],
#     [-1,  0,  1],
#     [-1,  0,  1],
# ], dtype=np.float32)
# kernel_prewitt_y = np.array([
#     [-1, -1, -1],
#     [ 0,  0,  0],
#     [ 1,  1,  1],
# ], dtype=np.float32)
# prewitt_x = cv2.filter2D(img_gray, ddepth=cv2.CV_64F, kernel=kernel_prewitt_x)
# prewitt_y = cv2.filter2D(img_gray, ddepth=cv2.CV_64F, kernel=kernel_prewitt_y)
# img_dst = cv2.convertScaleAbs(np.sqrt(prewitt_x**2 + prewitt_y**2))

# ⑦ Scharr フィルタ
#    Sobel の改良版。3×3カーネルでより正確な方向微分が得られる。
#    ksize に -1 を指定すると Scharr カーネルが自動で使われる。
# scharr_x = cv2.Scharr(img_gray, ddepth=cv2.CV_64F, dx=1, dy=0)
# scharr_y = cv2.Scharr(img_gray, ddepth=cv2.CV_64F, dx=0, dy=1)
# img_dst = cv2.convertScaleAbs(np.sqrt(scharr_x**2 + scharr_y**2))


cv2.imshow('src', img_src)
cv2.imshow('dst', img_dst)

cv2.imwrite(file_dst, img_dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
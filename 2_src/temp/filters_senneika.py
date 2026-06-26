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
#  フィルタ名              強さ調整  自然さ  ノイズ耐性  特徴
#  ──────────────────────────────────────────────────────────
#  シャープ化カーネル      △        ○      △         最もシンプルな鮮鋭化
#  強いシャープ化          △        △      ✕         エッジを強く強調・過剰になりやすい
#  アンシャープマスク      ◎        ◎      ○         最もよく使われる・自然な仕上がり
#  ハイブーストフィルタ    ◎        ◎      ○         アンシャープの強度を係数で調整
#  Laplacianによる鮮鋭化   △        △      ✕         全方向のエッジを加算・シャープすぎ注意
#  エンボス               △        ✕      △         エッジを浮き彫りにする特殊効果
# ============================================================


# ① シャープ化カーネル（基本形）
#    中心に大きな正の重み、周囲に負の重みを置く。
#    「自分 × 5 - 上下左右の平均」= エッジ成分を加算する効果
#    カーネルの合計が1 → 画像全体の明るさは変わらない
kernel = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],  # 中心(5) - 隣4方向(各-1) = 合計1
    [ 0, -1,  0],
], dtype=np.float32)
img_dst = cv2.filter2D(img_src, ddepth=-1, kernel=kernel)

# ② 強いシャープ化カーネル
#    斜め方向にも -1 を置くことで8方向すべてからエッジを強調する。
#    効果が強いのでかけすぎると不自然になりやすい。
# kernel = np.array([
#     [-1, -1, -1],
#     [-1,  9, -1],  # 中心(9) - 周囲8方向(各-1) = 合計1
#     [-1, -1, -1],
# ], dtype=np.float32)
# img_dst = cv2.filter2D(img_src, ddepth=-1, kernel=kernel)

# ③ アンシャープマスク（Unsharp Masking）
#    考え方：ぼかした画像との差（＝エッジ成分）を元画像に足す
#    式：出力 = 元画像 + strength × (元画像 - ぼかし画像)
#
#    strength: 強度（大きいほどシャープ・1.0〜3.0が目安）
#    ksize   : ぼかしのカーネルサイズ（大きいほど広い範囲のエッジを強調）
#    sigma   : ガウシアンの広がり（大きいほど緩やかなぼかし）
# strength = 1.5
# blurred = cv2.GaussianBlur(img_src, ksize=(5, 5), sigmaX=1.0)
# img_dst = cv2.addWeighted(img_src, 1 + strength, blurred, -strength, 0)

# ④ ハイブーストフィルタ
#    アンシャープマスクの式を変形したもの。
#    式：出力 = A × 元画像 - ぼかし画像
#         A = 1 のとき → アンシャープマスクと同じ
#         A > 1 のとき → 元画像成分が強くなりより鮮明に
#    ※ A = 1 + strength と置くとアンシャープマスクと完全に一致する
# A = 2.0  # 1.0より大きくすると元画像の成分が強まる
# blurred = cv2.GaussianBlur(img_src, ksize=(5, 5), sigmaX=1.0)
# img_dst = cv2.addWeighted(img_src, A, blurred, -(A - 1), 0)

# ⑤ Laplacian による鮮鋭化
#    エッジ検出（Laplacian）の結果を元画像に加算する。
#    式：出力 = 元画像 - Laplacian（符号に注意）
#    Laplacianは「周囲との差の合計」なのでそのまま引くとエッジが強調される。
#    ノイズにも反応しやすいので事前にぼかしを入れると安定する（LoG）。
# img_gray  = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)  # グレーのみ対応
# laplacian = cv2.Laplacian(img_gray, ddepth=cv2.CV_64F, ksize=3)
# img_dst   = cv2.convertScaleAbs(img_gray - laplacian)

# ⑥ エンボス（浮き彫り効果）
#    斜め方向の差分を使い、レリーフのような立体感を出す特殊効果。
#    鮮鋭化というより「エッジを浮き上がらせる」演出用フィルタ。
#    128を足すことでゼロ付近（変化なし）がグレーとして表示される。
# kernel = np.array([
#     [-2, -1,  0],
#     [-1,  1,  1],  # 左上方向の差分を強調
#     [ 0,  1,  2],
# ], dtype=np.float32)
# img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
# img_dst  = cv2.convertScaleAbs(cv2.filter2D(img_gray, cv2.CV_64F, kernel) + 128)


cv2.imshow('src', img_src)
cv2.imshow('dst', img_dst)

cv2.imwrite(file_dst, img_dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
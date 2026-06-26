import cv2

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
#  フィルタ名          ノイズ除去  エッジ保持  速度   特徴
#  ─────────────────────────────────────────────────────
#  平均値             ◎          ✕          ◎     最もシンプル
#  加重平均           ○          ✕          ◎     中心を重視した平均
#  ガウシアン         ○          △          ◎     自然なぼかし・最もよく使われる
#  中央値             ◎          ○          ○     ゴマ塩ノイズに強い
#  バイラテラル       ○          ◎          △     エッジを残しながらぼかす
#  非局所平均(NLM)    ◎          ◎          ✕     最高品質・最も重い
# ============================================================

# ① 平均値フィルタ
#    近傍ピクセルの平均値を出力する。最もシンプル。
#    カーネルサイズを大きくするほど強くぼける。
#    ksize=(5,5) → 5×5=25ピクセルの平均
img_dst = cv2.blur(img_src, ksize=(5, 5))

# ② 加重平均フィルタ（手動カーネル指定）
#    中心に近いほど重みを大きくした平均。
#    平均値より自然な仕上がり。
#    カーネルの数値が「重み」= 大きいほどその画素の影響が強い
# import numpy as np
# kernel = np.array([
#     [1, 2, 1],
#     [2, 4, 2],  # 中心（4）が一番重い
#     [1, 2, 1],
# ], dtype=np.float32)
# kernel /= kernel.sum()  # 合計が1になるよう正規化（明るさを変えないため）
# img_dst = cv2.filter2D(img_src, ddepth=-1, kernel=kernel)

# ③ ガウシアンフィルタ
#    距離に応じてガウス関数で重み付けした平均。
#    最もよく使われるぼかしフィルタ。
#    ksize は奇数のみ指定可能（3,5,7...）
#    sigmaX が大きいほど広い範囲を参照 → 強くぼける
# img_dst = cv2.GaussianBlur(img_src, ksize=(5, 5), sigmaX=1.0)

# ④ 中央値フィルタ
#    近傍ピクセルを並べて中央の値を出力する。
#    「ゴマ塩ノイズ」（白や黒の点ノイズ）に特に強い。
#    ksize は奇数のみ（3, 5, 7...）
# img_dst = cv2.medianBlur(img_src, ksize=5)

# ⑤ バイラテラルフィルタ
#    「空間的な近さ」と「色の近さ」の両方で重み付けする。
#    エッジ（輪郭）を残しながらノイズだけ除去できる。
#    d         : 参照する近傍の直径（大きいほど広範囲・重い）
#    sigmaColor: 色の許容差（大きいほど遠い色まで平均に含める）
#    sigmaSpace: 空間の許容差（大きいほど遠いピクセルまで参照）
# img_dst = cv2.bilateralFilter(img_src, d=9, sigmaColor=75, sigmaSpace=75)

# ⑥ 非局所平均フィルタ（Non-Local Means）
#    画像全体から「似たパターン」を探して平均する。
#    最も高品質なノイズ除去だが処理が重い。
#    h            : フィルタ強度（大きいほど強くノイズ除去・細部も消える）
#    templateWindowSize: パターン比較するパッチサイズ（奇数、通常7）
#    searchWindowSize  : 類似パッチを探す範囲（奇数、通常21）
# img_dst = cv2.fastNlMeansDenoisingColored(
#     img_src,
#     h=10,
#     hColor=10,
#     templateWindowSize=7,
#     searchWindowSize=21,
# )

cv2.imshow('src', img_src)
cv2.imshow('dst', img_dst)

cv2.imwrite(file_dst, img_dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
# opencvのライブラリ
import cv2

# ２つの画像表示用フレーム（枠）を用意する
cv2.namedWindow('src') # 入力画像表示用フレーム
cv2.namedWindow('dst') # 出力画像表示用フレーム

file_src = 'src.png' # ★処理したい画像のファイル名を入力★
file_dst = 'dst.png' # ★処理後の画像のファイル名を入力★

# 入力用画像ファイルから画像を読み込む 読み込んだ画像情報をimg_srcに蓄える
img_src = cv2.imread(file_src, 1)

# ★ 画像処理の内容を書く ★
# 例：画像を垂直反転する　結果の画像をimg_dstへ蓄える。
img_dst = cv2.flip(img_src, flipCode = 0) # 垂直反転

# フレーム内へ画像を表示する
cv2.imshow('src', img_src) # 入力画像を表示
cv2.imshow('dst', img_dst) # 出力画像を表示

# 出力画像を出力用画像ファイルへ保存する
cv2.imwrite(file_dst, img_dst) 

cv2.waitKey(0) # 何かキーを入力したら次へ進む
cv2.destroyAllWindows() # 全てのフレームを画面から消す

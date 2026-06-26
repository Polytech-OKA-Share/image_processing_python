# opencvのライブラリ
import cv2

# ファイル名を決める
file_src = 'src.png' # 入力用の画像ファイル名
file_dst = 'dst.png' # 出力用の画像ファイル名

# 入力用画像ファイルから画像を読み込む 読み込んだ画像情報をimg_srcに蓄える
img_src = cv2.imread(file_src, 1)
# 明るさのみを扱いやすくするためグレースケールへ変換
img_src = cv2.cvtColor(img_src,cv2. COLOR_BGR2GRAY)

# ２つの画像表示用フレーム（枠）を用意する
cv2.namedWindow('src') # 入力画像表示用フレーム
cv2.namedWindow('dst') # 出力画像表示用フレーム

# ここの処理を色々変えれば、いろいろに異なる画像処理結果を得ます
# ヒストグラム均一化という技法は、画素の度数分布が均一になるように濃度値変換
img_dst = cv2.equalizeHist(img_src)

# フレーム内へ画像を表示する
cv2.imshow('src', img_src) # 入力画像を表示
cv2.imshow('dst', img_dst) # 出力画像を表示

# 出力画像を出力用画像ファイルへ保存する
cv2.imwrite(file_dst, img_dst) 

cv2.waitKey(0) # 何かキーを入力したら次へ進む
cv2.destroyAllWindows() # 全てのフレームを画面から消す

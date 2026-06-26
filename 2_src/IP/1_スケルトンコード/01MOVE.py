# opencvのライブラリ
import cv2
# raspiのカメラを扱うためのライブラリ
from picamera2 import Picamera2

# ２つの画像表示用フレーム（枠）を用意する
cv2.namedWindow('src') # 入力画像表示用フレーム
cv2.namedWindow('dst') # 出力画像表示用フレーム

# 0番目のカメラによって撮影開始（Picamera2版）
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()

# 画像を取り出しては表示するを無限ループで繰り返します
while True:
    img_src = picam2.capture_array()  # カメラ画像の情報から画像を取り出しimg_srcに蓄える
    
    # Picamera2 は RGB なので BGR に変換（OpenCV 用）
    img_src = cv2.cvtColor(img_src, cv2.COLOR_RGB2BGR)
    
    # ★ 画像処理の内容を書く ★
    # 例：画像を垂直反転する　結果の画像をimg_dstへ蓄える。
    img_tmp = cv2.flip(img_src, flipCode = 0) # 垂直反転
    img_dst = cv2.resize(img_tmp, (100, 100))
    
    # フレーム内へ画像を表示する
    cv2.imshow('src', img_src) # 入力画像を表示
    cv2.imshow('dst', img_dst) # 出力画像を表示
    ch = cv2.waitKey(1) # キー入力を１秒待つ
    
    # 「ｑ」キーが押されたら無限ループを抜け出す
    if ch == ord('q'):
        break

picam2.stop() # カメラ撮影終了
picam2.close() # 
cv2.destroyAllWindows() # 全てのフレームを画面から消す

"""_summary_

動作確認用プログラム。事前に、
sudo apt install -y python3-picamera2 python3-opencv
で画像処理に必要なライブラリを取得しておく

"""
from picamera2 import Picamera2
import cv2

# カメラ初期化
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)
picam2.configure(config)
picam2.start()

print("'q'キーで終了")

while True:
    # フレーム取得
    frame = picam2.capture_array()
    
    # 表示
    cv2.imshow("Camera", frame)
    
    # qキーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 後処理
picam2.stop()
cv2.destroyAllWindows()

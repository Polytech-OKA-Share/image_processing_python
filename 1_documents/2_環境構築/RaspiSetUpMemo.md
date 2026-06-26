# Raspiのセットアップ

## 初期設定

1. OSはImagerより最新バージョンを入手する
2. `nmtui` でIPアドレス設定
3. プロキシ設定。内容は分かるだろうってことで略
    * */etc/apt/apt.comf.d/apt.conf*
    * */etc/environment*
    * *~/.bashrc*
4. サーバー側で作業。ホスト名とIPアドレスを登録しインターネット接続可能にする
5. Raspi を Reboot。`sudo apt update`が通ればOK。
6. 一度Raspiを落としてカメラモジュール接続
7. `rpicam-still -o test.jpg` を実行してカメラを認識しているか確認（最新OSの場合、カメラの認識設定不要）

## Python仮想環境　venv なし！！！

1. デスクトップにOpenCV用の環境作成　`python3 -m venv opencv-env`
2. アクティベート　`source opencv-env/bin/activate`
3. OpenCV関連ライブラリ入手　`pip install opencv-python`　`pip install opencv-contrib-python`

# Python環境
純粋にopencv、picamera2、spyderを地道に`sudo apt install`で！！！
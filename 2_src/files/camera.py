"""
camera.py - 学習済みCNNモデルでPicamera2のリアルタイム判別
使い方:
  python camera.py --model model.pkl
  q キーで終了
"""

import cv2
import numpy as np
import pickle
import argparse
from picamera2 import Picamera2

# ─────────────────────────────────────────
# モデル推論に必要な関数（train.pyと共通）
# ─────────────────────────────────────────
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def im2col(x_pad, kH, kW, stride):
    N, C, H, W = x_pad.shape
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1
    col_mat = np.zeros((N, C * kH * kW, out_H, out_W))
    ki_idx = 0
    for i in range(kH):
        for j in range(kW):
            col_mat[:, ki_idx*C:(ki_idx+1)*C, :, :] = \
                x_pad[:, :, i:i+out_H*stride:stride, j:j+out_W*stride:stride]
            ki_idx += 1
    return col_mat.transpose(0, 2, 3, 1).reshape(N * out_H * out_W, -1)

def conv2d_forward(x, W, b, stride=1, pad=0):
    N, C, H, Ww = x.shape
    F, _, kH, kW = W.shape
    out_H = (H + 2 * pad - kH) // stride + 1
    out_W = (Ww + 2 * pad - kW) // stride + 1
    x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
    col   = im2col(x_pad, kH, kW, stride)
    W_col = W.reshape(F, -1).T
    out   = col @ W_col + b
    return out.reshape(N, out_H, out_W, F).transpose(0, 3, 1, 2)

def maxpool_forward(x, pool=2, stride=2):
    N, C, H, W = x.shape
    out_H = (H - pool) // stride + 1
    out_W = (W - pool) // stride + 1
    out = np.zeros((N, C, out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            patch = x[:, :, i*stride:i*stride+pool, j*stride:j*stride+pool]
            out[:, :, i, j] = patch.reshape(N, C, -1).max(axis=2)
    return out

def predict(img_bgr, params, img_size):
    """BGR画像を受け取り (クラスindex, 確率) を返す"""
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_size, img_size))
    x       = resized.astype(np.float32) / 255.0
    x       = x[np.newaxis, np.newaxis, :, :]   # (1,1,H,W)

    out1  = relu(conv2d_forward(x, params['W1'], params['b1'], pad=1))
    pool1 = maxpool_forward(out1)
    out2  = relu(conv2d_forward(pool1, params['W2'], params['b2'], pad=1))
    pool2 = maxpool_forward(out2)
    flat  = pool2.reshape(1, -1)
    fc1   = relu(flat @ params['W3'] + params['b3'])
    fc2   = fc1 @ params['W4'] + params['b4']
    prob  = sigmoid(fc2)[0, 0]

    class_idx  = int(prob >= 0.5)
    confidence = prob if class_idx == 1 else 1.0 - prob
    return class_idx, float(confidence)

# ─────────────────────────────────────────
# 判定結果オーバーレイ描画
# ─────────────────────────────────────────
def draw_result(frame, class_name, confidence, class_idx):
    """スケルトンの 'dst' フレームに判定結果を描画"""
    dst = frame.copy()
    h, w = dst.shape[:2]

    color = (0, 200, 0) if class_idx == 1 else (0, 80, 255)

    # 上部バー背景
    cv2.rectangle(dst, (0, 0), (w, 55), (30, 30, 30), -1)

    # クラス名テキスト
    cv2.putText(dst, f"Class: {class_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 確信度バー
    bar_w = int(w * confidence)
    cv2.rectangle(dst, (0, 40), (bar_w, 55), color, -1)
    cv2.putText(dst, f"{confidence*100:.1f}%", (w - 80, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # 外枠
    cv2.rectangle(dst, (5, 60), (w - 5, h - 5), color, 3)
    return dst

# ─────────────────────────────────────────
# メイン
# ─────────────────────────────────────────
def main(model_path):
    # モデル読み込み
    print(f"モデル読み込み中: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    params      = model['params']
    class_names = model['class_names']
    img_size    = model['img_size']
    print(f"  クラス: {class_names[0]} (0) / {class_names[1]} (1)")

    # ２つの画像表示用フレーム（枠）を用意する
    cv2.namedWindow('src')  # 入力画像表示用フレーム
    cv2.namedWindow('dst')  # 出力画像表示用フレーム

    # 0番目のカメラによって撮影開始（Picamera2版）
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration())
    picam2.start()

    print("カメラ起動中... [q] で終了")
    frame_count = 0
    class_idx, confidence, class_name = 0, 0.5, class_names[0]

    # 画像を取り出しては表示するを無限ループで繰り返す
    while True:
        img_src = picam2.capture_array()  # カメラ画像の情報から画像を取り出しimg_srcに蓄える

        # Picamera2 は RGB なので BGR に変換（OpenCV 用）
        img_src = cv2.cvtColor(img_src, cv2.COLOR_RGB2BGR)

        # ★ 画像処理の内容 ★
        # 5フレームに1回CNNで推論（ラズパイ負荷軽減）
        if frame_count % 5 == 0:
            class_idx, confidence = predict(img_src, params, img_size)
            class_name = class_names[class_idx]

        # 判定結果オーバーレイを dst に描画
        img_dst = draw_result(img_src, class_name, confidence, class_idx)

        # フレーム内へ画像を表示する
        cv2.imshow('src', img_src)  # 入力画像を表示
        cv2.imshow('dst', img_dst)  # 出力画像を表示
        ch = cv2.waitKey(1)         # キー入力を１秒待つ

        # 「q」キーが押されたら無限ループを抜け出す
        if ch == ord('q'):
            break

        frame_count += 1

    picam2.stop()            # カメラ撮影終了
    picam2.close()           #
    cv2.destroyAllWindows()  # 全てのフレームを画面から消す
    print("終了")

# ─────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNNリアルタイムカメラ判別 (Picamera2)')
    parser.add_argument('--model', default='model.pkl', help='モデルファイルパス')
    args = parser.parse_args()
    main(args.model)

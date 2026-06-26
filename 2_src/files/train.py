"""
train.py - NumPy+OpenCVのみで実装したCNNの学習スクリプト
使い方:
  python train.py --class0 ./images/class0 --class1 ./images/class1
"""

import cv2
import numpy as np
import os
import argparse
import pickle
import time

# ─────────────────────────────────────────
# ハイパーパラメータ
# ─────────────────────────────────────────
IMG_SIZE   = 32        # 入力画像サイズ (32x32)
EPOCHS     = 30
LR         = 0.01
BATCH_SIZE = 16

# ─────────────────────────────────────────
# 活性化関数・損失関数
# ─────────────────────────────────────────
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def bce_loss(pred, label):
    eps = 1e-7
    return -np.mean(label * np.log(pred + eps) + (1 - label) * np.log(1 - pred + eps))

# ─────────────────────────────────────────
# 畳み込み層 (NumPy実装)
# ─────────────────────────────────────────
def conv2d_forward(x, W, b, stride=1, pad=0):
    """
    x: (N, C, H, W)
    W: (F, C, kH, kW)
    """
    N, C, H, Ww = x.shape
    F, _, kH, kW = W.shape
    out_H = (H + 2 * pad - kH) // stride + 1
    out_W = (Ww + 2 * pad - kW) // stride + 1

    x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
    col   = im2col(x_pad, kH, kW, stride)   # (N*out_H*out_W, C*kH*kW)
    W_col = W.reshape(F, -1).T              # (C*kH*kW, F)
    out   = col @ W_col + b                 # (N*out_H*out_W, F)
    out   = out.reshape(N, out_H, out_W, F).transpose(0, 3, 1, 2)
    return out, col, x_pad

def conv2d_backward(dout, col, x_pad, W, b, stride=1, kH=3, kW=3):
    F = dout.shape[1]
    N, _, out_H, out_W = dout.shape

    dout_flat = dout.transpose(0,2,3,1).reshape(-1, F)  # (N*out_H*out_W, F)
    dW = (col.T @ dout_flat).T.reshape(W.shape)
    db = dout_flat.sum(axis=0)

    W_col  = W.reshape(F, -1)
    dcol   = dout_flat @ W_col                          # (N*out_H*out_W, C*kH*kW)
    dx_pad = col2im(dcol, x_pad.shape, kH, kW, stride)
    # パディング除去
    if kH > 1:
        dx = dx_pad[:, :, 1:-1, 1:-1]
    else:
        dx = dx_pad
    return dx, dW, db

def im2col(x_pad, kH, kW, stride):
    N, C, H, W = x_pad.shape
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1
    cols = []
    for i in range(kH):
        for j in range(kW):
            col = x_pad[:, :, i:i+out_H*stride:stride, j:j+out_W*stride:stride]
            cols.append(col.reshape(N, C, out_H, out_W))
    # (N, C*kH*kW, out_H, out_W) → (N*out_H*out_W, C*kH*kW)
    col_mat = np.stack(cols, axis=1)          # (N, kH*kW, C, out_H, out_W) ← wrong
    # 正しい変換
    col_mat2 = np.zeros((N, C * kH * kW, out_H, out_W))
    idx = 0
    for i in range(kH):
        for j in range(kW):
            col_mat2[:, idx*C:(idx+1)*C, :, :] = x_pad[:, :, i:i+out_H*stride:stride, j:j+out_W*stride:stride]
            idx += 1
    return col_mat2.transpose(0,2,3,1).reshape(N * out_H * out_W, -1)

def col2im(dcol, x_pad_shape, kH, kW, stride):
    N, C, H, W = x_pad_shape
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1
    # dcol: (N*out_H*out_W, C*kH*kW)
    # dcol_4d: (N, C*kH*kW, out_H, out_W)
    dcol_4d = dcol.reshape(N, out_H, out_W, C * kH * kW).transpose(0, 3, 1, 2)
    dx_pad = np.zeros(x_pad_shape)
    ki_idx = 0
    for i in range(kH):
        for j in range(kW):
            # ki_idx番目のカーネル位置に対応するチャンネルブロック: [ki_idx*C : (ki_idx+1)*C]
            # dx_padのチャンネル次元はC (0..C-1) なので、C全チャンネルに加算
            dx_pad[:, :, i:i+out_H*stride:stride, j:j+out_W*stride:stride] += \
                dcol_4d[:, ki_idx*C:(ki_idx+1)*C, :, :]
            ki_idx += 1
    return dx_pad

# ─────────────────────────────────────────
# MaxPooling層
# ─────────────────────────────────────────
def maxpool_forward(x, pool=2, stride=2):
    N, C, H, W = x.shape
    out_H = (H - pool) // stride + 1
    out_W = (W - pool) // stride + 1
    out   = np.zeros((N, C, out_H, out_W))
    mask  = np.zeros_like(x)
    for i in range(out_H):
        for j in range(out_W):
            patch = x[:, :, i*stride:i*stride+pool, j*stride:j*stride+pool]
            out[:, :, i, j] = patch.reshape(N, C, -1).max(axis=2)
            idx = patch.reshape(N, C, -1).argmax(axis=2)
            mi = idx // pool + i * stride
            mj = idx % pool  + j * stride
            for n in range(N):
                for c in range(C):
                    mask[n, c, mi[n,c], mj[n,c]] += 1
    return out, mask

def maxpool_backward(dout, mask, pool=2, stride=2):
    N, C, out_H, out_W = dout.shape
    dx = np.zeros(mask.shape)
    for i in range(out_H):
        for j in range(out_W):
            dx[:, :, i*stride:i*stride+pool, j*stride:j*stride+pool] += \
                (mask[:, :, i*stride:i*stride+pool, j*stride:j*stride+pool] > 0) * \
                dout[:, :, i:i+1, j:j+1]
    return dx

# ─────────────────────────────────────────
# 全結合層
# ─────────────────────────────────────────
def fc_forward(x, W, b):
    return x @ W + b

def fc_backward(dout, x, W):
    dx = dout @ W.T
    dW = x.T @ dout
    db = dout.sum(axis=0)
    return dx, dW, db

# ─────────────────────────────────────────
# ネットワーク定義
# アーキテクチャ: Conv(8) → Pool → Conv(16) → Pool → FC(64) → FC(1)
# ─────────────────────────────────────────
def init_network():
    np.random.seed(42)
    fc_in = 16 * (IMG_SIZE // 4) * (IMG_SIZE // 4)  # Pool×2後のサイズ
    params = {
        # Conv1: 1ch入力→8フィルタ、3x3
        'W1': np.random.randn(8, 1, 3, 3) * np.sqrt(2.0 / (1*3*3)),
        'b1': np.zeros(8),
        # Conv2: 8ch→16フィルタ、3x3
        'W2': np.random.randn(16, 8, 3, 3) * np.sqrt(2.0 / (8*3*3)),
        'b2': np.zeros(16),
        # Pool×2で IMG_SIZE→IMG_SIZE//2→IMG_SIZE//4、16ch
        'W3': np.random.randn(fc_in, 64) * np.sqrt(2.0 / fc_in),
        'b3': np.zeros(64),
        'W4': np.random.randn(64, 1) * np.sqrt(2.0 / 64),
        'b4': np.zeros(1),
    }
    velocities = {k: np.zeros_like(v) for k, v in params.items()}
    return params, velocities

def forward(x, params, training=True):
    cache = {}
    # Conv1 + ReLU + Pool
    out1, col1, xpad1 = conv2d_forward(x, params['W1'], params['b1'], pad=1)
    act1 = relu(out1)
    pool1, mask1 = maxpool_forward(act1)

    # Conv2 + ReLU + Pool
    out2, col2, xpad2 = conv2d_forward(pool1, params['W2'], params['b2'], pad=1)
    act2 = relu(out2)
    pool2, mask2 = maxpool_forward(act2)

    # Flatten
    N = x.shape[0]
    flat = pool2.reshape(N, -1)

    # FC1 + ReLU
    fc1 = fc_forward(flat, params['W3'], params['b3'])
    act3 = relu(fc1)

    # FC2 + Sigmoid
    fc2 = fc_forward(act3, params['W4'], params['b4'])
    pred = sigmoid(fc2).reshape(-1)

    cache = {
        'x': x, 'out1': out1, 'col1': col1, 'xpad1': xpad1,
        'act1': act1, 'pool1': pool1, 'mask1': mask1,
        'out2': out2, 'col2': col2, 'xpad2': xpad2,
        'act2': act2, 'pool2': pool2, 'mask2': mask2,
        'flat': flat, 'fc1': fc1, 'act3': act3,
        'fc2': fc2,
    }
    return pred, cache

def backward(pred, label, cache, params, lr, velocities, momentum=0.9):
    N = pred.shape[0]
    # 出力層の勾配
    dpred = (pred - label) / N
    dfc2  = dpred.reshape(-1, 1) * sigmoid(cache['fc2']) * (1 - sigmoid(cache['fc2']))
    dact3, dW4, db4 = fc_backward(dfc2, cache['act3'], params['W4'])

    dfc1 = dact3 * relu_grad(cache['fc1'])
    dflat, dW3, db3 = fc_backward(dfc1, cache['flat'], params['W3'])

    # Unflatten
    dpool2 = dflat.reshape(cache['pool2'].shape)

    # Pool2 backward
    dact2  = maxpool_backward(dpool2, cache['mask2'])
    dout2  = dact2 * relu_grad(cache['out2'])
    dpool1, dW2, db2 = conv2d_backward(dout2, cache['col2'], cache['xpad2'],
                                        params['W2'], params['b2'], kH=3, kW=3)

    # Pool1 backward
    dact1  = maxpool_backward(dpool1, cache['mask1'])
    dout1  = dact1 * relu_grad(cache['out1'])
    _, dW1, db1 = conv2d_backward(dout1, cache['col1'], cache['xpad1'],
                                   params['W1'], params['b1'], kH=3, kW=3)

    # モメンタムSGD更新
    grads = {'W1':dW1,'b1':db1,'W2':dW2,'b2':db2,'W3':dW3,'b3':db3,'W4':dW4,'b4':db4}
    for k in params:
        velocities[k] = momentum * velocities[k] - lr * grads[k]
        params[k]    += velocities[k]

# ─────────────────────────────────────────
# データ前処理
# ─────────────────────────────────────────
def preprocess(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    norm = resized.astype(np.float32) / 255.0
    return norm[np.newaxis, np.newaxis, :, :]   # (1, 1, H, W)

def load_dataset(dir0, dir1):
    X, Y = [], []
    for label, d in enumerate([dir0, dir1]):
        for fname in os.listdir(d):
            if not fname.lower().endswith(('.png','.jpg','.jpeg','.bmp')):
                continue
            img = cv2.imread(os.path.join(d, fname))
            if img is None:
                continue
            x = preprocess(img)
            X.append(x)
            Y.append(float(label))
    X = np.concatenate(X, axis=0)  # (N,1,32,32)
    Y = np.array(Y)
    # シャッフル
    idx = np.random.permutation(len(Y))
    return X[idx], Y[idx]

# ─────────────────────────────────────────
# 学習ループ
# ─────────────────────────────────────────
def train(dir0, dir1, save_path='model.pkl', class_names=None):
    print("=== データ読み込み中... ===")
    X, Y = load_dataset(dir0, dir1)
    print(f"  クラス0: {(Y==0).sum()}枚  クラス1: {(Y==1).sum()}枚")

    params, velocities = init_network()

    if class_names is None:
        # 末尾スラッシュを除去してフォルダ名を正しく取得
        class_names = [os.path.basename(dir0.rstrip('/\\')),
                       os.path.basename(dir1.rstrip('/\\'))]

    n = len(Y)
    print(f"\n=== 学習開始 (EPOCHS={EPOCHS}, BATCH={BATCH_SIZE}, LR={LR}) ===")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        idx = np.random.permutation(n)
        X, Y = X[idx], Y[idx]
        losses = []
        correct = 0
        for i in range(0, n, BATCH_SIZE):
            xb = X[i:i+BATCH_SIZE]
            yb = Y[i:i+BATCH_SIZE]
            pred, cache = forward(xb, params)
            loss = bce_loss(pred, yb)
            losses.append(loss)
            backward(pred, yb, cache, params, LR, velocities)
            correct += ((pred >= 0.5) == yb).sum()

        acc  = correct / n * 100
        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{EPOCHS}  loss={np.mean(losses):.4f}  acc={acc:.1f}%  ({elapsed:.1f}s)")

    # 保存
    model = {'params': params, 'class_names': class_names, 'img_size': IMG_SIZE}
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✅ モデルを {save_path} に保存しました")
    print(f"   クラス0 = '{class_names[0]}'")
    print(f"   クラス1 = '{class_names[1]}'")

# ─────────────────────────────────────────
# エントリポイント
# ─────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN学習スクリプト (NumPy+OpenCV)')
    parser.add_argument('--class0', required=True, help='クラス0の画像フォルダ')
    parser.add_argument('--class1', required=True, help='クラス1の画像フォルダ')
    parser.add_argument('--model',  default='model.pkl', help='保存先モデルパス')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr',     type=float, default=LR)
    parser.add_argument('--batch',  type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    EPOCHS     = args.epochs
    LR         = args.lr
    BATCH_SIZE = args.batch

    train(args.class0, args.class1, save_path=args.model)

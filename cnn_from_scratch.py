"""
Convolutional Neural Network — built from scratch using only NumPy.

No PyTorch, no TensorFlow, no Keras. Every layer implements its own
forward pass AND backward pass (manual backpropagation + gradient descent).

Layers implemented:
    - Conv2D        (2D convolution, valid padding, stride 1 or more)
    - ReLU
    - MaxPool2D
    - Flatten
    - Dense (fully connected)
    - Softmax + Cross-Entropy loss

Includes a runnable demo that trains the network to classify small
synthetic images (so it works with zero external datasets / internet
access). Swap in real image data (e.g. MNIST) by replacing
`make_synthetic_dataset()`.

Run:
    python cnn_from_scratch.py
"""

import numpy as np


# --------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------

def _im2col(x, kh, kw, stride):
    """
    Turn image patches into columns so convolution becomes a matrix multiply.

    x: (N, C, H, W)
    Returns: (N, out_h, out_w, C*kh*kw), out_h, out_w
    """
    N, C, H, W = x.shape
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1

    cols = np.zeros((N, out_h, out_w, C, kh, kw), dtype=x.dtype)
    for i in range(out_h):
        for j in range(out_w):
            hs, ws = i * stride, j * stride
            cols[:, i, j, :, :, :] = x[:, :, hs:hs + kh, ws:ws + kw]

    cols = cols.reshape(N, out_h, out_w, C * kh * kw)
    return cols, out_h, out_w


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------

class Conv2D:
    """2D convolution layer: (N, C_in, H, W) -> (N, C_out, H', W')."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, lr=0.01):
        self.C_in = in_channels
        self.C_out = out_channels
        self.k = kernel_size
        self.stride = stride
        self.lr = lr

        # He initialization
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels)

        self._cache = None

    def forward(self, x):
        N, C, H, W = x.shape
        k, s = self.k, self.stride

        cols, out_h, out_w = _im2col(x, k, k, s)          # (N, oh, ow, C*k*k)
        cols_flat = cols.reshape(N * out_h * out_w, C * k * k)

        W_flat = self.W.reshape(self.C_out, C * k * k).T   # (C*k*k, C_out)
        out = cols_flat @ W_flat + self.b                  # (N*oh*ow, C_out)
        out = out.reshape(N, out_h, out_w, self.C_out).transpose(0, 3, 1, 2)

        self._cache = (x, cols_flat, out_h, out_w)
        return out

    def backward(self, dout):
        x, cols_flat, out_h, out_w = self._cache
        N, C, H, W = x.shape
        k, s = self.k, self.stride

        dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, self.C_out)  # (N*oh*ow, C_out)

        # Gradients w.r.t. weights and bias
        dW = (dout_flat.T @ cols_flat).reshape(self.C_out, C, k, k)
        db = dout_flat.sum(axis=0)

        # Gradient w.r.t. input (scatter-add patches back)
        W_flat = self.W.reshape(self.C_out, C * k * k)      # (C_out, C*k*k)
        dcols_flat = dout_flat @ W_flat                      # (N*oh*ow, C*k*k)
        dcols = dcols_flat.reshape(N, out_h, out_w, C, k, k)

        dx = np.zeros_like(x)
        for i in range(out_h):
            for j in range(out_w):
                hs, ws = i * s, j * s
                dx[:, :, hs:hs + k, ws:ws + k] += dcols[:, i, j, :, :, :]

        # SGD update
        self.W -= self.lr * dW / N
        self.b -= self.lr * db / N
        return dx


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class MaxPool2D:
    def __init__(self, size=2, stride=2):
        self.size = size
        self.stride = stride

    def forward(self, x):
        N, C, H, W = x.shape
        s, k = self.stride, self.size
        out_h, out_w = (H - k) // s + 1, (W - k) // s + 1

        out = np.zeros((N, C, out_h, out_w))
        self._argmax = np.zeros((N, C, out_h, out_w, 2), dtype=int)

        for i in range(out_h):
            for j in range(out_w):
                hs, ws = i * s, j * s
                window = x[:, :, hs:hs + k, ws:ws + k]              # (N,C,k,k)
                flat = window.reshape(N, C, -1)
                idx = np.argmax(flat, axis=-1)
                out[:, :, i, j] = np.take_along_axis(flat, idx[..., None], axis=-1)[..., 0]
                self._argmax[:, :, i, j, 0] = idx // k
                self._argmax[:, :, i, j, 1] = idx % k

        self._x_shape = x.shape
        return out

    def backward(self, dout):
        N, C, H, W = self._x_shape
        s, k = self.stride, self.size
        out_h, out_w = dout.shape[2], dout.shape[3]

        dx = np.zeros((N, C, H, W))
        for i in range(out_h):
            for j in range(out_w):
                hs, ws = i * s, j * s
                for n in range(N):
                    for c in range(C):
                        di, dj = self._argmax[n, c, i, j]
                        dx[n, c, hs + di, ws + dj] += dout[n, c, i, j]
        return dx


class Flatten:
    def forward(self, x):
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self._shape)


class Dense:
    def __init__(self, in_dim, out_dim, lr=0.01):
        scale = np.sqrt(2.0 / in_dim)
        self.W = np.random.randn(in_dim, out_dim) * scale
        self.b = np.zeros(out_dim)
        self.lr = lr

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        N = self.x.shape[0]
        dW = self.x.T @ dout
        db = dout.sum(axis=0)
        dx = dout @ self.W.T

        self.W -= self.lr * dW / N
        self.b -= self.lr * db / N
        return dx


class Dropout:
    """Randomly zeroes activations during training to reduce overfitting."""

    def __init__(self, p=0.3):
        self.p = p
        self.training = True

    def forward(self, x):
        if not self.training:
            return x
        self.mask = (np.random.rand(*x.shape) > self.p) / (1 - self.p)
        return x * self.mask

    def backward(self, dout):
        if not self.training:
            return dout
        return dout * self.mask


class SoftmaxCrossEntropy:
    """Combined softmax + cross-entropy loss (numerically stable)."""

    def forward(self, logits, labels):
        # labels: integer class indices, shape (N,)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.probs = exp / exp.sum(axis=1, keepdims=True)
        self.labels = labels

        N = logits.shape[0]
        correct_logp = -np.log(self.probs[np.arange(N), labels] + 1e-9)
        return correct_logp.mean()

    def backward(self):
        N = self.labels.shape[0]
        dlogits = self.probs.copy()
        dlogits[np.arange(N), self.labels] -= 1
        return dlogits / N


# --------------------------------------------------------------------------
# Network wrapper
# --------------------------------------------------------------------------

class CNN:
    def __init__(self, layers):
        self.layers = layers
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def train_step(self, x, y):
        logits = self.forward(x)
        loss = self.loss_fn.forward(logits, y)
        dlogits = self.loss_fn.backward()
        self.backward(dlogits)

        preds = logits.argmax(axis=1)
        acc = (preds == y).mean()
        return loss, acc

    def predict(self, x):
        self.set_training(False)
        out = self.forward(x).argmax(axis=1)
        self.set_training(True)
        return out

    def set_training(self, flag):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.training = flag

    def set_lr(self, lr):
        for layer in self.layers:
            if hasattr(layer, "lr"):
                layer.lr = lr

    # --- persistence -----------------------------------------------------
    def save(self, path):
        params = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "W"):
                params[f"W{i}"] = layer.W
                params[f"b{i}"] = layer.b
        np.savez(path, **params)

    def load(self, path):
        data = np.load(path)
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "W"):
                layer.W = data[f"W{i}"]
                layer.b = data[f"b{i}"]


# --------------------------------------------------------------------------
# Dataset loaders
# --------------------------------------------------------------------------

def load_dataset(seed=0):
    """8x8 handwritten digits, built into scikit-learn — no download needed."""
    from sklearn.datasets import load_digits

    data = load_digits()
    X = data.images.astype(np.float64) / 16.0        # scale pixels to [0, 1]
    X = X[:, None, :, :]                              # add channel dim -> (N,1,8,8)
    y = data.target

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    split = int(0.85 * len(X))
    return X[:split], y[:split], X[split:], y[split:]


def load_mnist_csv(train_path, test_path, max_train=5000, max_test=1000, seed=0):
    """
    Load real MNIST (28x28, 10 classes) from the classic CSV format:
    https://github.com/phoebetronic/mnist  (mnist_train.csv, mnist_test.csv)
    Each row: label, then 784 pixel values (0-255).

    max_train / max_test cap how many rows are read, since this pure-NumPy
    implementation (no GPU, no vectorized batched conv library) is far slower
    per-image than PyTorch/TensorFlow. A few thousand images trains in
    minutes on a laptop CPU; the full 60,000 would take much longer.
    """
    rng = np.random.default_rng(seed)

    def _load(path, cap):
        raw = np.loadtxt(path, delimiter=",", max_rows=cap + 1, skiprows=0)
        # some copies of this CSV have no header row; np.loadtxt handles that fine
        y = raw[:, 0].astype(int)
        X = raw[:, 1:].astype(np.float64) / 255.0
        X = X.reshape(-1, 1, 28, 28)
        idx = rng.permutation(len(X))
        return X[idx][:cap], y[idx][:cap]

    X_train, y_train = _load(train_path, max_train)
    X_test, y_test = _load(test_path, max_test)
    return X_train, y_train, X_test, y_test


def build_network(lr=0.05, n_classes=10):
    """Small network for 8x8 inputs (the built-in digits dataset)."""
    return CNN([
        Conv2D(in_channels=1, out_channels=8, kernel_size=3, stride=1, lr=lr),   # 8x8 -> 6x6
        ReLU(),
        Conv2D(in_channels=8, out_channels=16, kernel_size=3, stride=1, lr=lr),  # 6x6 -> 4x4
        ReLU(),
        MaxPool2D(size=2, stride=2),                                            # 4x4 -> 2x2
        Flatten(),
        Dense(in_dim=16 * 2 * 2, out_dim=32, lr=lr),
        ReLU(),
        Dropout(p=0.25),
        Dense(in_dim=32, out_dim=n_classes, lr=lr),
    ])


def build_network_mnist(lr=0.05, n_classes=10):
    """Slightly larger network for real 28x28 MNIST inputs."""
    return CNN([
        Conv2D(in_channels=1, out_channels=8, kernel_size=3, stride=1, lr=lr),   # 28x28 -> 26x26
        ReLU(),
        MaxPool2D(size=2, stride=2),                                            # 26x26 -> 13x13
        Conv2D(in_channels=8, out_channels=16, kernel_size=3, stride=1, lr=lr),  # 13x13 -> 11x11
        ReLU(),
        MaxPool2D(size=2, stride=2),                                            # 11x11 -> 5x5
        Flatten(),
        Dense(in_dim=16 * 5 * 5, out_dim=64, lr=lr),
        ReLU(),
        Dropout(p=0.25),
        Dense(in_dim=64, out_dim=n_classes, lr=lr),
    ])


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["digits", "mnist"], default="digits",
                         help="'digits' = built-in 8x8 demo (no download). "
                              "'mnist' = real 28x28 MNIST from CSV files (needs --train_csv/--test_csv).")
    parser.add_argument("--train_csv", default="mnist_train.csv")
    parser.add_argument("--test_csv", default="mnist_test.csv")
    parser.add_argument("--max_train", type=int, default=5000)
    parser.add_argument("--max_test", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    if args.dataset == "mnist":
        print(f"Loading MNIST from {args.train_csv} / {args.test_csv} "
              f"(capped at {args.max_train} train / {args.max_test} test rows)...")
        X_train, y_train, X_test, y_test = load_mnist_csv(
            args.train_csv, args.test_csv, args.max_train, args.max_test)
        net = build_network_mnist(lr=0.1)
        base_lr = 0.1
    else:
        print("Loading digit dataset (8x8 handwritten digits, 10 classes)...")
        X_train, y_train, X_test, y_test = load_dataset()
        net = build_network(lr=0.15)
        base_lr = 0.15

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    epochs = args.epochs
    batch_size = 32
    best_acc = 0.0

    for epoch in range(epochs):
        # simple learning-rate decay: start fast, slow down as training progresses
        lr = base_lr * (0.97 ** epoch)
        net.set_lr(lr)

        perm = np.random.permutation(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]

        losses, accs = [], []
        for i in range(0, len(X_train), batch_size):
            xb = X_train[i:i + batch_size]
            yb = y_train[i:i + batch_size]
            loss, acc = net.train_step(xb, yb)
            losses.append(loss)
            accs.append(acc)

        test_preds = net.predict(X_test)
        test_acc = (test_preds == y_test).mean()
        best_acc = max(best_acc, test_acc)

        print(f"Epoch {epoch + 1:2d}/{epochs} | lr={lr:.4f} | "
              f"train_loss={np.mean(losses):.4f} | "
              f"train_acc={np.mean(accs):.3f} | "
              f"test_acc={test_acc:.3f}")

    print(f"\nDone. Final test accuracy: {test_acc:.3f} | Best: {best_acc:.3f}")

    # Save trained weights so they can be reused without retraining
    net.save("cnn_weights.npz")
    print("Weights saved to cnn_weights.npz")

    # --- Example: predicting a single new image -----------------------
    sample = X_test[0:1]
    pred = net.predict(sample)[0]
    print(f"\nExample single prediction: true label = {y_test[0]}, predicted = {pred}")


if __name__ == "__main__":
    main()

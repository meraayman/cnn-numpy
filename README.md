# CNN From Scratch (NumPy Only)

A convolutional neural network implemented entirely from first principles in **NumPy**, with no reliance on PyTorch, TensorFlow, Keras, or any autograd engine. Every layer defines its own forward pass and its own analytically-derived backward pass, including manual backpropagation through 2D convolution, max-pooling, and fully-connected layers.

The goal of this project is pedagogical: to demonstrate a correct, working implementation of the mechanics that autograd frameworks normally hide — im2col convolution, gradient routing through a pooling operation, and end-to-end training via plain stochastic gradient descent — and to validate that implementation empirically on real image data (MNIST).

---

## Table of Contents

- [Motivation](#motivation)
- [Architecture](#architecture)
- [Mathematical Background](#mathematical-background)
- [Correctness Verification](#correctness-verification)
- [Repository Contents](#repository-contents)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Use Cases](#use-cases--who-this-is-for)
- [Limitations](#limitations)
- [Roadmap](#roadmap--future-work)
- [License](#license)

---

## Motivation

Most "CNN from scratch" tutorials either (a) stop at a toy example that never converges to a meaningful accuracy, or (b) quietly fall back on an autograd library for the backward pass, which defeats the purpose of writing one "from scratch." This project instead:

1. Implements **every** forward and backward pass by hand, with no automatic differentiation.
2. Validates correctness with a **gradient-check-style sanity test** before trusting the implementation on real data (see [Correctness Verification](#correctness-verification)).
3. Trains and reports results on **real MNIST digits**, not just a synthetic or built-in toy set, so the numbers reflect a genuine, reproducible benchmark.

## Architecture

Each layer below is implemented as an independent class exposing `forward()` and `backward()` methods, so layers can be composed into arbitrary feed-forward stacks:

| Layer | Description |
|---|---|
| `Conv2D` | 2D convolution using the **im2col** trick — image patches are unfolded into a matrix so the convolution reduces to a single matrix multiplication. Supports configurable stride, valid (no) padding, and He-initialized weights. |
| `ReLU` | Elementwise rectified linear activation. |
| `MaxPool2D` | Spatial max-pooling. The backward pass routes each upstream gradient back to the exact pixel that won its pooling window, using cached argmax indices from the forward pass. |
| `Flatten` | Reshapes feature maps into a flat vector for the dense head. |
| `Dense` | Fully-connected (affine) layer. |
| `Dropout` | Inverted dropout with a `training` flag, so it is active during training and disabled (identity) at inference time. |
| `SoftmaxCrossEntropy` | Combined softmax + cross-entropy loss, computed in a numerically stable way (max-subtraction trick) with a closed-form combined gradient. |

Two ready-made network configurations are provided:

- **`build_network()`** — a compact network for the built-in 8×8 scikit-learn digit dataset.
- **`build_network_mnist()`** — a larger network for real 28×28 MNIST images, which also generalizes to color input (`in_channels=3`) for non-MNIST use.

Also included are `preprocess_image()` and `load_images_from_folder()`, which resize an arbitrary input image — any original resolution, grayscale or color — down to a fixed shape suitable for the network. This removes the fixed-input-size constraint for anyone who wants to point the network at a custom image dataset instead of MNIST.

## Mathematical Background

- **Convolution as matrix multiplication (im2col):** rather than sliding a kernel with nested loops, each valid receptive-field patch of the input is extracted and flattened into a row of a matrix `cols`. Convolution then becomes `cols @ W_flat`, which lets NumPy's BLAS-backed matrix multiplication do the heavy lifting instead of interpreted Python loops.
- **Convolution backward pass:** gradients with respect to the weights are computed as `dout_flat.T @ cols_flat`; the gradient with respect to the input is obtained by projecting `dout` back through the weights into patch space and then scatter-adding overlapping patches back into the input tensor (accounting for the fact that, with stride < kernel size, each input pixel can contribute to multiple output positions).
- **Max-pooling backward pass:** since max-pooling is a piecewise-selection function, its gradient is a routing problem, not an arithmetic one — the incoming gradient for each pooling window is placed entirely at the input location that produced the maximum in the forward pass, and zero everywhere else in that window.
- **Softmax + cross-entropy:** implemented as a single fused operation both for numerical stability (subtracting the row-wise max before exponentiating) and because the combined gradient `softmax(logits) - one_hot(labels)` is simpler and cheaper than differentiating through softmax and cross-entropy separately.

## Correctness Verification

Hand-derived backpropagation is easy to get subtly wrong — a mistake can still produce a network that trains *slowly* rather than one that visibly fails, which makes bugs hard to catch on a full dataset. Before trusting this implementation on real data, its gradients were validated using an **overfitting sanity check**: the network was trained on a fixed batch of 16 images with no regularization until it reached 100% accuracy on that batch. A network that cannot memorize a handful of examples almost certainly has incorrect gradients, regardless of how plausible the code looks. Only after passing this check was the implementation trained on the full MNIST pipeline.

This is a standard debugging technique in deep learning research and is closer in spirit to a lightweight numerical gradient check than to ordinary integration testing.

## Repository Contents

```
.
├── cnn_from_scratch.py   # All layers, network configs, training loop, CLI entry point
├── README.md
├── cnn_weights.npz       # Pretrained weights from a completed MNIST training run
├── mnist_train.csv       # MNIST training set (CSV format, label + 784 pixel values per row)
└── mnist_test.csv        # MNIST test set (same format)
```

## Installation

```bash
pip install numpy scikit-learn pillow
```

- **NumPy** — all layer math.
- **scikit-learn** — only used for the built-in 8×8 digit demo dataset (`load_digits`); not required for MNIST mode.
- **Pillow** — only needed if you use `preprocess_image()` / `load_images_from_folder()` on your own images.

For MNIST training from raw data, download `mnist_train.csv.zip` and `mnist_test.csv.zip` from [phoebetronic/mnist](https://github.com/phoebetronic/mnist), unzip them, and place `mnist_train.csv` and `mnist_test.csv` in the project folder (or pass explicit paths with `--train_csv` / `--test_csv`).

## Usage

### 1. Quick demo — no download required

Trains on scikit-learn's built-in 8×8 handwritten digit dataset (1,797 images, 10 classes):

```bash
python cnn_from_scratch.py --dataset digits
```

### 2. Real MNIST (28×28)

```bash
python cnn_from_scratch.py --dataset mnist --max_train 5000 --max_test 1000 --epochs 30
```

| Flag | Purpose |
|---|---|
| `--dataset {digits,mnist}` | Which dataset/network configuration to use. |
| `--max_train`, `--max_test` | How many rows to load. Since this is pure NumPy on CPU (no batched GPU convolution), runtime is capped by limiting dataset size rather than by limiting model size. |
| `--epochs` | Number of training epochs. |
| `--train_csv`, `--test_csv` | Paths to the MNIST CSV files, if not stored in the project folder. |

Each run prints per-epoch training loss/accuracy and test accuracy, then saves the trained weights to `cnn_weights.npz` and finally prints one example single-image prediction.

### 3. Loading pretrained weights for inference

The repository includes `cnn_weights.npz`, the weights from a completed MNIST run, so you can skip training entirely and go straight to inference:

```python
from cnn_from_scratch import build_network_mnist

net = build_network_mnist()
net.load("cnn_weights.npz")

preds = net.predict(X)   # X: (N, 1, 28, 28), pixel values scaled to [0, 1]
```

### 4. Running inference on your own images

```python
from cnn_from_scratch import build_network_mnist, preprocess_image

net = build_network_mnist()
net.load("cnn_weights.npz")

x = preprocess_image("my_digit.png")   # resizes/normalizes to the network's expected input shape
pred = net.predict(x[None, ...])
```

## Results

| Dataset | Images (train / test) | Epochs | Final test accuracy | Best test accuracy |
|---|---|---|---|---|
| Built-in 8×8 digits | 1,527 / 270 | 60 | 84.0% | 84.0% |
| MNIST 28×28 | 5,000 / 1,000 | 30 | **90.8%** | **90.9%** |

All numbers were produced by training from scratch with pure NumPy on a CPU, on a subset of real MNIST, with no pretrained weights and no external deep learning framework. Note this is *not* competitive with state-of-the-art CNNs (which reach >99% on MNIST) — the point of this project is a transparent, correct, from-scratch implementation rather than raw accuracy, and accuracy is capped in part by training on a 5,000-image subset for reasonable CPU runtime.

## Use Cases / Who This Is For

This project is intended as an **educational reference implementation**, not a production model. It's a good fit if you want to:

- **Learn how backpropagation actually works** through convolution and pooling — by reading code that computes gradients explicitly, rather than by reading an autograd trace.
- **Study or teach a CNN course module** — the layer-by-layer structure (`Conv2D`, `ReLU`, `MaxPool2D`, `Flatten`, `Dense`, `Dropout`, `SoftmaxCrossEntropy`) maps directly onto the standard CNN curriculum, making it usable as lecture/lab material.
- **Debug your understanding of im2col-based convolution** — the `_im2col` helper and the `Conv2D` forward/backward pair are a self-contained, readable reference for this specific technique.
- **Prototype small custom image classifiers** without a heavyweight framework dependency, using `build_network_mnist()` plus `preprocess_image()` on your own small image dataset (keeping in mind the CPU-only, unvectorized-loop performance ceiling described below).
- **Verify your own from-scratch implementation** — the overfitting sanity-check methodology here can be reused as a general debugging pattern for any hand-written backprop code.

It is **not** intended for:
- Production image classification workloads (use PyTorch/TensorFlow with GPU acceleration instead).
- Large images or large datasets — see [Limitations](#limitations).
- Any application requiring state-of-the-art accuracy.

## Limitations

- **Pure NumPy, CPU-only.** There is no GPU acceleration, so this is far slower per-image than PyTorch or TensorFlow equivalents.
- **Partially unvectorized backward passes.** The `MaxPool2D` backward pass and part of the `Conv2D` backward pass (the patch scatter-add) use plain Python loops rather than fully vectorized NumPy operations. This is the primary performance bottleneck.
- **Trained on a subset of MNIST.** Training used 5,000 of the full 60,000 training images to keep runtime reasonable on a CPU; this caps achievable accuracy relative to training on the full set.
- **No convolution padding options beyond "valid."** Only valid (no-padding) convolution is implemented.

## Roadmap / Future Work

- Vectorize the `MaxPool2D` and `Conv2D` backward-pass loops for a meaningful speed improvement.
- Train on the full 60,000-image MNIST training set.
- Add data augmentation (small rotations, shifts, or noise) to improve generalization.
- Add "same" padding support to `Conv2D`.
- Add a numerical gradient-checking utility (finite differences vs. analytical gradients) as an automated correctness test, formalizing the sanity check described above.

## License

MIT

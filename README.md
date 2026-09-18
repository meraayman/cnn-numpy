# CNN From Scratch (NumPy Only)

A convolutional neural network built entirely from scratch using **NumPy** — no PyTorch, TensorFlow, or Keras. Every layer implements its own forward pass **and** backward pass by hand, including manual backpropagation through convolution, max-pooling, and dense layers.

This project was built to actually understand what's happening inside a CNN, rather than relying on an autograd engine to do it for you.

## Why this is different from most "from scratch" tutorials

Most educational NumPy CNN implementations stop at a toy example that never really converges well. Before trusting this one, I verified backprop was actually correct by checking that the network could **overfit a tiny 16-image batch to 100% accuracy** — if a network can't memorize a handful of examples, its gradients are wrong, no matter how good the code looks. Only after that sanity check did I trust it on the real dataset.

## Architecture

Layers implemented from scratch:
- `Conv2D` — 2D convolution using the im2col trick (patches reshaped into a matrix so convolution becomes one big matrix multiply)
- `ReLU`
- `MaxPool2D` — backward pass routes gradients to the exact pixel that won each pooling window (via tracked argmax positions)
- `Flatten`
- `Dense` (fully connected)
- `Dropout` — with a train/eval switch so it's disabled during inference
- `SoftmaxCrossEntropy` — numerically stable combined loss

Two network configurations are included:
- A small network for the built-in 8x8 digit demo (`build_network`)
- A larger network for real 28x28 images (`build_network_mnist`), which also supports color input via `in_channels=3`

Also included: `preprocess_image()` and `load_images_from_folder()`, which resize any image — any original size, color or grayscale — down to a fixed shape so it can be fed into the network. This removes the fixed-input-size limitation for custom datasets beyond MNIST.

## Dataset

- **Quick demo (no download):** scikit-learn's built-in 8x8 handwritten digit dataset (10 classes, 1,797 images)
- **Real MNIST:** [phoebetronic/mnist on GitHub](https://github.com/phoebetronic/mnist) — classic 28x28, 10-class handwritten digits in CSV format

## Setup

```bash
pip install numpy scikit-learn pillow
```

For MNIST mode, download `mnist_train.csv.zip` and `mnist_test.csv.zip` from the link above, unzip them, and place `mnist_train.csv` and `mnist_test.csv` in this project folder.

## Usage

Quick demo, no download required:
```bash
python cnn_from_scratch.py --dataset digits
```

Real MNIST:
```bash
python cnn_from_scratch.py --dataset mnist --max_train 5000 --max_test 1000 --epochs 30
```

Flags:
- `--max_train` / `--max_test` — how many rows to load (pure NumPy is CPU-only and slower per-image than GPU frameworks, so this caps runtime)
- `--epochs` — training length
- `--train_csv` / `--test_csv` — paths to the MNIST CSV files, if not in the same folder

## Results

| Dataset | Images (train/test) | Epochs | Final test accuracy | Best test accuracy |
|---|---|---|---|---|
| Built-in 8x8 digits | 1,527 / 270 | 60 | 84% | 84% |
| MNIST 28x28 | 5,000 / 1,000 | 30 | **90.8%** | **90.9%** |

Trained from scratch, pure NumPy, CPU only, on a subset of real MNIST — no pretrained weights, no external deep learning framework.

## Limitations

- Pure NumPy, CPU-only — no GPU acceleration, so it's far slower per-image than PyTorch/TensorFlow
- A few backward passes (`MaxPool2D`, part of `Conv2D`) use plain Python loops rather than fully vectorized operations, which is the main speed bottleneck
- Trained on a subset of MNIST rather than the full 60,000 images, for reasonable runtime

## What I'd improve next

- Vectorize the `MaxPool2D` and `Conv2D` backward-pass loops for a real speed boost
- Train on the full 60,000-image MNIST set instead of a 5,000-image subset
- Add data augmentation (small rotations/shifts) to improve generalization

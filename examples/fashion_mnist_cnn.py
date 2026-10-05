"""Train the mnist_cnn.py baseline CNN on Fashion-MNIST.

DATA 621 Week 5, Part B. The model is copied unchanged from
examples/mnist_cnn.py. Changes from that script:
  - data: get_fashion_mnist instead of get_mnist (same interface)
  - images standardized with the training-set mean and std
  - fixed random seed (weight init, dropout masks, shuffling)
  - 10 epochs, Adam lr 0.001, batch size 64
  - per epoch: training loss, training accuracy and test accuracy
  - at the end: 10x10 confusion matrix, per-class accuracy and the pair of
    classes confused most often
"""

import time

import numpy as np

from dptiny import (
    Variable,
    is_available,
    is_gpu,
    no_grad,
    softmax_cross_entropy,
    test_mode,
    to_cpu,
    to_gpu,
    use_gpu,
)
from dptiny.backend import xp
from dptiny.data import DataLoader
from dptiny.data.fashion_mnist import CLASSES, get_fashion_mnist
from dptiny.nn import (
    Conv2d,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    ReLU,
    Sequential,
)
from dptiny.optim import Adam

SEED = 0
BATCH_SIZE = 64
MAX_EPOCH = 10
LR = 0.001

if is_available():
    use_gpu()
    print("GPU enabled for training.")
else:
    print("GPU not available; training on CPU.")

# All of DPTiny's randomness (weight init, dropout, DataLoader shuffling)
# goes through xp.random, so seeding it makes the run reproducible.
np.random.seed(SEED)
xp.random.seed(SEED)

print("Loading Fashion-MNIST dataset...")
X_train, X_test, y_train, y_test = get_fashion_mnist(flatten=False)

# Standardize with statistics of the TRAINING set only (no test leakage).
mean = float(X_train.mean())
std = float(X_train.std())
X_train = ((X_train - mean) / std).astype(np.float32)
X_test = ((X_test - mean) / std).astype(np.float32)
print(f"Training-set pixel mean = {mean:.4f}, std = {std:.4f}")
print(f"After standardizing: train mean = {X_train.mean():.4f}, "
      f"std = {X_train.std():.4f}")

if is_gpu():
    X_train = to_gpu(X_train)
    X_test = to_gpu(X_test)
    y_train = to_gpu(y_train)
    y_test = to_gpu(y_test)

# Baseline model from examples/mnist_cnn.py, unchanged.
model = Sequential(
    Conv2d(1, 16, 3, pad=1),
    ReLU(),
    MaxPool2d(2),
    Conv2d(16, 32, 3, pad=1),
    ReLU(),
    MaxPool2d(2),
    Flatten(),
    Linear(32 * 7 * 7, 128),
    ReLU(),
    Dropout(0.3),
    Linear(128, 10),
)
if is_gpu():
    model.to_gpu()

data_loader = DataLoader((X_train, y_train), BATCH_SIZE)
test_loader = DataLoader((X_test, y_test), BATCH_SIZE, shuffle=False)
optimizer = Adam(model, lr=LR)


def predict(loader):
    """Return (predictions, labels) for every sample, as NumPy arrays."""
    preds, labels = [], []
    with test_mode(), no_grad():
        for x, t in loader:
            y = model(Variable(x))
            preds.append(to_cpu(y.data.argmax(axis=1)))
            labels.append(to_cpu(t))
    return np.concatenate(preds), np.concatenate(labels)


start_time = time.time()
for epoch in range(MAX_EPOCH):
    model.train()
    sum_loss, sum_correct, n_seen = 0.0, 0, 0
    for x, t in data_loader:
        y = model(Variable(x))
        loss = softmax_cross_entropy(y, t)

        model.cleargrads()
        loss.backward()
        optimizer.update()

        # Training metrics, averaged over every sample seen this epoch
        # (computed in training mode, i.e. with dropout active).
        sum_loss += float(loss.data) * len(t)
        sum_correct += int((y.data.argmax(axis=1) == t).sum())
        n_seen += len(t)

    train_loss = sum_loss / n_seen
    train_acc = sum_correct / n_seen
    test_pred, test_true = predict(test_loader)
    test_acc = float((test_pred == test_true).mean())
    print(
        f"epoch {epoch + 1:2d}/{MAX_EPOCH} | "
        f"train loss {train_loss:.4f} | train acc {train_acc:.4f} | "
        f"test acc {test_acc:.4f} | {time.time() - start_time:.1f}s"
    )

# ----- Final evaluation on the test set -----
test_pred, test_true = predict(test_loader)
n_classes = len(CLASSES)

# confusion[i, j] = number of test images of true class i predicted as j
confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
np.add.at(confusion, (test_true, test_pred), 1)

print(f"\nTraining completed in {time.time() - start_time:.1f} seconds")
print(f"Final test accuracy: {np.trace(confusion) / confusion.sum():.4f}")

short = [c.split("/")[0][:7] for c in CLASSES]  # short column labels
print("\nConfusion matrix (rows = true class, columns = predicted class):")
print(" " * 13 + "".join(f"{s:>8}" for s in short))
for i in range(n_classes):
    print(f"{CLASSES[i]:>12} " + "".join(f"{v:8d}" for v in confusion[i]))

print("\nPer-class accuracy (correct / total of that class):")
per_class = confusion.diagonal() / confusion.sum(axis=1)
for i, acc in enumerate(per_class):
    print(f"  {i} {CLASSES[i]:<12} {acc:.4f}  "
          f"({confusion[i, i]}/{confusion[i].sum()})")

# Most-confused pair: add both directions, i->j and j->i, for each pair.
off = confusion.copy()
np.fill_diagonal(off, 0)
pair_counts = off + off.T
i, j = np.unravel_index(np.argmax(np.triu(pair_counts, 1)), off.shape)
print(
    f"\nMost confused pair: {CLASSES[i]} <-> {CLASSES[j]}: "
    f"{pair_counts[i, j]} test images "
    f"({off[i, j]} {CLASSES[i]} predicted as {CLASSES[j]}, "
    f"{off[j, i]} {CLASSES[j]} predicted as {CLASSES[i]})"
)

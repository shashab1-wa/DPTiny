"""Fashion-MNIST dataset loader.

Drop-in replacement for :func:`dptiny.data.mnist.get_mnist`: same arguments,
same return values, shapes and dtypes. The four gzipped IDX files are
downloaded from the project's GitHub repository on first use, cached, and
parsed with a small NumPy IDX reader (no scikit-learn, torchvision or
TensorFlow).
"""

import gzip
import hashlib
import os
import tempfile
import urllib.request
from typing import Optional, Tuple

import numpy as np

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "dptiny")
_SUBDIR = "fashion-mnist"

_BASE_URL = (
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/"
    "master/data/fashion/"
)

# File name -> MD5 checksum published in the Fashion-MNIST README.
_FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}

# Class names in label order (label 0 .. 9).
CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

# IDX data-type code for unsigned bytes (the only type Fashion-MNIST uses).
_IDX_UINT8 = 0x08


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: str, md5: Optional[str] = None) -> None:
    """Download ``url`` to ``dest`` atomically.

    The data is streamed into a temporary file in the same directory and only
    renamed to ``dest`` once the download has finished (and, if given, the
    checksum matches). If anything goes wrong - network error, Ctrl-C, bad
    checksum - the temporary file is deleted, so the cache never contains a
    half-written file.
    """
    directory = os.path.dirname(dest)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url) as resp:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
        if md5 is not None and _md5(tmp_path) != md5:
            raise ValueError(f"Checksum mismatch for {url}")
        os.replace(tmp_path, dest)  # atomic rename on the same filesystem
    except BaseException:  # includes KeyboardInterrupt
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _ensure_files(cache_dir: str) -> None:
    """Download any of the four files that are not cached yet."""
    os.makedirs(cache_dir, exist_ok=True)
    for name, md5 in _FILES.items():
        path = os.path.join(cache_dir, name)
        if not os.path.exists(path):
            print(f"Downloading {name} ...")
            _download(_BASE_URL + name, path, md5)


def read_idx(path: str) -> np.ndarray:
    """Parse an IDX file (optionally gzipped) into a ``uint8`` NumPy array.

    IDX layout (all integers big-endian):
        bytes 0-1   zero
        byte  2     data-type code (0x08 = unsigned byte)
        byte  3     number of dimensions ``d``
        4*d bytes   size of each dimension (uint32)
        rest        raw data, row-major

    Raises:
        ValueError: if the file is not a uint8 IDX file, or its data size
            does not match the dimensions in its header.
    """
    opener = gzip.open if path.endswith(".gz") else open
    try:
        with opener(path, "rb") as f:
            raw = f.read()
    except (OSError, EOFError) as e:  # bad/truncated gzip stream
        raise ValueError(f"{path}: cannot read file ({e})") from e

    if len(raw) < 4:
        raise ValueError(f"{path}: too short to be an IDX file")
    if raw[0] != 0 or raw[1] != 0:
        raise ValueError(f"{path}: bad magic number, not an IDX file")
    if raw[2] != _IDX_UINT8:
        raise ValueError(
            f"{path}: data-type code 0x{raw[2]:02x}, expected 0x08 (uint8)"
        )

    ndim = raw[3]
    header_size = 4 + 4 * ndim
    if ndim == 0 or len(raw) < header_size:
        raise ValueError(f"{path}: truncated or invalid IDX header")
    dims = tuple(int(d) for d in np.frombuffer(raw, ">u4", ndim, offset=4))

    expected = int(np.prod(dims, dtype=np.int64))
    actual = len(raw) - header_size
    if actual != expected:
        raise ValueError(
            f"{path}: header says {dims} = {expected} bytes of data, "
            f"file has {actual}"
        )
    return np.frombuffer(raw, np.uint8, offset=header_size).reshape(dims)


def get_fashion_mnist(
    normalize: bool = True,
    flatten: bool = True,
    data_home: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the Fashion-MNIST dataset.

    Same interface as :func:`dptiny.data.mnist.get_mnist`.

    Args:
        normalize: If ``True``, scale pixel values to ``[0, 1]``.
        flatten: If ``True``, return images as ``(N, 784)`` vectors,
            otherwise as ``(N, 1, 28, 28)``.
        data_home: Cache directory (default ``~/.cache/dptiny``). Files are
            stored in its ``fashion-mnist`` subfolder.

    Returns:
        ``(X_train, X_test, y_train, y_test)``: float32 images (60,000 and
        10,000 of them) and int32 labels in ``0..9``.
    """
    base = data_home if data_home is not None else _CACHE_DIR
    cache_dir = os.path.join(base, _SUBDIR)
    _ensure_files(cache_dir)

    def load(name):
        return read_idx(os.path.join(cache_dir, name))

    X_train = load("train-images-idx3-ubyte.gz")
    y_train = load("train-labels-idx1-ubyte.gz")
    X_test = load("t10k-images-idx3-ubyte.gz")
    y_test = load("t10k-labels-idx1-ubyte.gz")

    for X, y, n in ((X_train, y_train, 60000), (X_test, y_test, 10000)):
        if X.shape != (n, 28, 28) or y.shape != (n,):
            raise ValueError(
                f"Unexpected shapes {X.shape} / {y.shape}, expected "
                f"({n}, 28, 28) / ({n},)"
            )

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    if normalize:
        X_train /= 255.0
        X_test /= 255.0

    shape = (-1, 784) if flatten else (-1, 1, 28, 28)
    return (
        X_train.reshape(shape),
        X_test.reshape(shape),
        y_train,
        y_test,
    )

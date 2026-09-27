from typing import Callable, Optional
import numpy as np

from k3_node.data import Data, InMemoryDataset


class Digits(InMemoryDataset):
    r"""The handwritten digits of scikit-learn (1,797 images of 8x8 pixels) as graphs.

    A small, offline stand-in for PyG's ``MNISTSuperpixels``. Every non-blank pixel becomes a node
    with its intensity (in ``[0, 1]``) as the only feature ``x`` and its 2D position ``pos``, scaled
    to ``[0, 28)`` like MNIST. Neighboring pixels (including diagonals) are connected in both
    directions. ``y`` holds the digit (10 classes).

    Args:
        train (bool): If ``True``, loads the first 1,500 images, otherwise the remaining 297.
            (default: ``True``)
        transform (callable, optional): A function applied to each graph when it is accessed.

    Example:
        ```python
        from k3_node.datasets import Digits

        dataset = Digits(train=True)
        print(len(dataset), dataset.num_classes)  # 1500 10
        ```
    """

    num_train = 1500

    def __init__(self, train: bool = True, transform: Optional[Callable] = None):
        super().__init__(None, transform)
        from sklearn.datasets import load_digits

        images, labels = load_digits(return_X_y=True)
        images = images.reshape(-1, 8, 8) / 16.0
        split = slice(0, self.num_train) if train else slice(self.num_train, None)
        self.data, self.slices = self.collate(
            [self._to_graph(img, y) for img, y in zip(images[split], labels[split])]
        )

    @staticmethod
    def _to_graph(image, label):
        rows, cols = np.nonzero(image)
        index = -np.ones((8, 8), dtype=np.int64)
        index[rows, cols] = np.arange(len(rows))
        src, dst = [], []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                r, c = rows + dr, cols + dc
                ok = (r >= 0) & (r < 8) & (c >= 0) & (c < 8)
                nbr = index[r[ok], c[ok]]
                keep = nbr >= 0
                src.append(np.nonzero(ok)[0][keep])
                dst.append(nbr[keep])
        return Data(
            x=image[rows, cols][:, None].astype(np.float32),
            pos=(np.stack([cols, rows], axis=1) * 3.5 + 1.75).astype(np.float32),
            edge_index=np.stack([np.concatenate(src), np.concatenate(dst)]).astype(np.int64),
            y=np.array([label], dtype=np.int64),
        )

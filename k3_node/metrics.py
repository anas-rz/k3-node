"""Metrics for graph learning tasks."""
import numpy as np
from typing import Optional

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="k3_node")
class F1Score(keras.metrics.F1Score):
    r"""F1 score that also accepts logits, e.g. for multi-label node classification.

    Same as :class:`keras.metrics.F1Score`, plus ``from_logits``: when :obj:`True`, a sigmoid is
    applied to the predictions first, so models can output logits (as used with
    ``BinaryCrossentropy(from_logits=True)``).

    Example:
        ```python
        import numpy as np
        from k3_node.metrics import F1Score

        f1 = F1Score(average="micro", from_logits=True)
        y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype="float32")
        logits = np.array([[2.0, -1.0, 0.5], [-3.0, 1.5, 0.2]], dtype="float32")
        f1.update_state(y_true, logits)
        print(round(float(f1.result()), 2))  # 0.86
        ```
    """

    def __init__(self, average=None, threshold=0.5, from_logits=False, name="f1_score", dtype=None):
        super().__init__(average=average, threshold=threshold, name=name, dtype=dtype)
        self.from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.from_logits:
            y_pred = ops.sigmoid(y_pred)
        return super().update_state(y_true, y_pred, sample_weight=sample_weight)

    def get_config(self):
        return {**super().get_config(), "from_logits": self.from_logits}


def precision_recall_at_k(user_emb, item_emb, train_edge_index, test_edge_index, k: int = 20,
                          num_users: Optional[int] = None, batch_size: int = 8192):
    r"""Top-``k`` recommendation quality, as in PyG's LightGCN example.

    Every user's items are ranked by the dot product of the embeddings, excluding the items the
    user interacted with during training. Returns the precision@k and recall@k averaged over the
    users with at least one test interaction.

    Args:
        user_emb: User embeddings ``[num_users, dim]``.
        item_emb: Item embeddings ``[num_items, dim]``.
        train_edge_index: Training interactions ``(user, item)``; item ids may be offset by
            ``num_users`` (as in a homogeneous user-item graph).
        test_edge_index: Test interactions ``(user, item)``, with the same numbering.
        k (int): The number of recommendations per user. (default: ``20``)
        num_users (int, optional): The item-id offset. (default: ``len(user_emb)``)
        batch_size (int): Users scored at once. (default: ``8192``)

    Example:
        ```python
        import numpy as np
        from k3_node.metrics import precision_recall_at_k

        users, items = np.eye(2, 4, dtype="float32"), np.eye(3, 4, dtype="float32")  # toy embeddings
        train = np.array([[0], [2]])  # user 0 already has item 2 (id 2 + num_users = 4 in the graph)
        test = np.array([[0, 1], [0 + 2, 1 + 2]])  # user 0 likes item 0, user 1 likes item 1
        print(precision_recall_at_k(users, items, train + [[0], [2]], test, k=1))  # (1.0, 1.0)
        ```
    """
    from keras import ops as _ops

    user_emb, item_emb = (np.asarray(_ops.convert_to_numpy(e)) for e in (user_emb, item_emb))
    num_users = len(user_emb) if num_users is None else num_users
    train = np.asarray(_ops.convert_to_numpy(train_edge_index)).astype(np.int64)
    test = np.asarray(_ops.convert_to_numpy(test_edge_index)).astype(np.int64)
    train = train[:, train[0] < num_users]
    precision = recall = examples = 0.0
    for start in range(0, len(user_emb), batch_size):
        end = min(start + batch_size, len(user_emb))
        logits = user_emb[start:end] @ item_emb.T
        m = (train[0] >= start) & (train[0] < end)
        logits[train[0, m] - start, train[1, m] - num_users] = -np.inf  # skip already known items
        truth = np.zeros_like(logits, dtype=bool)
        m = (test[0] >= start) & (test[0] < end)
        truth[test[0, m] - start, test[1, m] - num_users] = True
        count = truth.sum(axis=1)
        top = np.argpartition(-logits, k - 1, axis=1)[:, :k]
        hits = np.take_along_axis(truth, top, axis=1).sum(axis=1)
        precision += float((hits / k)[count > 0].sum())
        recall += float((hits / np.maximum(count, 1e-6))[count > 0].sum())
        examples += int((count > 0).sum())
    return precision / examples, recall / examples

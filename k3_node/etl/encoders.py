"""Feature preprocessors and encoders for tabular data."""

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np


class NumericalEncoder:
    r"""Encodes numerical tabular features with scaling and missing value imputation.

    Args:
        strategy: Scaling method (``"standard"``, ``"minmax"``, ``"log1p"``, or ``"none"``).
            (default: ``"standard"``)
        impute_strategy: How to fill missing values / NaNs (``"mean"``, ``"median"``,
            ``"zero"``, or float constant). (default: ``"mean"``)
    """

    def __init__(self, strategy: str = "standard", impute_strategy: Union[str, float] = "mean"):
        self.strategy = strategy.lower()
        self.impute_strategy = impute_strategy
        self.mean_: Optional[float] = None
        self.std_: Optional[float] = None
        self.min_: Optional[float] = None
        self.max_: Optional[float] = None
        self.fill_value_: float = 0.0

    def fit(self, values: Sequence[Any]):
        arr = np.asarray(values, dtype=np.float64).flatten()
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            valid = np.array([0.0])

        if self.impute_strategy == "mean":
            self.fill_value_ = float(np.mean(valid))
        elif self.impute_strategy == "median":
            self.fill_value_ = float(np.median(valid))
        elif self.impute_strategy == "min":
            self.fill_value_ = float(np.min(valid))
        elif self.impute_strategy == "zero":
            self.fill_value_ = 0.0
        elif isinstance(self.impute_strategy, (int, float)):
            self.fill_value_ = float(self.impute_strategy)
        else:
            self.fill_value_ = float(np.mean(valid))

        self.mean_ = float(np.mean(valid))
        self.std_ = float(np.std(valid))
        if self.std_ < 1e-8:
            self.std_ = 1.0

        self.min_ = float(np.min(valid))
        self.max_ = float(np.max(valid))
        if abs(self.max_ - self.min_) < 1e-8:
            self.max_ = self.min_ + 1.0

        return self

    def transform(self, values: Sequence[Any]) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32).flatten()
        arr = np.nan_to_num(arr, nan=self.fill_value_)

        if self.strategy == "standard":
            mean = self.mean_ if self.mean_ is not None else 0.0
            std = self.std_ if self.std_ is not None else 1.0
            res = (arr - mean) / std
        elif self.strategy == "minmax":
            vmin = self.min_ if self.min_ is not None else 0.0
            vmax = self.max_ if self.max_ is not None else 1.0
            res = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
        elif self.strategy == "log1p":
            res = np.log1p(np.maximum(arr, 0.0))
        elif self.strategy == "none":
            res = arr
        else:
            raise ValueError(f"Unknown numerical strategy '{self.strategy}'.")

        return res.reshape(-1, 1)

    def fit_transform(self, values: Sequence[Any]) -> np.ndarray:
        return self.fit(values).transform(values)


class CategoricalEncoder:
    r"""Encodes categorical strings or integer values into one-hot or ordinal representations.

    Args:
        strategy: Encoding method (``"onehot"``, ``"ordinal"``, or ``"hash"``).
            (default: ``"onehot"``)
        handle_unknown: How to handle unseen categories during transform (``"ignore"``,
            ``"error"``, or ``"use_encoded_value"``). (default: ``"ignore"``)
        unknown_value: Numerical value assigned to unseen categories when using ordinal encoding.
            (default: ``-1``)
        hash_dim: Output dimension when using ``"hash"`` strategy. (default: ``16``)
    """

    def __init__(
        self,
        strategy: str = "onehot",
        handle_unknown: str = "ignore",
        unknown_value: int = -1,
        hash_dim: int = 16,
    ):
        self.strategy = strategy.lower()
        self.handle_unknown = handle_unknown.lower()
        self.unknown_value = unknown_value
        self.hash_dim = hash_dim
        self.vocab_: Dict[Any, int] = {}
        self.inv_vocab_: List[Any] = []

    def fit(self, values: Sequence[Any]):
        arr = [str(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else "__MISSING__" for v in values]
        unique_cats = sorted(list(set(arr)))
        self.vocab_ = {cat: idx for idx, cat in enumerate(unique_cats)}
        self.inv_vocab_ = unique_cats
        return self

    def transform(self, values: Sequence[Any]) -> np.ndarray:
        arr = [str(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else "__MISSING__" for v in values]
        num_samples = len(arr)

        if self.strategy == "onehot":
            num_classes = len(self.vocab_)
            if num_classes == 0:
                return np.zeros((num_samples, 1), dtype=np.float32)
            out = np.zeros((num_samples, num_classes), dtype=np.float32)
            for i, val in enumerate(arr):
                if val in self.vocab_:
                    out[i, self.vocab_[val]] = 1.0
                elif self.handle_unknown == "error":
                    raise ValueError(f"Encountered unknown category: '{val}'")
            return out

        elif self.strategy == "ordinal":
            out = np.zeros((num_samples, 1), dtype=np.int64)
            for i, val in enumerate(arr):
                if val in self.vocab_:
                    out[i, 0] = self.vocab_[val]
                elif self.handle_unknown == "error":
                    raise ValueError(f"Encountered unknown category: '{val}'")
                else:
                    out[i, 0] = self.unknown_value
            return out

        elif self.strategy == "hash":
            out = np.zeros((num_samples, self.hash_dim), dtype=np.float32)
            for i, val in enumerate(arr):
                h = abs(hash(val)) % self.hash_dim
                out[i, h] = 1.0
            return out

        else:
            raise ValueError(f"Unknown categorical strategy '{self.strategy}'.")

    def fit_transform(self, values: Sequence[Any]) -> np.ndarray:
        return self.fit(values).transform(values)


class TabularEncoder:
    r"""Column-wise encoder aggregating multiple numerical and categorical column encoders.

    Args:
        column_encoders: Optional dictionary mapping column names to :class:`NumericalEncoder`
            or :class:`CategoricalEncoder` instances.
        default_numerical_strategy: Strategy used for detected numerical columns without an explicit encoder.
            (default: ``"standard"``)
        default_categorical_strategy: Strategy used for detected categorical columns without an explicit encoder.
            (default: ``"onehot"``)
    """

    def __init__(
        self,
        column_encoders: Optional[Dict[str, Union[NumericalEncoder, CategoricalEncoder]]] = None,
        default_numerical_strategy: str = "standard",
        default_categorical_strategy: str = "onehot",
    ):
        self.column_encoders = column_encoders or {}
        self.default_numerical_strategy = default_numerical_strategy
        self.default_categorical_strategy = default_categorical_strategy
        self.fitted_encoders_: Dict[str, Union[NumericalEncoder, CategoricalEncoder]] = {}
        self.column_order_: List[str] = []

    def fit(self, df_or_dict: Any, columns: Optional[List[str]] = None):
        columns = columns or _get_column_names(df_or_dict)
        self.column_order_ = list(columns)
        self.fitted_encoders_ = {}

        for col in self.column_order_:
            vals = _get_column_values(df_or_dict, col)
            if col in self.column_encoders:
                enc = self.column_encoders[col]
            else:
                if _is_numerical_series(vals):
                    enc = NumericalEncoder(strategy=self.default_numerical_strategy)
                else:
                    enc = CategoricalEncoder(strategy=self.default_categorical_strategy)
            enc.fit(vals)
            self.fitted_encoders_[col] = enc

        return self

    def transform(self, df_or_dict: Any) -> np.ndarray:
        parts = []
        for col in self.column_order_:
            vals = _get_column_values(df_or_dict, col)
            enc = self.fitted_encoders_[col]
            encoded = enc.transform(vals)
            parts.append(encoded)

        if not parts:
            num_rows = len(_get_column_values(df_or_dict, list(self.column_encoders.keys())[0])) if self.column_encoders else 0
            return np.zeros((num_rows, 0), dtype=np.float32)

        return np.concatenate(parts, axis=1).astype(np.float32)

    def fit_transform(self, df_or_dict: Any, columns: Optional[List[str]] = None) -> np.ndarray:
        return self.fit(df_or_dict, columns=columns).transform(df_or_dict)


def _get_column_names(df_or_dict: Any) -> List[str]:
    if hasattr(df_or_dict, "columns"):
        return list(df_or_dict.columns)
    elif isinstance(df_or_dict, dict):
        return list(df_or_dict.keys())
    raise TypeError(f"Expected pandas DataFrame or dictionary of columns, got {type(df_or_dict)}")


def _get_column_values(df_or_dict: Any, col: str) -> List[Any]:
    if hasattr(df_or_dict, "__getitem__"):
        series = df_or_dict[col]
        if hasattr(series, "tolist"):
            return series.tolist()
        elif hasattr(series, "to_numpy"):
            return series.to_numpy().tolist()
        elif isinstance(series, np.ndarray):
            return series.tolist()
        return list(series)
    raise TypeError(f"Cannot extract column '{col}' from object of type {type(df_or_dict)}")


def _is_numerical_series(vals: Sequence[Any]) -> bool:
    count_num = 0
    total = 0
    for v in vals:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        total += 1
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            count_num += 1
    return total > 0 and (count_num / total) > 0.8

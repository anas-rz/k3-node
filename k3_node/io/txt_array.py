from typing import List, Optional, Union
import numpy as np
from keras import ops


def parse_txt_array(
    src: List[str],
    sep: Optional[str] = None,
    start: int = 0,
    end: Optional[int] = None,
    dtype: Optional[str] = None,
):
    """Parses a list of string rows into a tensor."""
    lines = [line.strip() for line in src if line.strip()]
    if len(lines) == 0:
        return ops.zeros((0,), dtype=dtype or "float32")

    split_lines = []
    is_float = False
    for line in lines:
        parts = line.split(sep)[start:end]
        row = []
        for x in parts:
            x_str = x.strip()
            if not x_str:
                continue
            if "." in x_str or "e" in x_str.lower():
                is_float = True
                row.append(float(x_str))
            else:
                try:
                    row.append(int(x_str))
                except ValueError:
                    is_float = True
                    row.append(float(x_str))
        split_lines.append(row)

    if dtype is None:
        dtype = "float32" if is_float else "int64"

    arr = np.array(split_lines, dtype=dtype)
    if arr.ndim > 1 and arr.shape[1] == 1:
        arr = np.squeeze(arr, axis=1)
    return ops.convert_to_tensor(arr, dtype=dtype)


def read_txt_array(
    path: str,
    sep: Optional[str] = None,
    start: int = 0,
    end: Optional[int] = None,
    dtype: Optional[str] = None,
):
    """Reads a text array from a file and returns a tensor."""
    with open(path, "r", encoding="utf-8") as f:
        src = f.read().split("\n")
    return parse_txt_array(src, sep, start, end, dtype)


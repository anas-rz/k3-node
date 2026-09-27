"""Transforms that use node positions (``data.pos``), meshes (``data.face``) or images.

They run on the host with NumPy and return tensors of the same kind as their inputs, so they work
with every Keras backend.
"""
import math
import random
import re
from typing import Any, List, Optional, Sequence, Tuple, Union

import keras
import numpy as np

from k3_node.data import Data, HeteroData
from k3_node.transforms.base_transform import BaseTransform, functional_transform
from k3_node.transforms.utils import match_tensor, to_numpy, to_undirected


def _np(x):
    return None if x is None else to_numpy(x)


def _out(value, like, dtype=None):
    """Returns the NumPy result ``value`` as the same kind of tensor (and dtype) as ``like``."""
    if dtype is None:
        dtype = keras.backend.standardize_dtype(like.dtype) if like is not None else None
    value = np.asarray(value)
    if dtype is not None:
        value = value.astype(dtype)
    if isinstance(like, np.ndarray):
        return value
    return match_tensor(value, like, dtype=dtype)


def _is_tensor(x) -> bool:
    return hasattr(x, "shape") and hasattr(x, "dtype") and len(x.shape) > 0


def _normalize(x, axis=-1):
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


def _set_edge_attr(data, new, cat: bool, like):
    """Stores ``new`` edge features, appended to the existing ones if ``cat``."""
    pseudo = data.edge_attr
    if pseudo is not None and cat:
        p = _np(pseudo)
        p = p.reshape(-1, 1) if p.ndim == 1 else p
        data.edge_attr = _out(np.concatenate([p, new.astype(p.dtype)], axis=-1), pseudo)
    else:
        data.edge_attr = _out(new, like)


def _get_angle(v1, v2):
    cross = np.cross(v1, v2)
    cross_norm = np.linalg.norm(cross, axis=-1) if cross.ndim > 1 else np.abs(cross)
    return np.arctan2(cross_norm, (v1 * v2).sum(-1))


def _point_pair_features(pos_i, pos_j, norm_i, norm_j):
    pseudo = pos_j - pos_i
    return np.stack([
        np.linalg.norm(pseudo, axis=-1),
        _get_angle(norm_i, pseudo),
        _get_angle(norm_j, pseudo),
        _get_angle(norm_i, norm_j),
    ], axis=-1)


def _scatter_sum(src, index, dim_size):
    out = np.zeros((dim_size,) + src.shape[1:], dtype=src.dtype)
    np.add.at(out, index, src)
    return out


def _scatter_mean(src, index, dim_size):
    count = np.bincount(index, minlength=dim_size).reshape((-1,) + (1,) * (src.ndim - 1))
    return _scatter_sum(src, index, dim_size) / np.maximum(count, 1)


def _scatter_max(src, index, dim_size):
    out = np.full((dim_size,) + src.shape[1:], -np.inf, dtype=src.dtype)
    np.maximum.at(out, index, src)
    return np.where(np.isinf(out), 0, out)


def _edges(data):
    edge_index = _np(data.edge_index)
    return edge_index[0], edge_index[1], _np(data.pos)


@functional_transform('distance')
class Distance(BaseTransform):
    r"""Saves the relative Euclidean distance of linked nodes in edge attributes
    (functional name: :obj:`distance`).

    Args:
        norm (bool, optional): If set to :obj:`False`, output will not be
            normalized to the specified interval. (default: :obj:`True`)
        max_value (float, optional): If set and :obj:`norm=True`, normalization
            will be performed based on this value instead of maximum distance.
            (default: :obj:`None`)
        cat (bool, optional): If set to :obj:`False`, all existing edge
            attributes will be replaced. (default: :obj:`True`)
        interval (tuple, optional): A tuple specifying the lower and upper
            bound for normalization. (default: :obj:`(0.0, 1.0)`)
    """
    def __init__(
        self,
        norm: bool = True,
        max_value: Optional[float] = None,
        cat: bool = True,
        interval: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self.norm = norm
        self.max = max_value
        self.cat = cat
        self.interval = interval

    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        assert data.edge_index is not None
        row, col, pos = _edges(data)

        dist = np.linalg.norm(pos[col] - pos[row], axis=-1).reshape(-1, 1)

        if self.norm and dist.size > 0:
            max_val = dist.max() if self.max is None else self.max
            if max_val > 0:
                dist = dist / max_val
            length = self.interval[1] - self.interval[0]
            dist = length * dist + self.interval[0]

        _set_edge_attr(data, dist, self.cat, data.pos)
        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(norm={self.norm}, '
                f'max_value={self.max})')


@functional_transform('cartesian')
class Cartesian(BaseTransform):
    r"""Saves the relative Cartesian coordinates of linked nodes in edge
    attributes (functional name: :obj:`cartesian`).

    Args:
        norm (bool, optional): If set to :obj:`False`, the output will not be
            normalized to the specified interval. (default: :obj:`True`)
        max_value (float, optional): If set and :obj:`norm=True`, normalization
            will be performed based on this value instead of maximum coordinate.
            (default: :obj:`None`)
        cat (bool, optional): If set to :obj:`False`, all existing edge
            attributes will be replaced. (default: :obj:`True`)
        interval (tuple, optional): A tuple specifying the lower and upper
            bound for normalization. (default: :obj:`(0.0, 1.0)`)
    """
    def __init__(
        self,
        norm: bool = True,
        max_value: Optional[float] = None,
        cat: bool = True,
        interval: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self.norm = norm
        self.max = max_value
        self.cat = cat
        self.interval = interval

    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        assert data.edge_index is not None
        row, col, pos = _edges(data)

        cart = pos[row] - pos[col]
        cart = cart.reshape(-1, 1) if cart.ndim == 1 else cart

        if self.norm and cart.size > 0:
            max_val = np.abs(cart).max() if self.max is None else self.max
            length = self.interval[1] - self.interval[0]
            center = (self.interval[0] + self.interval[1]) / 2
            if max_val > 0:
                cart = length * (cart / (2 * max_val)) + center
            else:
                cart = cart + center

        _set_edge_attr(data, cart, self.cat, data.pos)
        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(norm={self.norm}, '
                f'max_value={self.max})')


@functional_transform('local_cartesian')
class LocalCartesian(BaseTransform):
    r"""Saves relative Cartesian coordinates, normalized per neighborhood to an interval
    (functional name: :obj:`local_cartesian`)."""
    def __init__(
        self,
        norm: bool = True,
        cat: bool = True,
        interval: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self.norm = norm
        self.cat = cat
        self.interval = interval

    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        assert data.edge_index is not None
        row, col, pos = _edges(data)

        cart = pos[row] - pos[col]
        cart = cart.reshape(-1, 1) if cart.ndim == 1 else cart

        if self.norm and cart.size > 0:
            max_value = _scatter_max(np.abs(cart), col, pos.shape[0]).max(axis=-1, keepdims=True)
            length = self.interval[1] - self.interval[0]
            center = (self.interval[0] + self.interval[1]) / 2
            denom = 2 * max_value[col]
            denom = np.where(denom == 0, 1.0, denom)
            cart = length * cart / denom + center

        _set_edge_attr(data, cart, self.cat, data.pos)
        return data


@functional_transform('polar')
class Polar(BaseTransform):
    r"""Saves the polar coordinates of linked nodes in its edge attributes
    (functional name: :obj:`polar`)."""
    def __init__(
        self,
        norm: bool = True,
        max_value: Optional[float] = None,
        cat: bool = True,
    ) -> None:
        self.norm = norm
        self.max = max_value
        self.cat = cat

    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        assert data.edge_index is not None
        row, col, pos = _edges(data)
        assert pos.ndim == 2 and pos.shape[1] == 2

        cart = pos[col] - pos[row]
        rho = np.linalg.norm(cart, axis=-1).reshape(-1, 1)
        theta = np.arctan2(cart[..., 1], cart[..., 0]).reshape(-1, 1)
        theta = theta + (theta < 0) * (2 * math.pi)

        if self.norm:
            max_val = rho.max() if self.max is None else self.max
            if max_val > 0:
                rho = rho / max_val
            theta = theta / (2 * math.pi)

        _set_edge_attr(data, np.concatenate([rho, theta], axis=-1), self.cat, data.pos)
        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(norm={self.norm}, '
                f'max_value={self.max})')


@functional_transform('spherical')
class Spherical(BaseTransform):
    r"""Saves the spherical coordinates of linked nodes in its edge attributes
    (functional name: :obj:`spherical`)."""
    def __init__(
        self,
        norm: bool = True,
        max_value: Optional[float] = None,
        cat: bool = True,
    ) -> None:
        self.norm = norm
        self.max = max_value
        self.cat = cat

    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        assert data.edge_index is not None
        row, col, pos = _edges(data)
        assert pos.ndim == 2 and pos.shape[1] == 3

        cart = pos[col] - pos[row]
        rho = np.linalg.norm(cart, axis=-1).reshape(-1, 1)
        theta = np.arctan2(cart[..., 1], cart[..., 0]).reshape(-1, 1)
        theta = theta + (theta < 0) * (2 * math.pi)
        denom = np.where(rho[:, 0] == 0, 1.0, rho[:, 0])
        phi = np.arccos(np.clip(cart[..., 2] / denom, -1.0, 1.0)).reshape(-1, 1)

        if self.norm:
            max_val = rho.max() if self.max is None else self.max
            if max_val > 0:
                rho = rho / max_val
            theta = theta / (2 * math.pi)
            phi = phi / math.pi

        _set_edge_attr(data, np.concatenate([rho, theta, phi], axis=-1), self.cat, data.pos)
        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(norm={self.norm}, '
                f'max_value={self.max})')


@functional_transform('point_pair_features')
class PointPairFeatures(BaseTransform):
    r"""Computes rotation-invariant Point Pair Features in edge attributes
    (functional name: :obj:`point_pair_features`)."""
    def __init__(self, cat: bool = True) -> None:
        self.cat = cat

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        assert data.pos is not None and data.norm is not None
        row, col, pos = _edges(data)
        norm = _np(data.norm)
        assert pos.shape[-1] == 3 and pos.shape == norm.shape

        ppf = _point_pair_features(pos[row], pos[col], norm[row], norm[col])
        _set_edge_attr(data, ppf, self.cat, data.pos)
        return data


@functional_transform('center')
class Center(BaseTransform):
    r"""Centers node positions :obj:`data.pos` around the origin
    (functional name: :obj:`center`).
    """
    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.node_stores:
            if hasattr(store, 'pos') and store.pos is not None:
                pos = _np(store.pos)
                store.pos = _out(pos - pos.mean(axis=-2, keepdims=True), store.pos)
        return data


@functional_transform('normalize_rotation')
class NormalizeRotation(BaseTransform):
    r"""Rotates all points according to the eigenvectors of the point cloud
    (functional name: :obj:`normalize_rotation`)."""
    def __init__(self, max_points: int = -1, sort: bool = False) -> None:
        self.max_points = max_points
        self.sort = sort

    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        pos_all = _np(data.pos)
        pos = pos_all

        if self.max_points > 0 and pos.shape[0] > self.max_points:
            pos = pos[np.random.permutation(pos.shape[0])[:self.max_points]]

        pos = pos - pos.mean(axis=0, keepdims=True)
        e, v = np.linalg.eigh(pos.T @ pos)
        if self.sort:
            v = v[:, np.argsort(-e)]

        data.pos = _out(pos_all @ v, data.pos)
        if 'normal' in data and data.normal is not None:
            data.normal = _out(_normalize(_np(data.normal) @ v), data.normal)
        return data


@functional_transform('normalize_scale')
class NormalizeScale(BaseTransform):
    r"""Centers and normalizes node positions to the interval :math:`(-1, 1)`
    (functional name: :obj:`normalize_scale`)."""
    def __init__(self) -> None:
        self.center = Center()

    def forward(self, data: Data) -> Data:
        data = self.center(data)
        assert data.pos is not None
        pos = _np(data.pos)
        data.pos = _out(pos * ((1.0 / np.abs(pos).max()) * 0.999999), data.pos)
        return data


@functional_transform('random_jitter')
class RandomJitter(BaseTransform):
    r"""Translates node positions by randomly sampled translation values within a given interval
    (functional name: :obj:`random_jitter`)."""
    def __init__(
        self,
        translate: Union[float, int, Sequence[Union[float, int]]],
    ) -> None:
        self.translate = translate

    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        pos = _np(data.pos)
        dim = pos.shape[-1]

        if isinstance(self.translate, (int, float)):
            translate = [self.translate] * dim
        else:
            assert len(self.translate) == dim
            translate = self.translate

        bound = np.abs(np.asarray(translate, dtype=np.float64))
        data.pos = _out(pos + np.random.uniform(-bound, bound, size=pos.shape), data.pos)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.translate})'


@functional_transform('random_flip')
class RandomFlip(BaseTransform):
    """Flips node positions along a given axis randomly with a given probability
    (functional name: :obj:`random_flip`)."""
    def __init__(self, axis: int, p: float = 0.5) -> None:
        self.axis = axis
        self.p = p

    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        if random.random() < self.p:
            pos = _np(data.pos).copy()
            pos[..., self.axis] = -pos[..., self.axis]
            data.pos = _out(pos, data.pos)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(axis={self.axis}, p={self.p})'


@functional_transform('linear_transformation')
class LinearTransformation(BaseTransform):
    r"""Transforms node positions :obj:`data.pos` with a square transformation matrix
    (functional name: :obj:`linear_transformation`)."""
    def __init__(self, matrix: Any):
        matrix = np.asarray(_np(matrix))
        assert matrix.ndim == 2, 'Transformation matrix should be two-dimensional.'
        assert matrix.shape[0] == matrix.shape[1], (
            f'Transformation matrix should be square (got {matrix.shape})')
        self.matrix = matrix.T

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.node_stores:
            if not hasattr(store, 'pos') or store.pos is None:
                continue
            pos = _np(store.pos)
            pos = pos.reshape(-1, 1) if pos.ndim == 1 else pos
            assert pos.shape[-1] == self.matrix.shape[-2], (
                'Node position matrix and transformation matrix have incompatible shape')
            store.pos = _out(pos @ self.matrix.astype(pos.dtype), store.pos)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(\n{self.matrix}\n)'


@functional_transform('random_scale')
class RandomScale(BaseTransform):
    r"""Scales node positions by a randomly sampled factor within a given interval
    (functional name: :obj:`random_scale`)."""
    def __init__(self, scales: Tuple[float, float]) -> None:
        assert isinstance(scales, (tuple, list)) and len(scales) == 2
        self.scales = scales

    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        data.pos = _out(_np(data.pos) * random.uniform(*self.scales), data.pos)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.scales})'


@functional_transform('random_rotate')
class RandomRotate(BaseTransform):
    r"""Rotates node positions around a specific axis by a randomly sampled angle
    (functional name: :obj:`random_rotate`)."""
    def __init__(
        self,
        degrees: Union[Tuple[float, float], float],
        axis: int = 0,
    ) -> None:
        if isinstance(degrees, (int, float)):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.axis = axis

    def forward(self, data: Data) -> Data:
        assert data.pos is not None

        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        if data.pos.shape[-1] == 2:
            matrix = [[cos, sin], [-sin, cos]]
        elif self.axis == 0:
            matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
        elif self.axis == 1:
            matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
        else:
            matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]

        return LinearTransformation(np.array(matrix, dtype=np.float32))(data)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.degrees}, '
                f'axis={self.axis})')


@functional_transform('random_shear')
class RandomShear(BaseTransform):
    r"""Shears node positions by randomly sampled factors within a given interval
    (functional name: :obj:`random_shear`)."""
    def __init__(self, shear: Union[float, int]) -> None:
        self.shear = abs(shear)

    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        dim = data.pos.shape[-1]
        matrix = np.random.uniform(-self.shear, self.shear, size=(dim, dim))
        np.fill_diagonal(matrix, 1)
        return LinearTransformation(matrix)(data)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.shear})'


@functional_transform('face_to_edge')
class FaceToEdge(BaseTransform):
    r"""Converts mesh faces of shape :obj:`[3, num_faces]` or :obj:`[4, num_faces]`
    to edge indices of shape :obj:`[2, num_edges]` (functional name: :obj:`face_to_edge`).
    """
    def __init__(self, remove_faces: bool = True) -> None:
        self.remove_faces = remove_faces

    def forward(self, data: Data) -> Data:
        if hasattr(data, 'face') and data.face is not None:
            face = _np(data.face)

            if face.shape[0] not in [3, 4]:
                raise RuntimeError(f"Expected 'face' tensor with shape "
                                   f"[3, num_faces] or [4, num_faces] "
                                   f"(got {list(face.shape)})")

            if face.shape[0] == 3:
                edge_index = np.concatenate([face[:2], face[1:], face[::2]], axis=1)
            else:
                edge_index = np.concatenate(
                    [face[:2], face[1:3], face[2:4], face[::2], face[1::2], face[::3]], axis=1)

            edge_index = _np(to_undirected(edge_index, num_nodes=data.num_nodes))
            data.edge_index = _out(edge_index, data.face)
            if self.remove_faces:
                data.face = None

        return data


@functional_transform('sample_points')
class SamplePoints(BaseTransform):
    r"""Uniformly samples a fixed number of points on the mesh surfaces
    (functional name: :obj:`sample_points`)."""
    def __init__(
        self,
        num: int,
        remove_faces: bool = True,
        include_normals: bool = False,
    ) -> None:
        self.num = num
        self.remove_faces = remove_faces
        self.include_normals = include_normals

    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        assert data.face is not None
        like = data.pos
        pos, face = _np(data.pos).astype(np.float64), _np(data.face)
        assert pos.shape[1] == 3 and face.shape[0] == 3

        pos_max = np.abs(pos).max()
        pos = pos / pos_max

        area = np.linalg.norm(np.cross(pos[face[1]] - pos[face[0]], pos[face[2]] - pos[face[0]]), axis=1) / 2
        sample = np.random.choice(face.shape[1], self.num, replace=True, p=area / area.sum())
        face = face[:, sample]

        frac = np.random.rand(self.num, 2)
        mask = frac.sum(axis=-1) > 1
        frac[mask] = 1 - frac[mask]

        vec1 = pos[face[1]] - pos[face[0]]
        vec2 = pos[face[2]] - pos[face[0]]

        if self.include_normals:
            data.normal = _out(_normalize(np.cross(vec1, vec2)), like)

        pos_sampled = pos[face[0]] + frac[:, :1] * vec1 + frac[:, 1:] * vec2
        data.pos = _out(pos_sampled * pos_max, like)

        if self.remove_faces:
            data.face = None

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num})'


@functional_transform('fixed_points')
class FixedPoints(BaseTransform):
    r"""Samples a fixed number of points and features from a point cloud
    (functional name: :obj:`fixed_points`)."""
    def __init__(
        self,
        num: int,
        replace: bool = True,
        allow_duplicates: bool = False,
    ) -> None:
        self.num = num
        self.replace = replace
        self.allow_duplicates = allow_duplicates

    def forward(self, data: Data) -> Data:
        num_nodes = data.num_nodes
        assert num_nodes is not None

        if self.replace:
            choice = np.random.choice(num_nodes, self.num, replace=True)
        elif not self.allow_duplicates:
            choice = np.random.permutation(num_nodes)[:self.num]
        else:
            choice = np.concatenate([
                np.random.permutation(num_nodes)
                for _ in range(math.ceil(self.num / num_nodes))
            ])[:self.num]

        for key, value in list(data.items()):
            if key == 'num_nodes':
                data.num_nodes = len(choice)
            elif bool(re.search('edge', key)):
                continue
            elif _is_tensor(value) and value.shape[0] == num_nodes and value.shape[0] != 1:
                data[key] = _out(_np(value)[choice], value)

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num}, replace={self.replace})'


@functional_transform('generate_mesh_normals')
class GenerateMeshNormals(BaseTransform):
    r"""Generate normal vectors for each mesh node based on neighboring faces
    (functional name: :obj:`generate_mesh_normals`)."""
    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        assert data.face is not None
        pos, face = _np(data.pos), _np(data.face)

        face_norm = _normalize(np.cross(pos[face[1]] - pos[face[0]], pos[face[2]] - pos[face[0]]))
        norm = _scatter_sum(np.tile(face_norm, (3, 1)), face.reshape(-1), pos.shape[0])
        data.norm = _out(_normalize(norm), data.pos)
        return data


@functional_transform('delaunay')
class Delaunay(BaseTransform):
    r"""Computes the delaunay triangulation of a set of points
    (functional name: :obj:`delaunay`)."""
    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        num_points = data.pos.shape[0]

        if num_points < 2:
            data.edge_index = _out(np.zeros((2, 0)), None, dtype="int64")
        elif num_points == 2:
            data.edge_index = _out(np.array([[0, 1], [1, 0]]), None, dtype="int64")
        elif num_points == 3:
            data.face = _out(np.array([[0], [1], [2]]), None, dtype="int64")
        else:
            try:
                import scipy.spatial
                tri = scipy.spatial.Delaunay(_np(data.pos), qhull_options='QJ')
            except Exception as e:
                raise RuntimeError(f"Delaunay triangulation failed: {e}")
            data.face = _out(tri.simplices.T, None, dtype="int64")

        return data


@functional_transform('to_slic')
class ToSLIC(BaseTransform):
    r"""Converts an image of shape ``[channels, height, width]`` to a superpixel graph using SLIC
    (functional name: :obj:`to_slic`)."""
    def __init__(
        self,
        add_seg: bool = False,
        add_img: bool = False,
        **kwargs: Any,
    ) -> None:
        self.add_seg = add_seg
        self.add_img = add_img
        self.kwargs = kwargs

    def forward(self, img: Any) -> Data:
        try:
            from skimage.segmentation import slic
        except ImportError:
            raise ImportError("ToSLIC requires scikit-image to be installed.")

        img_t = np.transpose(_np(img), (1, 2, 0))
        h, w, c = img_t.shape

        seg = slic(img_t.astype(np.float64), start_label=0, **self.kwargs)
        num_segments = int(seg.max()) + 1
        x = _scatter_mean(img_t.reshape(h * w, c), seg.reshape(-1), num_segments)

        pos_y, pos_x = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
        pos = _scatter_mean(np.stack([pos_x.reshape(-1), pos_y.reshape(-1)], axis=-1), seg.reshape(-1), num_segments)

        data = Data(x=_out(x, img), pos=_out(pos, img, dtype="float32"))
        if self.add_seg:
            data.seg = _out(seg.reshape(1, h, w), None, dtype="int64")
        if self.add_img:
            data.img = _out(np.transpose(img_t, (2, 0, 1)).reshape(1, c, h, w), img)
        return data


@functional_transform('grid_sampling')
class GridSampling(BaseTransform):
    r"""Clusters points into fixed-sized voxels (functional name: :obj:`grid_sampling`)."""
    def __init__(
        self,
        size: Union[float, List[float], Any],
        start: Optional[Union[float, List[float], Any]] = None,
        end: Optional[Union[float, List[float], Any]] = None,
    ) -> None:
        self.size = size
        self.start = start
        self.end = end

    def forward(self, data: Data) -> Data:
        num_nodes = data.num_nodes
        assert data.pos is not None

        pos = _np(data.pos)
        if pos.ndim == 1:
            pos = pos[:, None]
        dim = pos.shape[-1]

        size = np.broadcast_to(np.asarray(_np(self.size), dtype=pos.dtype), (dim,))
        start = pos.min(axis=0) if self.start is None else np.broadcast_to(
            np.asarray(_np(self.start), dtype=pos.dtype), (dim,))

        coord = np.floor((pos - start) / size).astype(np.int64)
        if getattr(data, 'batch', None) is not None:
            coord = np.concatenate([coord, _np(data.batch).reshape(-1, 1)], axis=-1)

        unique, c = np.unique(coord, axis=0, return_inverse=True)
        c = c.reshape(-1)
        num_clusters = unique.shape[0]
        perm = np.zeros(num_clusters, dtype=np.int64)
        perm[c] = np.arange(len(c))

        for key, item in list(data.items()):
            if bool(re.search('edge', key)):
                raise ValueError(f"'{self.__class__.__name__}' does not support coarsening of edges")

            if _is_tensor(item) and item.shape[0] == num_nodes:
                value = _np(item)
                if key == 'y':
                    one_hot = np.eye(int(value.max()) + 1, dtype=np.float32)[value]
                    data[key] = _out(_scatter_sum(one_hot, c, num_clusters).argmax(axis=-1), item)
                elif key == 'batch':
                    data[key] = _out(value[perm], item)
                else:
                    data[key] = _out(_scatter_mean(value, c, num_clusters), item)

        data.num_nodes = num_clusters
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size})'


class RandomTranslate(RandomJitter):
    r"""Alias for RandomJitter."""
    pass

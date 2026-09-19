import math
import random
import re
from itertools import repeat
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from k3_node.data import Data, HeteroData
from k3_node.transforms.base_transform import BaseTransform, functional_transform
from k3_node.transforms.utils import match_tensor, to_numpy, to_undirected


def _get_angle(v1: Tensor, v2: Tensor) -> Tensor:
    cross = torch.cross(v1, v2, dim=-1)
    if cross.dim() > 1:
        cross_norm = cross.norm(p=2, dim=-1)
    else:
        cross_norm = cross.abs()
    dot = (v1 * v2).sum(dim=-1)
    return torch.atan2(cross_norm, dot)


def _point_pair_features(pos_i: Tensor, pos_j: Tensor, norm_i: Tensor, norm_j: Tensor) -> Tensor:
    pseudo = pos_j - pos_i
    return torch.stack([
        pseudo.norm(p=2, dim=-1),
        _get_angle(norm_i, pseudo),
        _get_angle(norm_j, pseudo),
        _get_angle(norm_i, norm_j),
    ], dim=-1)


def _scatter_max(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    out = src.new_zeros((dim_size, src.size(-1)))
    # index: [E], src: [E, D]
    for i in range(dim_size):
        mask = (index == i)
        if mask.any():
            out[i] = src[mask].max(dim=0)[0]
    return out


def _scatter_mean(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    out = src.new_zeros((dim_size,) + src.shape[1:])
    counts = src.new_zeros((dim_size,) + (1,) * (src.dim() - 1))
    for i in range(dim_size):
        mask = (index == i)
        cnt = mask.sum().item()
        if cnt > 0:
            out[i] = src[mask].sum(dim=0) / cnt
            counts[i] = cnt
    return out


def _scatter_sum(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    out = src.new_zeros((dim_size,) + src.shape[1:])
    for i in range(dim_size):
        mask = (index == i)
        if mask.any():
            out[i] = src[mask].sum(dim=0)
    return out


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
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        dist = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)

        if self.norm and dist.numel() > 0:
            max_val = dist.max() if self.max is None else self.max
            if max_val > 0:
                dist = dist / max_val
            length = self.interval[1] - self.interval[0]
            dist = length * dist + self.interval[0]

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, dist.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = dist

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
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        cart = pos[row] - pos[col]
        cart = cart.view(-1, 1) if cart.dim() == 1 else cart

        if self.norm and cart.numel() > 0:
            max_val = cart.abs().max() if self.max is None else self.max
            length = self.interval[1] - self.interval[0]
            if max_val > 0:
                cart = length * (cart / (2 * max_val)) + (self.interval[0] + self.interval[1]) / 2
            else:
                cart = cart + (self.interval[0] + self.interval[1]) / 2

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, cart.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = cart

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(norm={self.norm}, '
                f'max_value={self.max})')


@functional_transform('local_cartesian')
class LocalCartesian(BaseTransform):
    r"""Saves relative Cartesian coordinates neighborhood-normalized to interval."""
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
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        cart = pos[row] - pos[col]
        cart = cart.view(-1, 1) if cart.dim() == 1 else cart

        if self.norm and cart.numel() > 0:
            try:
                from torch_geometric.utils import scatter
                max_value = scatter(cart.abs(), col, 0, pos.size(0), reduce='max')
            except Exception:
                max_value = _scatter_max(cart.abs(), col, pos.size(0))
            max_value = max_value.max(dim=-1, keepdim=True)[0]

            length = self.interval[1] - self.interval[0]
            center = (self.interval[0] + self.interval[1]) / 2
            denom = 2 * max_value[col]
            denom = torch.where(denom == 0, torch.ones_like(denom), denom)
            cart = length * cart / denom + center

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, cart.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = cart

        return data


@functional_transform('polar')
class Polar(BaseTransform):
    r"""Saves the polar coordinates of linked nodes in its edge attributes."""
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
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr
        assert pos.dim() == 2 and pos.size(1) == 2

        cart = pos[col] - pos[row]
        rho = torch.norm(cart, p=2, dim=-1).view(-1, 1)

        theta = torch.atan2(cart[..., 1], cart[..., 0]).view(-1, 1)
        theta = theta + (theta < 0).type_as(theta) * (2 * math.pi)

        if self.norm:
            max_val = rho.max() if self.max is None else self.max
            if max_val > 0:
                rho = rho / max_val
            theta = theta / (2 * math.pi)

        polar = torch.cat([rho, theta], dim=-1)

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, polar.type_as(pos)], dim=-1)
        else:
            data.edge_attr = polar

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(norm={self.norm}, '
                f'max_value={self.max})')


@functional_transform('spherical')
class Spherical(BaseTransform):
    r"""Saves the spherical coordinates of linked nodes in its edge attributes."""
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
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr
        assert pos.dim() == 2 and pos.size(1) == 3

        cart = pos[col] - pos[row]
        rho = torch.norm(cart, p=2, dim=-1).view(-1, 1)

        theta = torch.atan2(cart[..., 1], cart[..., 0]).view(-1, 1)
        theta = theta + (theta < 0).type_as(theta) * (2 * math.pi)

        denom = rho.view(-1)
        denom_safe = torch.where(denom == 0, torch.ones_like(denom), denom)
        cos_phi = torch.clamp(cart[..., 2] / denom_safe, -1.0, 1.0)
        phi = torch.acos(cos_phi).view(-1, 1)

        if self.norm:
            max_val = rho.max() if self.max is None else self.max
            if max_val > 0:
                rho = rho / max_val
            theta = theta / (2 * math.pi)
            phi = phi / math.pi

        spher = torch.cat([rho, theta, phi], dim=-1)

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, spher.type_as(pos)], dim=-1)
        else:
            data.edge_attr = spher

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(norm={self.norm}, '
                f'max_value={self.max})')


@functional_transform('point_pair_features')
class PointPairFeatures(BaseTransform):
    r"""Computes rotation-invariant Point Pair Features in edge attributes."""
    def __init__(self, cat: bool = True) -> None:
        self.cat = cat

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        assert data.pos is not None and data.norm is not None
        assert data.pos.size(-1) == 3
        assert data.pos.size() == data.norm.size()

        row, col = data.edge_index
        pos, norm, pseudo = data.pos, data.norm, data.edge_attr

        ppf = _point_pair_features(pos[row], pos[col], norm[row], norm[col])

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, ppf.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = ppf

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
                store.pos = store.pos - store.pos.mean(dim=-2, keepdim=True)
        return data


@functional_transform('normalize_rotation')
class NormalizeRotation(BaseTransform):
    r"""Rotates all points according to the eigenvectors of the point cloud."""
    def __init__(self, max_points: int = -1, sort: bool = False) -> None:
        self.max_points = max_points
        self.sort = sort

    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        pos = data.pos

        if self.max_points > 0 and pos.size(0) > self.max_points:
            perm = torch.randperm(pos.size(0))
            pos = pos[perm[:self.max_points]]

        pos = pos - pos.mean(dim=0, keepdim=True)
        C = torch.matmul(pos.t(), pos)
        e, v = torch.linalg.eig(C)
        e, v = torch.view_as_real(e), v.real

        if self.sort:
            indices = e[:, 0].argsort(descending=True)
            v = v.t()[indices].t()

        data.pos = torch.matmul(data.pos, v)

        if 'normal' in data and data.normal is not None:
            data.normal = F.normalize(torch.matmul(data.normal, v), p=2, dim=-1)

        return data


@functional_transform('normalize_scale')
class NormalizeScale(BaseTransform):
    r"""Centers and normalizes node positions to the interval :math:`(-1, 1)`."""
    def __init__(self) -> None:
        self.center = Center()

    def forward(self, data: Data) -> Data:
        data = self.center(data)

        assert data.pos is not None
        scale = (1.0 / data.pos.abs().max()) * 0.999999
        data.pos = data.pos * scale

        return data


@functional_transform('random_jitter')
class RandomJitter(BaseTransform):
    r"""Translates node positions by randomly sampled translation values within a given interval."""
    def __init__(
        self,
        translate: Union[float, int, Sequence[Union[float, int]]],
    ) -> None:
        self.translate = translate

    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        num_nodes, dim = data.pos.size()

        if isinstance(self.translate, (int, float)):
            translate = list(repeat(self.translate, times=dim))
        else:
            assert len(self.translate) == dim
            translate = self.translate

        jitter = data.pos.new_empty(num_nodes, dim)
        for d in range(dim):
            jitter[:, d].uniform_(-abs(translate[d]), abs(translate[d]))

        data.pos = data.pos + jitter
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.translate})'


@functional_transform('random_flip')
class RandomFlip(BaseTransform):
    """Flips node positions along a given axis randomly with a given probability."""
    def __init__(self, axis: int, p: float = 0.5) -> None:
        self.axis = axis
        self.p = p

    def forward(self, data: Data) -> Data:
        assert data.pos is not None

        if random.random() < self.p:
            pos = data.pos.clone()
            pos[..., self.axis] = -pos[..., self.axis]
            data.pos = pos
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(axis={self.axis}, p={self.p})'


@functional_transform('linear_transformation')
class LinearTransformation(BaseTransform):
    r"""Transforms node positions :obj:`data.pos` with a square transformation matrix."""
    def __init__(self, matrix: Tensor):
        if not isinstance(matrix, Tensor):
            matrix = torch.tensor(matrix)
        assert matrix.dim() == 2, 'Transformation matrix should be two-dimensional.'
        assert matrix.size(0) == matrix.size(1), (
            f'Transformation matrix should be square (got {matrix.size()})')

        self.matrix = matrix.t()

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.node_stores:
            if not hasattr(store, 'pos') or store.pos is None:
                continue

            pos = store.pos.view(-1, 1) if store.pos.dim() == 1 else store.pos
            assert pos.size(-1) == self.matrix.size(-2), (
                'Node position matrix and transformation matrix have incompatible shape')
            store.pos = pos @ self.matrix.to(pos.device, pos.dtype)

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(\n{self.matrix.cpu().numpy()}\n)'


@functional_transform('random_scale')
class RandomScale(BaseTransform):
    r"""Scales node positions by a randomly sampled factor within a given interval."""
    def __init__(self, scales: Tuple[float, float]) -> None:
        assert isinstance(scales, (tuple, list)) and len(scales) == 2
        self.scales = scales

    def forward(self, data: Data) -> Data:
        assert data.pos is not None

        scale = random.uniform(*self.scales)
        data.pos = data.pos * scale
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.scales})'


@functional_transform('random_rotate')
class RandomRotate(BaseTransform):
    r"""Rotates node positions around a specific axis by a randomly sampled angle."""
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

        if data.pos.size(-1) == 2:
            matrix = [[cos, sin], [-sin, cos]]
        else:
            if self.axis == 0:
                matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
            elif self.axis == 1:
                matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
            else:
                matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]

        return LinearTransformation(torch.tensor(matrix, dtype=torch.float32))(data)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.degrees}, '
                f'axis={self.axis})')


@functional_transform('random_shear')
class RandomShear(BaseTransform):
    r"""Shears node positions by randomly sampled factors within a given interval."""
    def __init__(self, shear: Union[float, int]) -> None:
        self.shear = abs(shear)

    def forward(self, data: Data) -> Data:
        assert data.pos is not None

        dim = data.pos.size(-1)
        matrix = data.pos.new_empty(dim, dim).uniform_(-self.shear, self.shear)
        eye = torch.arange(dim, dtype=torch.long)
        matrix[eye, eye] = 1

        return LinearTransformation(matrix)(data)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.shear})'


@functional_transform('face_to_edge')
class FaceToEdge(BaseTransform):
    r"""Converts mesh faces of shape :obj:`[3, num_faces]` or :obj:`[4, num_faces]`
    to edge indices of shape :obj:`[2, num_edges]`.
    """
    def __init__(self, remove_faces: bool = True) -> None:
        self.remove_faces = remove_faces

    def forward(self, data: Data) -> Data:
        if hasattr(data, 'face') and data.face is not None:
            face = data.face

            if face.size(0) not in [3, 4]:
                raise RuntimeError(f"Expected 'face' tensor with shape "
                                   f"[3, num_faces] or [4, num_faces] "
                                   f"(got {list(face.size())})")

            if face.size(0) == 3:
                edge_index = torch.cat([
                    face[:2],
                    face[1:],
                    face[::2],
                ], dim=1)
            else:
                edge_index = torch.cat([
                    face[:2],
                    face[1:3],
                    face[2:4],
                    face[::2],
                    face[1::2],
                    face[::3],
                ], dim=1)

            edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)
            data.edge_index = edge_index
            if self.remove_faces:
                data.face = None

        return data


@functional_transform('sample_points')
class SamplePoints(BaseTransform):
    r"""Uniformly samples a fixed number of points on the mesh surfaces."""
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

        pos, face = data.pos, data.face
        assert pos.size(1) == 3 and face.size(0) == 3

        pos_max = pos.abs().max()
        pos = pos / pos_max

        area = (pos[face[1]] - pos[face[0]]).cross(
            pos[face[2]] - pos[face[0]],
            dim=1,
        )
        area = area.norm(p=2, dim=1).abs() / 2

        prob = area / area.sum()
        sample = torch.multinomial(prob, self.num, replacement=True)
        face = face[:, sample]

        frac = torch.rand(self.num, 2, device=pos.device)
        mask = frac.sum(dim=-1) > 1
        frac[mask] = 1 - frac[mask]

        vec1 = pos[face[1]] - pos[face[0]]
        vec2 = pos[face[2]] - pos[face[0]]

        if self.include_normals:
            data.normal = F.normalize(vec1.cross(vec2, dim=1), p=2, dim=-1)

        pos_sampled = pos[face[0]]
        pos_sampled += frac[:, :1] * vec1
        pos_sampled += frac[:, 1:] * vec2

        pos_sampled = pos_sampled * pos_max
        data.pos = pos_sampled

        if self.remove_faces:
            data.face = None

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num})'


@functional_transform('fixed_points')
class FixedPoints(BaseTransform):
    r"""Samples a fixed number of points and features from a point cloud."""
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
            choice = torch.from_numpy(
                np.random.choice(num_nodes, self.num, replace=True)).long()
        elif not self.allow_duplicates:
            choice = torch.randperm(num_nodes)[:self.num]
        else:
            choice = torch.cat([
                torch.randperm(num_nodes)
                for _ in range(math.ceil(self.num / num_nodes))
            ], dim=0)[:self.num]

        for key, value in list(data.items()):
            if key == 'num_nodes':
                data.num_nodes = choice.size(0)
            elif bool(re.search('edge', key)):
                continue
            elif isinstance(value, Tensor) and value.size(0) == num_nodes and value.size(0) != 1:
                data[key] = value[choice]

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num}, replace={self.replace})'


@functional_transform('generate_mesh_normals')
class GenerateMeshNormals(BaseTransform):
    r"""Generate normal vectors for each mesh node based on neighboring faces."""
    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        assert data.face is not None
        pos, face = data.pos, data.face

        vec1 = pos[face[1]] - pos[face[0]]
        vec2 = pos[face[2]] - pos[face[0]]
        face_norm = F.normalize(vec1.cross(vec2, dim=1), p=2, dim=-1)

        face_norm = face_norm.repeat(3, 1)
        idx = face.view(-1)

        try:
            from torch_geometric.utils import scatter
            norm = scatter(face_norm, idx, 0, pos.size(0), reduce='sum')
        except Exception:
            norm = _scatter_sum(face_norm, idx, pos.size(0))

        norm = F.normalize(norm, p=2, dim=-1)
        data.norm = norm
        return data


@functional_transform('delaunay')
class Delaunay(BaseTransform):
    r"""Computes the delaunay triangulation of a set of points."""
    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        device = data.pos.device

        if data.pos.size(0) < 2:
            data.edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        elif data.pos.size(0) == 2:
            data.edge_index = torch.tensor([[0, 1], [1, 0]], device=device)
        elif data.pos.size(0) == 3:
            data.face = torch.tensor([[0], [1], [2]], device=device)
        else:
            try:
                import scipy.spatial
                pos = data.pos.cpu().numpy()
                tri = scipy.spatial.Delaunay(pos, qhull_options='QJ')
                face = torch.from_numpy(tri.simplices)
                data.face = face.t().contiguous().to(device, torch.long)
            except Exception as e:
                raise RuntimeError(f"Delaunay triangulation failed: {e}")

        return data


@functional_transform('to_slic')
class ToSLIC(BaseTransform):
    r"""Converts an image to a superpixel representation using SLIC."""
    def __init__(
        self,
        add_seg: bool = False,
        add_img: bool = False,
        **kwargs: Any,
    ) -> None:
        self.add_seg = add_seg
        self.add_img = add_img
        self.kwargs = kwargs

    def forward(self, img: Tensor) -> Data:
        try:
            from skimage.segmentation import slic
        except ImportError:
            raise ImportError("ToSLIC requires scikit-image to be installed.")

        img_t = img.permute(1, 2, 0)
        h, w, c = img_t.size()

        seg = slic(img_t.to(torch.double).cpu().numpy(), start_label=0, **self.kwargs)
        seg = torch.from_numpy(seg).to(img.device)

        num_segments = int(seg.max().item()) + 1
        x = _scatter_mean(img_t.view(h * w, c), seg.view(h * w), num_segments)

        pos_y = torch.arange(h, dtype=torch.float, device=img.device)
        pos_y = pos_y.view(-1, 1).repeat(1, w).view(h * w)
        pos_x = torch.arange(w, dtype=torch.float, device=img.device)
        pos_x = pos_x.view(1, -1).repeat(h, 1).view(h * w)

        pos = torch.stack([pos_x, pos_y], dim=-1)
        pos = _scatter_mean(pos, seg.view(h * w), num_segments)

        data = Data(x=x, pos=pos)

        if self.add_seg:
            data.seg = seg.view(1, h, w)
        if self.add_img:
            data.img = img_t.permute(2, 0, 1).view(1, c, h, w)

        return data


@functional_transform('grid_sampling')
class GridSampling(BaseTransform):
    r"""Clusters points into fixed-sized voxels."""
    def __init__(
        self,
        size: Union[float, List[float], Tensor],
        start: Optional[Union[float, List[float], Tensor]] = None,
        end: Optional[Union[float, List[float], Tensor]] = None,
    ) -> None:
        self.size = size
        self.start = start
        self.end = end

    def forward(self, data: Data) -> Data:
        num_nodes = data.num_nodes
        assert data.pos is not None

        pos = data.pos
        if pos.dim() == 1:
            pos = pos.unsqueeze(-1)
        dim = pos.size(-1)

        size = self.size
        if not isinstance(size, Tensor):
            size = torch.tensor(size, dtype=pos.dtype, device=pos.device)
        if size.numel() == 1:
            size = size.repeat(dim)

        start = self.start
        if start is None:
            start = pos.min(dim=0)[0]
        elif not isinstance(start, Tensor):
            start = torch.tensor(start, dtype=pos.dtype, device=pos.device)
            if start.numel() == 1:
                start = start.repeat(dim)

        coord = torch.floor((pos - start) / size).long()
        if hasattr(data, 'batch') and data.batch is not None:
            coord = torch.cat([coord, data.batch.view(-1, 1)], dim=-1)

        unique, c = torch.unique(coord, dim=0, sorted=True, return_inverse=True)
        perm = torch.arange(c.size(0), dtype=c.dtype, device=c.device)
        perm = c.new_empty(unique.size(0)).scatter_(0, c, perm)

        num_clusters = unique.size(0)

        for key, item in list(data.items()):
            if bool(re.search('edge', key)):
                raise ValueError(f"'{self.__class__.__name__}' does not support coarsening of edges")

            if torch.is_tensor(item) and item.size(0) == num_nodes:
                if key == 'y':
                    one_hot_y = F.one_hot(item)
                    y_sum = _scatter_sum(one_hot_y.to(torch.float32), c, num_clusters)
                    data[key] = y_sum.argmax(dim=-1)
                elif key == 'batch':
                    data[key] = item[perm]
                else:
                    data[key] = _scatter_mean(item, c, num_clusters)

        data.num_nodes = num_clusters
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size})'


class RandomTranslate(RandomJitter):
    r"""Alias for RandomJitter."""
    pass

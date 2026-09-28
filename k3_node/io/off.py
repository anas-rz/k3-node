from typing import List

import numpy as np

from k3_node.data import Data


def parse_off(src: List[str]) -> Data:
    r"""Parses the lines of an OFF (Object File Format) mesh into ``Data(pos, face)``;
    quadrilaterals are split into two triangles."""
    if src[0] == 'OFF':
        src = src[1:]
    else:  # some files lack the line break after "OFF"
        src[0] = src[0][3:]
    num_nodes, num_faces = (int(item) for item in src[0].split()[:2])
    pos = np.array([[float(v) for v in line.split()[:3]] for line in src[1:1 + num_nodes]], dtype=np.float32)
    faces = [[int(v) for v in line.strip().split()] for line in src[1 + num_nodes:1 + num_nodes + num_faces]]
    tri = [f[1:4] for f in faces if f[0] == 3]
    for f in faces:
        if f[0] == 4:
            tri += [[f[1], f[2], f[3]], [f[1], f[3], f[4]]]
    face = np.array(tri, dtype=np.int64).reshape(-1, 3).T
    return Data(pos=pos, face=face)


def read_off(path: str) -> Data:
    r"""Reads an OFF (Object File Format) mesh file into ``Data(pos, face)``."""
    with open(path) as f:
        return parse_off(f.read().split('\n')[:-1])

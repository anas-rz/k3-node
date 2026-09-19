import keras.ops as ops
from k3_node.models.dimenet import (
    DimeNet,
    DimeNetPlusPlus,
    BesselBasisLayer,
    SphericalBasisLayer,
    triplets,
)


def test_triplets():
    edge_index = ops.convert_to_tensor([
        [0, 1, 1, 2, 0, 2],
        [1, 0, 2, 1, 2, 0],
    ])
    col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(edge_index, num_nodes=3)
    assert len(idx_i) > 0
    assert len(idx_kj) == len(idx_ji)


def test_bessel_basis_layer():
    bessel = BesselBasisLayer(num_radial=6, cutoff=5.0)
    dist = ops.convert_to_tensor([1.0, 2.0, 3.0])
    out = bessel(dist)
    assert out.shape == (3, 6)


def test_spherical_basis_layer():
    sbf = SphericalBasisLayer(num_spherical=3, num_radial=6, cutoff=5.0)
    dist = ops.convert_to_tensor([1.0, 2.0])
    angle = ops.convert_to_tensor([0.5, 1.2])
    idx_kj = ops.convert_to_tensor([0, 1])
    out = sbf(dist, angle, idx_kj)
    assert out.shape == (2, 18)


def test_dimenet():
    z = ops.convert_to_tensor([1, 6, 8, 1])
    pos = ops.convert_to_tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ], dtype="float32")

    model = DimeNet(
        hidden_channels=16,
        out_channels=1,
        num_blocks=2,
        num_bilinear=8,
        num_spherical=3,
        num_radial=6,
        cutoff=5.0,
    )
    out = model(z, pos)
    assert out.shape == (1,)

    # With batch
    batch = ops.convert_to_tensor([0, 0, 1, 1])
    out_b = model(z, pos, batch=batch)
    assert out_b.shape == (2, 1)


def test_dimenet_plus_plus():
    z = ops.convert_to_tensor([1, 6, 8, 1])
    pos = ops.convert_to_tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ], dtype="float32")

    model = DimeNetPlusPlus(
        hidden_channels=16,
        out_channels=1,
        num_blocks=2,
        int_emb_size=8,
        basis_emb_size=8,
        out_emb_channels=16,
        num_spherical=3,
        num_radial=6,
        cutoff=5.0,
    )
    out = model(z, pos)
    assert out.shape == (1,)


from k3_node.layers.conv.gat_conv import GATConv
from k3_node.layers.conv.sage_conv import SAGEConv
from k3_node.layers.conv.rgcn_conv import CuGraphRGCNConv


class CuGraphGATConv(GATConv):
    r"""An optimized / multi-backend compatible version of :class:`GATConv`.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import CuGraphGATConv

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        edge_index = np.random.randint(0, 10, size=(2, 30))  # 30 random edges

        layer = CuGraphGATConv(8, 16, heads=2)
        out = layer(x, edge_index)
        print(tuple(out.shape))  # (10, 32)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            bias=bias,
            **kwargs,
        )


class CuGraphSAGEConv(SAGEConv):
    r"""An optimized / multi-backend compatible version of :class:`SAGEConv`.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import CuGraphSAGEConv

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        edge_index = np.random.randint(0, 10, size=(2, 30))  # 30 random edges

        layer = CuGraphSAGEConv(8, 16)
        out = layer(x, edge_index)
        print(tuple(out.shape))  # (10, 16)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            aggr=aggr,
            normalize=normalize,
            root_weight=root_weight,
            project=project,
            bias=bias,
            **kwargs,
        )


__all__ = ["CuGraphGATConv", "CuGraphSAGEConv", "CuGraphRGCNConv"]

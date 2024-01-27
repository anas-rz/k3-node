from keras import ops

from k3_node.layers.aggr import Aggregation



class DeepSetsAggregation(Aggregation):
    def __init__(
        self,
        local_mlp=None,
        global_mlp=None,
    ):
        super().__init__()

        self.local_mlp = local_mlp
        self.global_mlp = global_mlp


    def call(self, x, index=None,  axis=-2):

        if self.local_mlp is not None:
            x = self.local_mlp(x) 

        x = self.reduce(x, index=index, axis=axis, reduce_fn=ops.segment_sum)

        if self.global_mlp is not None:
            x = self.global_mlp(x)  # Assuming batch handling within MLP

        return x
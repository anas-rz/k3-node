from keras import ops


def edge_index_to_adjacency_matrix(edge_index):
    edge_index = ops.convert_to_tensor(edge_index)
    num_nodes = ops.max(edge_index) + 1
    adjacency_matrix = ops.zeros((num_nodes, num_nodes), dtype="float32")

    indices = ops.transpose(edge_index, axes=[1, 0])
    updates = ops.ones(shape=(ops.shape(edge_index)[1],), dtype="float32")
    adjacency_matrix = ops.scatter_update(adjacency_matrix, indices, updates)

    return adjacency_matrix

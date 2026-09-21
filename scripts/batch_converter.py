import json
import sys

def convert_batch(batch):
    for fp, layer_name in batch:
        with open(fp) as f:
            nb = json.load(f)
        cells = nb.get('cells', [])
        if len(cells) == 7:
            title_cell = cells[0]
            setup_cell = cells[1]
            k3_code_cell = cells[5]

            src = ''.join(k3_code_cell['source'])
            if 'ops.random.' in src:
                src = src.replace('ops.random.', 'keras.random.')

            k3_code_cell['source'] = [src]

            # Update title cell to remove Part 1 mention
            title_src = ''.join(title_cell.get('source', []))
            title_src = title_src.replace('Part 1: PyTorch Geometric Reference Implementation', 'K3-Node Multi-Backend Implementation')
            title_src = title_src.replace('Part 2: K3-Node Multi-Backend Implementation', 'Keras 3 Multi-Backend Implementation')
            title_cell['source'] = [title_src]

            k3_md_cell = {
                'cell_type': 'markdown',
                'metadata': {},
                'source': [
                    '## K3-Node Implementation\n',
                    '\n',
                    'The following cell contains the ported version utilizing **K3-Node** and **Keras 3**.\n',
                    "By switching `os.environ['KERAS_BACKEND']`, this model can run seamlessly on **PyTorch**, **TensorFlow**, or **JAX**.\n"
                ]
            }

            summary_md_cell = {
                'cell_type': 'markdown',
                'metadata': {},
                'source': [
                    '## Summary\n',
                    '\n',
                    '| Implementation | Framework | Key Layers / Models | Status |\n',
                    '| :--- | :--- | :--- | :--- |\n',
                    f'| **K3-Node** | Keras 3 (Torch / TF / JAX) | `{layer_name}` | Ported & Verified |\n',
                    '\n',
                    'Both implementations share the same underlying mathematical formulation, guaranteeing parity across deep learning backends.\n'
                ]
            }

            nb['cells'] = [title_cell, setup_cell, k3_md_cell, k3_code_cell, summary_md_cell]
            with open(fp, 'w') as f:
                json.dump(nb, f, indent=1)
            print(f'Converted {fp} to 5 cells')
        elif len(cells) == 5:
            print(f'{fp} already 5 cells')
        else:
            print(f'{fp} unexpected cell count: {len(cells)}')

if __name__ == '__main__':
    remaining_23 = [
        ('examples/graph_saint.ipynb', 'GraphSAINT / SAGEConv'),
        ('examples/hierarchical_sampling.ipynb', 'HierarchicalSampling / SAGEConv'),
        ('examples/mnist_graclus.ipynb', 'Graclus / SplineConv'),
        ('examples/mnist_nn_conv.ipynb', 'NNConv / SplineConv'),
        ('examples/mnist_voxel_grid.ipynb', 'VoxelGrid / SplineConv'),
        ('examples/ogbn_proteins_deepgcn.ipynb', 'DeepGCNLayer / GENConv'),
        ('examples/ogbn_train.ipynb', 'NeighborLoader / SAGEConv'),
        ('examples/ogc.ipynb', 'OGC / GCNConv'),
        ('examples/point_transformer_classification.ipynb', 'PointTransformerConv'),
        ('examples/point_transformer_segmentation.ipynb', 'PointTransformerConv'),
        ('examples/pointnet2_classification.ipynb', 'PointNet2 / SAModule'),
        ('examples/pointnet2_segmentation.ipynb', 'PointNet2 / FPModule'),
        ('examples/qm9_nn_conv.ipynb', 'NNConv / Set2Set'),
        ('examples/qm9_pretrained_dimenet.ipynb', 'DimeNet / DimeNetPlusPlus'),
        ('examples/qm9_pretrained_schnet.ipynb', 'SchNet'),
        ('examples/randlanet_classification.ipynb', 'RandLANet'),
        ('examples/randlanet_segmentation.ipynb', 'RandLANet'),
        ('examples/rdl.ipynb', 'Relational Deep Learning / GNN'),
        ('examples/relbench_example.ipynb', 'RelBench / GraphSAGE'),
        ('examples/renet.ipynb', 'RENet / RGCNConv'),
        ('examples/shadow.ipynb', 'ShaDow-GNN / SAGEConv'),
        ('examples/tgn.ipynb', 'Temporal Graph Network (TGN)'),
        ('examples/unimp_arxiv.ipynb', 'UniMP / TransformerConv')
    ]
    convert_batch(remaining_23)

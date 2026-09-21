import os
import inspect
import keras
import k3_node.layers as layers

all_layers = []
for name, cls in inspect.getmembers(layers, inspect.isclass):
    if issubclass(cls, keras.layers.Layer):
        mod = cls.__module__
        all_layers.append((mod, name, cls))

all_layers.sort(key=lambda x: (x[0], x[1]))

categories = {
    'conv': 'Convolution Layers (`k3_node.layers.conv`)',
    'norm': 'Normalization Layers (`k3_node.layers.norm`)',
    'pool': 'Pooling Layers (`k3_node.layers.pool`)',
    'aggr': 'Aggregation Layers (`k3_node.layers.aggr`)',
    'dense': 'Dense / Linear Layers (`k3_node.layers.dense`)',
    'attention': 'Attention Layers (`k3_node.layers.attention`)',
    'kge': 'Knowledge Graph Embedding (`k3_node.layers.kge`)',
}

lines = [
    '# K3-Node Layers Multi-Backend Verification Checklist\n\n',
    '> Tracking layer-by-layer learning and training verification across PyTorch, TensorFlow, and JAX backends.\n\n',
    '## Overall Progress Summary\n\n',
    '| Category | Total | Tested | Passed | Failed | Base/Abstract/Vendor |\n',
    '| :--- | :--- | :--- | :--- | :--- | :--- |\n',
]

cat_counts = {}
for mod, name, cls in all_layers:
    cat = mod.split('.')[2] if len(mod.split('.')) > 2 else 'other'
    cat_counts[cat] = cat_counts.get(cat, 0) + 1

for cat, title in categories.items():
    cnt = cat_counts.get(cat, 0)
    lines.append(f'| {title} | {cnt} | 0 | 0 | 0 | 0 |\n')

lines.append('\n---\n\n')

for cat, title in categories.items():
    lines.append(f'## {title}\n\n')
    lines.append('| Layer Name | Module | PyTorch | TensorFlow | JAX | Learning Verified | Notes / Fixes |\n')
    lines.append('| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n')
    for mod, name, cls in all_layers:
        c = mod.split('.')[2] if len(mod.split('.')) > 2 else 'other'
        if c == cat:
            lines.append(f'| `{name}` | `{mod}` | [ ] Untested | [ ] Untested | [ ] Untested | [ ] | |\n')
    lines.append('\n')

with open('LAYERS_CHECKLIST.md', 'w') as f:
    f.writelines(lines)

print(f'Initialized LAYERS_CHECKLIST.md with {len(all_layers)} layers.')

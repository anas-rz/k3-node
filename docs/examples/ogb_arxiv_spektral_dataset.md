# Node Classification on OGBN-Arxiv with ARMAConv

**Author:** K3-Node Team<br>
**Backend:** TensorFlow<br>
**Dataset:** `ogbn-arxiv`<br>
**Description:** Large-scale node classification on the `ogbn-arxiv` citation benchmark using K3-Node's `ARMAConv` layer, Spektral graph preprocessing, and a custom TensorFlow training loop.

[:simple-googlecolab: **View in Colab**](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/tensorflow/ogb_arxiv_spektral_dataset.ipynb){ .md-button .md-button--primary } &nbsp; [:octicons-mark-github-16: **GitHub source**](https://github.com/anas-rz/k3-node/blob/main/examples/tensorflow/ogb_arxiv_spektral_dataset.ipynb){ .md-button }

---

## Setup & Installation

```bash
pip install spektral -qq
pip install --upgrade keras -qq
pip install ogb -qq
git clone https://github.com/anas-rz/k3-node.git
```

```python
import os, sys
os.environ['KERAS_BACKEND'] = 'tensorflow'
sys.path.append('/content/k3-node')
```

```python
import numpy as np
from ogb.nodeproppred import NodePropPredDataset
from keras.layers import BatchNormalization, Dropout, Input
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.models import Model
from keras.optimizers import Adam

from spektral.datasets.ogb import OGB
from spektral.transforms import AdjToSpTensor, GCNFilter

from pprint import pprint

from k3_node.layers import ARMAConv
```

## Load data

```python
dataset_name = "ogbn-arxiv"
ogb_dataset = NodePropPredDataset(dataset_name)
dataset = OGB(ogb_dataset, transforms=[GCNFilter(), AdjToSpTensor()])
graph = dataset[0]
x, adj, y = graph.x, graph.a, graph.y
```

## Parameters

```python
channels = 256  # Number of channels for GCN layers
dropout = 0.5  # Dropout rate for the features
learning_rate = 1e-2  # Learning rate
epochs = 200  # Number of training epochs
N = dataset.n_nodes  # Number of nodes in the graph
F = dataset.n_node_features  # Original size of node features
n_out = ogb_dataset.num_classes  # OGB labels are sparse indices
```

## Data splits

```python
idx = ogb_dataset.get_idx_split()
idx_tr, idx_va, idx_te = idx["train"], idx["valid"], idx["test"]
mask_tr = np.zeros(N, dtype=bool)
mask_va = np.zeros(N, dtype=bool)
mask_te = np.zeros(N, dtype=bool)
mask_tr[idx_tr] = True
mask_va[idx_va] = True
mask_te[idx_te] = True
masks = [mask_tr, mask_va, mask_te]
```

## Model definition

```python
x_in = Input(shape=(F,))
a_in = Input((N,), sparse=True)
x_1 = ARMAConv(channels, activation="relu")([x_in, a_in])
x_1 = BatchNormalization()(x_1)
x_1 = Dropout(dropout)(x_1)
x_2 = ARMAConv(channels, activation="relu")([x_1, a_in])
x_2 = BatchNormalization()(x_2)
x_2 = Dropout(dropout)(x_2)
x_3 = ARMAConv(n_out, activation="softmax")([x_2, a_in])
```

## Build model

```python
model = Model(inputs=[x_in, a_in], outputs=x_3)
optimizer = Adam(learning_rate=learning_rate)
loss_fn = SparseCategoricalCrossentropy()
acc_metric = SparseCategoricalAccuracy()
model.summary()
```

??? example "View Output"
    ```text
    [1mModel: "functional_5"[0m
    
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃[1m [0m[1mLayer (type)             [0m[1m [0m┃[1m [0m[1mOutput Shape          [0m[1m [0m┃[1m [0m[1m   Param #[0m[1m [0m┃[1m [0m[1mConnected to              [0m[1m [0m┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │ input_layer_2             │ ([38;5;45mNone[0m, [38;5;34m128[0m)            │          [38;5;34m0[0m │ -                          │
    │ ([38;5;33mInputLayer[0m)              │                        │            │                            │
    ├───────────────────────────┼────────────────────────┼────────────┼────────────────────────────┤
    │ input_layer_3             │ ([38;5;45mNone[0m, [38;5;34m169343[0m)         │          [38;5;34m0[0m │ -                          │
    │ ([38;5;33mInputLayer[0m)              │                        │            │                            │
    ├───────────────────────────┼────────────────────────┼────────────┼────────────────────────────┤
    │ arma_conv_3 ([38;5;33mARMAConv[0m)    │ ([38;5;45mNone[0m, [38;5;34m256[0m)            │     [38;5;34m65,792[0m │ input_layer_2[[38;5;34m0[0m][[38;5;34m0[0m],       │
    │                           │                        │            │ input_layer_3[[38;5;34m0[0m][[38;5;34m0[0m]        │
    ├───────────────────────────┼────────────────────────┼────────────┼────────────────────────────┤
    │ batch_normalization_2     │ ([38;5;45mNone[0m, [38;5;34m256[0m)            │      [38;5;34m1,024[0m │ arma_conv_3[[38;5;34m0[0m][[38;5;34m0[0m]          │
    │ ([38;5;33mBatchNormalization[0m)      │                        │            │                            │
    ├───────────────────────────┼────────────────────────┼────────────┼────────────────────────────┤
    │ dropout_6 ([38;5;33mDropout[0m)       │ ([38;5;45mNone[0m, [38;5;34m256[0m)            │          [38;5;34m0[0m │ batch_normalization_2[[38;5;34m0[0m][[38;5;34m…[0m │
    ├───────────────────────────┼────────────────────────┼────────────┼────────────────────────────┤
    │ arma_conv_4 ([38;5;33mARMAConv[0m)    │ ([38;5;45mNone[0m, [38;5;34m256[0m)            │    [38;5;34m131,328[0m │ dropout_6[[38;5;34m0[0m][[38;5;34m0[0m],           │
    │                           │                        │            │ input_layer_3[[38;5;34m0[0m][[38;5;34m0[0m]        │
    ├───────────────────────────┼────────────────────────┼────────────┼────────────────────────────┤
    │ batch_normalization_3     │ ([38;5;45mNone[0m, [38;5;34m256[0m)            │      [38;5;34m1,024[0m │ arma_conv_4[[38;5;34m0[0m][[38;5;34m0[0m]          │
    │ ([38;5;33mBatchNormalization[0m)      │                        │            │                            │
    ├───────────────────────────┼────────────────────────┼────────────┼────────────────────────────┤
    │ dropout_8 ([38;5;33mDropout[0m)       │ ([38;5;45mNone[0m, [38;5;34m256[0m)            │          [38;5;34m0[0m │ batch_normalization_3[[38;5;34m0[0m][[38;5;34m…[0m │
    ├───────────────────────────┼────────────────────────┼────────────┼────────────────────────────┤
    │ arma_conv_5 ([38;5;33mARMAConv[0m)    │ ([38;5;45mNone[0m, [38;5;34m40[0m)             │     [38;5;34m20,520[0m │ dropout_8[[38;5;34m0[0m][[38;5;34m0[0m],           │
    │                           │                        │            │ input_layer_3[[38;5;34m0[0m][[38;5;34m0[0m]        │
    └───────────────────────────┴────────────────────────┴────────────┴────────────────────────────┘
    
    [1m Total params: [0m[38;5;34m219,688[0m (858.16 KB)
    
    [1m Trainable params: [0m[38;5;34m218,664[0m (854.16 KB)
    
    [1m Non-trainable params: [0m[38;5;34m1,024[0m (4.00 KB)
    ```

```python
import tensorflow as tf
# Training function
@tf.function
def train(inputs, target, mask):
    acc_metric.reset_state()
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target[mask], predictions[mask]) + sum(model.losses)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    acc_metric.update_state(target[mask], predictions[mask])
    return loss, acc_metric.result()
```

```python
@tf.function
def evaluate(inputs, target, mask):
    acc_metric.reset_state()
    predictions = model(inputs, training=True)
    loss = loss_fn(target[mask], predictions[mask]) + sum(model.losses)
    acc_metric.update_state(target[mask], predictions[mask])
    return loss, acc_metric.result()
```

## Train model

```python
for i in range(1, 1 + epochs):
    tr_loss, tr_acc = train([x, adj], y, mask_tr)
    eval_loss, eval_acc = evaluate([x, adj], y, mask_va) # TODO Add more metrics
    pprint(f"EPOCH {i}: Training Loss {tr_loss.numpy()} - Training Accuracy {tr_acc}, Validation Loss: {eval_loss} - Validation Accuracy {eval_acc}")
test_loss, test_acc = evaluate([x, adj], y, mask_te)
pprint(f"Test Loss: {test_loss} Test Accuracy: {test_acc}")
```

??? example "View Output"
    ```text
    ('EPOCH 1: Training Loss 6.727264881134033 - Training Accuracy '
     '0.016780110076069832, Validation Loss: 3.410824775695801 - Validation '
     'Accuracy 0.3783012926578522')
    ('EPOCH 2: Training Loss 3.9297218322753906 - Training Accuracy '
     '0.3478848934173584, Validation Loss: 2.8849685192108154 - Validation '
     'Accuracy 0.4568609595298767')
    ('EPOCH 3: Training Loss 3.318859815597534 - Training Accuracy '
     '0.4080997705459595, Validation Loss: 2.4148788452148438 - Validation '
     'Accuracy 0.5107554197311401')
    ('EPOCH 4: Training Loss 2.793854236602783 - Training Accuracy '
     '0.4620468318462372, Validation Loss: 2.2219178676605225 - Validation '
     'Accuracy 0.521762490272522')
    ('EPOCH 5: Training Loss 2.5515968799591064 - Training Accuracy '
     '0.4713165760040283, Validation Loss: 2.018460512161255 - Validation Accuracy '
     '0.5272995829582214')
    ('EPOCH 6: Training Loss 2.343061685562134 - Training Accuracy '
     '0.4847428500652313, Validation Loss: 1.9317702054977417 - Validation '
     'Accuracy 0.5317292809486389')
    ('EPOCH 7: Training Loss 2.1674623489379883 - Training Accuracy '
     '0.49503523111343384, Validation Loss: 1.8545085191726685 - Validation '
     'Accuracy 0.5411255359649658')
    ('EPOCH 8: Training Loss 2.11173939704895 - Training Accuracy '
     '0.4981251657009125, Validation Loss: 1.7800989151000977 - Validation '
     'Accuracy 0.5560925006866455')
    ('EPOCH 9: Training Loss 1.9838796854019165 - Training Accuracy '
     '0.5201064348220825, Validation Loss: 1.7576695680618286 - Validation '
     'Accuracy 0.5606228113174438')
    ('EPOCH 10: Training Loss 1.9363586902618408 - Training Accuracy '
     '0.523944079875946, Validation Loss: 1.7202118635177612 - Validation Accuracy '
     '0.5622336268424988')
    ('EPOCH 11: Training Loss 1.8912638425827026 - Training Accuracy '
     '0.5276938080787659, Validation Loss: 1.6574574708938599 - Validation '
     'Accuracy 0.569985568523407')
    ('EPOCH 12: Training Loss 1.8354926109313965 - Training Accuracy '
     '0.5311025977134705, Validation Loss: 1.6102063655853271 - Validation '
     'Accuracy 0.5720997452735901')
    ('EPOCH 13: Training Loss 1.7894525527954102 - Training Accuracy '
     '0.5274958610534668, Validation Loss: 1.5806093215942383 - Validation '
     'Accuracy 0.5734084844589233')
    ('EPOCH 14: Training Loss 1.7548953294754028 - Training Accuracy '
     '0.5338516235351562, Validation Loss: 1.551721215248108 - Validation Accuracy '
     '0.5788784623146057')
    ('EPOCH 15: Training Loss 1.7175843715667725 - Training Accuracy '
     '0.5436821579933167, Validation Loss: 1.539406418800354 - Validation Accuracy '
     '0.5825027823448181')
    ('EPOCH 16: Training Loss 1.6858525276184082 - Training Accuracy '
     '0.5506647229194641, Validation Loss: 1.527065634727478 - Validation Accuracy '
     '0.5825027823448181')
    ('EPOCH 17: Training Loss 1.6732540130615234 - Training Accuracy '
     '0.5565916299819946, Validation Loss: 1.5017892122268677 - Validation '
     'Accuracy 0.5888452529907227')
    ('EPOCH 18: Training Loss 1.6359189748764038 - Training Accuracy '
     '0.5619797706604004, Validation Loss: 1.4573795795440674 - Validation '
     'Accuracy 0.598711371421814')
    ('EPOCH 19: Training Loss 1.6184966564178467 - Training Accuracy '
     '0.5625185370445251, Validation Loss: 1.4577335119247437 - Validation '
     'Accuracy 0.597100555896759')
    ('EPOCH 20: Training Loss 1.5965512990951538 - Training Accuracy '
     '0.5649266839027405, Validation Loss: 1.4392340183258057 - Validation '
     'Accuracy 0.5991140604019165')
    ('EPOCH 21: Training Loss 1.5803583860397339 - Training Accuracy '
     '0.5667300820350647, Validation Loss: 1.425147533416748 - Validation Accuracy '
     '0.6031075119972229')
    ('EPOCH 22: Training Loss 1.5569713115692139 - Training Accuracy '
     '0.5717552900314331, Validation Loss: 1.4219950437545776 - Validation '
     'Accuracy 0.6032417416572571')
    ('EPOCH 23: Training Loss 1.5593047142028809 - Training Accuracy '
     '0.571854293346405, Validation Loss: 1.4094254970550537 - Validation Accuracy '
     '0.6045169234275818')
    ('EPOCH 24: Training Loss 1.5369622707366943 - Training Accuracy '
     '0.576989471912384, Validation Loss: 1.3923264741897583 - Validation Accuracy '
     '0.6079062819480896')
    ('EPOCH 25: Training Loss 1.520815134048462 - Training Accuracy '
     '0.5785069465637207, Validation Loss: 1.3835208415985107 - Validation '
     'Accuracy 0.6137790083885193')
    ('EPOCH 26: Training Loss 1.5089367628097534 - Training Accuracy '
     '0.5809481143951416, Validation Loss: 1.3704345226287842 - Validation '
     'Accuracy 0.615423321723938')
    ('EPOCH 27: Training Loss 1.4931271076202393 - Training Accuracy '
     '0.5852915644645691, Validation Loss: 1.3579683303833008 - Validation '
     'Accuracy 0.6213295459747314')
    ('EPOCH 28: Training Loss 1.4887363910675049 - Training Accuracy '
     '0.5866550803184509, Validation Loss: 1.3461371660232544 - Validation '
     'Accuracy 0.6239135265350342')
    ('EPOCH 29: Training Loss 1.4740759134292603 - Training Accuracy '
     '0.5887113809585571, Validation Loss: 1.3379082679748535 - Validation '
     'Accuracy 0.6216987371444702')
    ('EPOCH 30: Training Loss 1.463323712348938 - Training Accuracy '
     '0.591658353805542, Validation Loss: 1.3307310342788696 - Validation Accuracy '
     '0.6240142583847046')
    ('EPOCH 31: Training Loss 1.4544093608856201 - Training Accuracy '
     '0.5915813446044922, Validation Loss: 1.3400272130966187 - Validation '
     'Accuracy 0.6198194622993469')
    ('EPOCH 32: Training Loss 1.4515955448150635 - Training Accuracy '
     '0.592164158821106, Validation Loss: 1.331037998199463 - Validation Accuracy '
     '0.6226047873497009')
    ('EPOCH 33: Training Loss 1.4364632368087769 - Training Accuracy '
     '0.5968044996261597, Validation Loss: 1.3277623653411865 - Validation '
     'Accuracy 0.6233094930648804')
    ('EPOCH 34: Training Loss 1.436464786529541 - Training Accuracy '
     '0.595737874507904, Validation Loss: 1.3087931871414185 - Validation Accuracy '
     '0.6263297200202942')
    ('EPOCH 35: Training Loss 1.42304527759552 - Training Accuracy '
     '0.5981460213661194, Validation Loss: 1.2955769300460815 - Validation '
     'Accuracy 0.6301553845405579')
    ('EPOCH 36: Training Loss 1.4147931337356567 - Training Accuracy '
     '0.6017637848854065, Validation Loss: 1.3002129793167114 - Validation '
     'Accuracy 0.6282089948654175')
    ('EPOCH 37: Training Loss 1.4113359451293945 - Training Accuracy '
     '0.599333643913269, Validation Loss: 1.291733741760254 - Validation Accuracy '
     '0.6307258605957031')
    ('EPOCH 38: Training Loss 1.4051276445388794 - Training Accuracy '
     '0.6038420796394348, Validation Loss: 1.288472294807434 - Validation Accuracy '
     '0.633007824420929')
    ('EPOCH 39: Training Loss 1.399509072303772 - Training Accuracy '
     '0.6053705215454102, Validation Loss: 1.2792085409164429 - Validation '
     'Accuracy 0.6336789727210999')
    ('EPOCH 40: Training Loss 1.3949776887893677 - Training Accuracy '
     '0.6073058247566223, Validation Loss: 1.2780910730361938 - Validation '
     'Accuracy 0.6332091689109802')
    ('EPOCH 41: Training Loss 1.3871749639511108 - Training Accuracy '
     '0.6079986095428467, Validation Loss: 1.2762337923049927 - Validation '
     'Accuracy 0.6346186399459839')
    ('EPOCH 42: Training Loss 1.3786572217941284 - Training Accuracy '
     '0.6085593700408936, Validation Loss: 1.2680222988128662 - Validation '
     'Accuracy 0.6314976811408997')
    ('EPOCH 43: Training Loss 1.3770931959152222 - Training Accuracy '
     '0.6088563203811646, Validation Loss: 1.2642197608947754 - Validation '
     'Accuracy 0.636531412601471')
    ('EPOCH 44: Training Loss 1.3731147050857544 - Training Accuracy '
     '0.6105167269706726, Validation Loss: 1.2605690956115723 - Validation '
     'Accuracy 0.6356253623962402')
    ('EPOCH 45: Training Loss 1.3635729551315308 - Training Accuracy '
     '0.6117592453956604, Validation Loss: 1.260422945022583 - Validation Accuracy '
     '0.6403906345367432')
    ('EPOCH 46: Training Loss 1.361771583557129 - Training Accuracy '
     '0.611792266368866, Validation Loss: 1.2555997371673584 - Validation Accuracy '
     '0.6380751132965088')
    ('EPOCH 47: Training Loss 1.3539824485778809 - Training Accuracy '
     '0.6140574812889099, Validation Loss: 1.2477115392684937 - Validation '
     'Accuracy 0.6402899622917175')
    ('EPOCH 48: Training Loss 1.3536924123764038 - Training Accuracy '
     '0.6136946082115173, Validation Loss: 1.2510639429092407 - Validation '
     'Accuracy 0.6385449171066284')
    ('EPOCH 49: Training Loss 1.3514790534973145 - Training Accuracy '
     '0.6152450442314148, Validation Loss: 1.2499184608459473 - Validation '
     'Accuracy 0.6376053094863892')
    ('EPOCH 50: Training Loss 1.3473351001739502 - Training Accuracy '
     '0.617301344871521, Validation Loss: 1.2432115077972412 - Validation Accuracy '
     '0.6407262086868286')
    ('EPOCH 51: Training Loss 1.3405267000198364 - Training Accuracy '
     '0.6174772381782532, Validation Loss: 1.2458784580230713 - Validation '
     'Accuracy 0.637471079826355')
    ('EPOCH 52: Training Loss 1.339882254600525 - Training Accuracy '
     '0.6165755987167358, Validation Loss: 1.2324086427688599 - Validation '
     'Accuracy 0.6396187543869019')
    ('EPOCH 53: Training Loss 1.3347035646438599 - Training Accuracy '
     '0.6178621053695679, Validation Loss: 1.2356542348861694 - Validation '
     'Accuracy 0.6422027349472046')
    ('EPOCH 54: Training Loss 1.3284987211227417 - Training Accuracy '
     '0.6205781698226929, Validation Loss: 1.2274048328399658 - Validation '
     'Accuracy 0.6438135504722595')
    ('EPOCH 55: Training Loss 1.324997067451477 - Training Accuracy '
     '0.6212049722671509, Validation Loss: 1.2241066694259644 - Validation '
     'Accuracy 0.6436122059822083')
    ('EPOCH 56: Training Loss 1.3260091543197632 - Training Accuracy '
     '0.6191706657409668, Validation Loss: 1.2232692241668701 - Validation '
     'Accuracy 0.643410861492157')
    ('EPOCH 57: Training Loss 1.3215252161026 - Training Accuracy '
     '0.6220296621322632, Validation Loss: 1.2200995683670044 - Validation '
     'Accuracy 0.6442497968673706')
    ('EPOCH 58: Training Loss 1.3219250440597534 - Training Accuracy '
     '0.622359573841095, Validation Loss: 1.2220678329467773 - Validation Accuracy '
     '0.644585371017456')
    ('EPOCH 59: Training Loss 1.3132096529006958 - Training Accuracy '
     '0.6229533553123474, Validation Loss: 1.2184311151504517 - Validation '
     'Accuracy 0.6438471078872681')
    ('EPOCH 60: Training Loss 1.311281442642212 - Training Accuracy '
     '0.6243058443069458, Validation Loss: 1.2182846069335938 - Validation '
     'Accuracy 0.6440148949623108')
    ('EPOCH 61: Training Loss 1.3103069067001343 - Training Accuracy '
     '0.6225684881210327, Validation Loss: 1.2183341979980469 - Validation '
     'Accuracy 0.6436122059822083')
    ('EPOCH 62: Training Loss 1.3072173595428467 - Training Accuracy '
     '0.6240639686584473, Validation Loss: 1.2165509462356567 - Validation '
     'Accuracy 0.6424040794372559')
    ('EPOCH 63: Training Loss 1.3091460466384888 - Training Accuracy '
     '0.6238550543785095, Validation Loss: 1.2113066911697388 - Validation '
     'Accuracy 0.6439142227172852')
    ('EPOCH 64: Training Loss 1.3001068830490112 - Training Accuracy '
     '0.6256254315376282, Validation Loss: 1.2142322063446045 - Validation '
     'Accuracy 0.646196186542511')
    ('EPOCH 65: Training Loss 1.3011871576309204 - Training Accuracy '
     '0.6243938207626343, Validation Loss: 1.20695960521698 - Validation Accuracy '
     '0.645457923412323')
    ('EPOCH 66: Training Loss 1.2959412336349487 - Training Accuracy '
     '0.6249986290931702, Validation Loss: 1.209287405014038 - Validation Accuracy '
     '0.6432430744171143')
    ('EPOCH 67: Training Loss 1.292757272720337 - Training Accuracy '
     '0.628198504447937, Validation Loss: 1.2023972272872925 - Validation Accuracy '
     '0.6471022367477417')
    ('EPOCH 68: Training Loss 1.293175458908081 - Training Accuracy '
     '0.6272198557853699, Validation Loss: 1.2032721042633057 - Validation '
     'Accuracy 0.6483774781227112')
    ('EPOCH 69: Training Loss 1.2914886474609375 - Training Accuracy '
     '0.6277366876602173, Validation Loss: 1.1964781284332275 - Validation '
     'Accuracy 0.6501895785331726')
    ('EPOCH 70: Training Loss 1.2863305807113647 - Training Accuracy '
     '0.6289242506027222, Validation Loss: 1.2012525796890259 - Validation '
     'Accuracy 0.6446524858474731')
    ('EPOCH 71: Training Loss 1.2854753732681274 - Training Accuracy '
     '0.6266480684280396, Validation Loss: 1.1970255374908447 - Validation '
     'Accuracy 0.6488472819328308')
    ('EPOCH 72: Training Loss 1.2854725122451782 - Training Accuracy '
     '0.6285833716392517, Validation Loss: 1.1978278160095215 - Validation '
     'Accuracy 0.6491492986679077')
    ('EPOCH 73: Training Loss 1.2802027463912964 - Training Accuracy '
     '0.6294630765914917, Validation Loss: 1.1937916278839111 - Validation '
     'Accuracy 0.6464310884475708')
    ('EPOCH 74: Training Loss 1.2773677110671997 - Training Accuracy '
     '0.6304966807365417, Validation Loss: 1.195153832435608 - Validation Accuracy '
     '0.6482768058776855')
    ('EPOCH 75: Training Loss 1.2734477519989014 - Training Accuracy '
     '0.6301338076591492, Validation Loss: 1.1959137916564941 - Validation '
     'Accuracy 0.6490486264228821')
    ('EPOCH 76: Training Loss 1.2745192050933838 - Training Accuracy '
     '0.6314423680305481, Validation Loss: 1.1942942142486572 - Validation '
     'Accuracy 0.6459277272224426')
    ('EPOCH 77: Training Loss 1.2734097242355347 - Training Accuracy '
     '0.6308375597000122, Validation Loss: 1.1928750276565552 - Validation '
     'Accuracy 0.6500217914581299')
    ('EPOCH 78: Training Loss 1.2692279815673828 - Training Accuracy '
     '0.6323220729827881, Validation Loss: 1.1910803318023682 - Validation '
     'Accuracy 0.6516997218132019')
    ('EPOCH 79: Training Loss 1.2652544975280762 - Training Accuracy '
     '0.633509635925293, Validation Loss: 1.1837472915649414 - Validation Accuracy '
     '0.6514983773231506')
    ('EPOCH 80: Training Loss 1.2650147676467896 - Training Accuracy '
     '0.6327838897705078, Validation Loss: 1.182461142539978 - Validation Accuracy '
     '0.6534112095832825')
    ('EPOCH 81: Training Loss 1.2625006437301636 - Training Accuracy '
     '0.6325639486312866, Validation Loss: 1.1812994480133057 - Validation '
     'Accuracy 0.6519010663032532')
    ('EPOCH 82: Training Loss 1.2630996704101562 - Training Accuracy '
     '0.633256733417511, Validation Loss: 1.1861939430236816 - Validation Accuracy '
     '0.6508943438529968')
    ('EPOCH 83: Training Loss 1.2616714239120483 - Training Accuracy '
     '0.6336085796356201, Validation Loss: 1.1845701932907104 - Validation '
     'Accuracy 0.6490486264228821')
    ('EPOCH 84: Training Loss 1.2622517347335815 - Training Accuracy '
     '0.6326959133148193, Validation Loss: 1.1798264980316162 - Validation '
     'Accuracy 0.6521695256233215')
    ('EPOCH 85: Training Loss 1.2610725164413452 - Training Accuracy '
     '0.6347522139549255, Validation Loss: 1.1810328960418701 - Validation '
     'Accuracy 0.6508943438529968')
    ('EPOCH 86: Training Loss 1.2566906213760376 - Training Accuracy '
     '0.6333886981010437, Validation Loss: 1.175683856010437 - Validation Accuracy '
     '0.6516661643981934')
    ('EPOCH 87: Training Loss 1.253786325454712 - Training Accuracy '
     '0.6363246440887451, Validation Loss: 1.1719753742218018 - Validation '
     'Accuracy 0.654552161693573')
    ('EPOCH 88: Training Loss 1.2523829936981201 - Training Accuracy '
     '0.6348071694374084, Validation Loss: 1.175817608833313 - Validation Accuracy '
     '0.6538474559783936')
    ('EPOCH 89: Training Loss 1.2515549659729004 - Training Accuracy '
     '0.6356539130210876, Validation Loss: 1.1799074411392212 - Validation '
     'Accuracy 0.6511963605880737')
    ('EPOCH 90: Training Loss 1.2520288228988647 - Training Accuracy '
     '0.6357198357582092, Validation Loss: 1.1808761358261108 - Validation '
     'Accuracy 0.6519681811332703')
    ('EPOCH 91: Training Loss 1.2474076747894287 - Training Accuracy '
     '0.6363576650619507, Validation Loss: 1.1670340299606323 - Validation '
     'Accuracy 0.6567670106887817')
    ('EPOCH 92: Training Loss 1.2434743642807007 - Training Accuracy '
     '0.6373472809791565, Validation Loss: 1.1718522310256958 - Validation '
     'Accuracy 0.6523708701133728')
    ('EPOCH 93: Training Loss 1.2469767332077026 - Training Accuracy '
     '0.6355109214782715, Validation Loss: 1.1699178218841553 - Validation '
     'Accuracy 0.6549548506736755')
    ('EPOCH 94: Training Loss 1.2443809509277344 - Training Accuracy '
     '0.6383699178695679, Validation Loss: 1.171436071395874 - Validation Accuracy '
     '0.653813898563385')
    ('EPOCH 95: Training Loss 1.2430126667022705 - Training Accuracy '
     '0.6382269859313965, Validation Loss: 1.168982744216919 - Validation Accuracy '
     '0.6566327810287476')
    ('EPOCH 96: Training Loss 1.2378493547439575 - Training Accuracy '
     '0.6393375992774963, Validation Loss: 1.1733343601226807 - Validation '
     'Accuracy 0.6531427502632141')
    ('EPOCH 97: Training Loss 1.2399876117706299 - Training Accuracy '
     '0.6385238766670227, Validation Loss: 1.1707701683044434 - Validation '
     'Accuracy 0.652874231338501')
    ('EPOCH 98: Training Loss 1.2329277992248535 - Training Accuracy '
     '0.6410199999809265, Validation Loss: 1.171721339225769 - Validation Accuracy '
     '0.6516661643981934')
    ('EPOCH 99: Training Loss 1.2340316772460938 - Training Accuracy '
     '0.6403272747993469, Validation Loss: 1.167144775390625 - Validation Accuracy '
     '0.6529077887535095')
    ('EPOCH 100: Training Loss 1.2381778955459595 - Training Accuracy '
     '0.6391066908836365, Validation Loss: 1.162628412246704 - Validation Accuracy '
     '0.6568341255187988')
    ('EPOCH 101: Training Loss 1.233432650566101 - Training Accuracy '
     '0.6394475698471069, Validation Loss: 1.1696255207061768 - Validation '
     'Accuracy 0.6542501449584961')
    ('EPOCH 102: Training Loss 1.2293239831924438 - Training Accuracy '
     '0.6402392983436584, Validation Loss: 1.1646771430969238 - Validation '
     'Accuracy 0.6539481282234192')
    ('EPOCH 103: Training Loss 1.2285866737365723 - Training Accuracy '
     '0.6420536637306213, Validation Loss: 1.16614830493927 - Validation Accuracy '
     '0.6574046015739441')
    ('EPOCH 104: Training Loss 1.228315830230713 - Training Accuracy '
     '0.6418666839599609, Validation Loss: 1.16093111038208 - Validation Accuracy '
     '0.6551226377487183')
    ('EPOCH 105: Training Loss 1.2285031080245972 - Training Accuracy '
     '0.640855073928833, Validation Loss: 1.1631016731262207 - Validation Accuracy '
     '0.6531091928482056')
    ('EPOCH 106: Training Loss 1.2242642641067505 - Training Accuracy '
     '0.643021285533905, Validation Loss: 1.1600404977798462 - Validation Accuracy '
     '0.6546528339385986')
    ('EPOCH 107: Training Loss 1.2235885858535767 - Training Accuracy '
     '0.6438680291175842, Validation Loss: 1.1607531309127808 - Validation '
     'Accuracy 0.6555588841438293')
    ('EPOCH 108: Training Loss 1.2275550365447998 - Training Accuracy '
     '0.6412509083747864, Validation Loss: 1.1590447425842285 - Validation '
     'Accuracy 0.6568341255187988')
    ('EPOCH 109: Training Loss 1.2208281755447388 - Training Accuracy '
     '0.643395185470581, Validation Loss: 1.155989170074463 - Validation Accuracy '
     '0.6575052738189697')
    ('EPOCH 110: Training Loss 1.2195017337799072 - Training Accuracy '
     '0.6419546604156494, Validation Loss: 1.1538552045822144 - Validation '
     'Accuracy 0.6587469577789307')
    ('EPOCH 111: Training Loss 1.219171404838562 - Training Accuracy '
     '0.6433072090148926, Validation Loss: 1.1576976776123047 - Validation '
     'Accuracy 0.6580421924591064')
    ('EPOCH 112: Training Loss 1.2182495594024658 - Training Accuracy '
     '0.6432961821556091, Validation Loss: 1.1583119630813599 - Validation '
     'Accuracy 0.6565321087837219')
    ('EPOCH 113: Training Loss 1.2141237258911133 - Training Accuracy '
     '0.6427574157714844, Validation Loss: 1.157296061515808 - Validation Accuracy '
     '0.6574717164039612')
    ('EPOCH 114: Training Loss 1.2167208194732666 - Training Accuracy '
     '0.6455064415931702, Validation Loss: 1.1541063785552979 - Validation '
     'Accuracy 0.6569012403488159')
    ('EPOCH 115: Training Loss 1.2118271589279175 - Training Accuracy '
     '0.6444947719573975, Validation Loss: 1.151504635810852 - Validation Accuracy '
     '0.6578744053840637')
    ('EPOCH 116: Training Loss 1.2146108150482178 - Training Accuracy '
     '0.6450555920600891, Validation Loss: 1.1530565023422241 - Validation '
     'Accuracy 0.6570019125938416')
    ('EPOCH 117: Training Loss 1.2112553119659424 - Training Accuracy '
     '0.6453414559364319, Validation Loss: 1.1556538343429565 - Validation '
     'Accuracy 0.6545186042785645')
    ('EPOCH 118: Training Loss 1.2078008651733398 - Training Accuracy '
     '0.6441649198532104, Validation Loss: 1.1565064191818237 - Validation '
     'Accuracy 0.6578744053840637')
    ('EPOCH 119: Training Loss 1.2049778699874878 - Training Accuracy '
     '0.6459792852401733, Validation Loss: 1.1514333486557007 - Validation '
     'Accuracy 0.6580421924591064')
    ('EPOCH 120: Training Loss 1.2063854932785034 - Training Accuracy '
     '0.6473097801208496, Validation Loss: 1.1458617448806763 - Validation '
     'Accuracy 0.6591496467590332')
    ('EPOCH 121: Training Loss 1.2098307609558105 - Training Accuracy '
     '0.6442858576774597, Validation Loss: 1.1520638465881348 - Validation '
     'Accuracy 0.6566663384437561')
    ('EPOCH 122: Training Loss 1.200109601020813 - Training Accuracy '
     '0.6463751196861267, Validation Loss: 1.1415117979049683 - Validation '
     'Accuracy 0.6602905988693237')
    ('EPOCH 123: Training Loss 1.1996217966079712 - Training Accuracy '
     '0.6476946473121643, Validation Loss: 1.1482740640640259 - Validation '
     'Accuracy 0.6593509912490845')
    ('EPOCH 124: Training Loss 1.2033480405807495 - Training Accuracy '
     '0.6476726531982422, Validation Loss: 1.1477364301681519 - Validation '
     'Accuracy 0.6589818596839905')
    ('EPOCH 125: Training Loss 1.2008922100067139 - Training Accuracy '
     '0.6463971138000488, Validation Loss: 1.1446365118026733 - Validation '
     'Accuracy 0.6607268452644348')
    ('EPOCH 126: Training Loss 1.2004257440567017 - Training Accuracy '
     '0.6464411020278931, Validation Loss: 1.1420772075653076 - Validation '
     'Accuracy 0.6593509912490845')
    ('EPOCH 127: Training Loss 1.1940407752990723 - Training Accuracy '
     '0.6470348834991455, Validation Loss: 1.1447041034698486 - Validation '
     'Accuracy 0.6586798429489136')
    ('EPOCH 128: Training Loss 1.1932281255722046 - Training Accuracy '
     '0.6489812135696411, Validation Loss: 1.1481304168701172 - Validation '
     'Accuracy 0.6571696996688843')
    ('EPOCH 129: Training Loss 1.198566198348999 - Training Accuracy '
     '0.646946907043457, Validation Loss: 1.146925449371338 - Validation Accuracy '
     '0.6580086350440979')
    ('EPOCH 130: Training Loss 1.1918668746948242 - Training Accuracy '
     '0.6496519446372986, Validation Loss: 1.144922137260437 - Validation Accuracy '
     '0.659384548664093')
    ('EPOCH 131: Training Loss 1.1930668354034424 - Training Accuracy '
     '0.6479585766792297, Validation Loss: 1.1437592506408691 - Validation '
     'Accuracy 0.6587469577789307')
    ('EPOCH 132: Training Loss 1.1931535005569458 - Training Accuracy '
     '0.6490252017974854, Validation Loss: 1.1409430503845215 - Validation '
     'Accuracy 0.6609618067741394')
    ('EPOCH 133: Training Loss 1.193694829940796 - Training Accuracy '
     '0.6480575203895569, Validation Loss: 1.1478307247161865 - Validation '
     'Accuracy 0.6568341255187988')
    ('EPOCH 134: Training Loss 1.1920877695083618 - Training Accuracy '
     '0.6489921808242798, Validation Loss: 1.1420891284942627 - Validation '
     'Accuracy 0.6564649939537048')
    ('EPOCH 135: Training Loss 1.1857073307037354 - Training Accuracy '
     '0.6508505344390869, Validation Loss: 1.1421453952789307 - Validation '
     'Accuracy 0.6616665124893188')
    ('EPOCH 136: Training Loss 1.1874955892562866 - Training Accuracy '
     '0.6492011547088623, Validation Loss: 1.139595627784729 - Validation Accuracy '
     '0.6602234840393066')
    ('EPOCH 137: Training Loss 1.1897166967391968 - Training Accuracy '
     '0.6497069597244263, Validation Loss: 1.1422919034957886 - Validation '
     'Accuracy 0.6593174338340759')
    ('EPOCH 138: Training Loss 1.1833295822143555 - Training Accuracy '
     '0.6500038504600525, Validation Loss: 1.1443202495574951 - Validation '
     'Accuracy 0.6577401757240295')
    ('EPOCH 139: Training Loss 1.1832233667373657 - Training Accuracy '
     '0.6502347588539124, Validation Loss: 1.1400026082992554 - Validation '
     'Accuracy 0.661431610584259')
    ('EPOCH 140: Training Loss 1.1827813386917114 - Training Accuracy '
     '0.6510375142097473, Validation Loss: 1.1412684917449951 - Validation '
     'Accuracy 0.6571361422538757')
    ('EPOCH 141: Training Loss 1.179983139038086 - Training Accuracy '
     '0.6512573957443237, Validation Loss: 1.1359436511993408 - Validation '
     'Accuracy 0.6632101535797119')
    ('EPOCH 142: Training Loss 1.1797080039978027 - Training Accuracy '
     '0.6525879502296448, Validation Loss: 1.144956350326538 - Validation Accuracy '
     '0.6587805151939392')
    ('EPOCH 143: Training Loss 1.1813266277313232 - Training Accuracy '
     '0.6519611477851868, Validation Loss: 1.149204134941101 - Validation Accuracy '
     '0.6589483022689819')
    ('EPOCH 144: Training Loss 1.1774513721466064 - Training Accuracy '
     '0.6528958082199097, Validation Loss: 1.145519733428955 - Validation Accuracy '
     '0.6564985513687134')
    ('EPOCH 145: Training Loss 1.176845908164978 - Training Accuracy '
     '0.6527748703956604, Validation Loss: 1.1322463750839233 - Validation '
     'Accuracy 0.6641162633895874')
    ('EPOCH 146: Training Loss 1.1762384176254272 - Training Accuracy '
     '0.6531267762184143, Validation Loss: 1.1336995363235474 - Validation '
     'Accuracy 0.6623376607894897')
    ('EPOCH 147: Training Loss 1.1763666868209839 - Training Accuracy '
     '0.6530387997627258, Validation Loss: 1.1386879682540894 - Validation '
     'Accuracy 0.6615993976593018')
    ('EPOCH 148: Training Loss 1.1721452474594116 - Training Accuracy '
     '0.6545782685279846, Validation Loss: 1.1390571594238281 - Validation '
     'Accuracy 0.659015417098999')
    ('EPOCH 149: Training Loss 1.173622727394104 - Training Accuracy '
     '0.6542923450469971, Validation Loss: 1.1370043754577637 - Validation '
     'Accuracy 0.6599550247192383')
    ('EPOCH 150: Training Loss 1.174483060836792 - Training Accuracy '
     '0.6530387997627258, Validation Loss: 1.1390354633331299 - Validation '
     'Accuracy 0.6612638235092163')
    ('EPOCH 151: Training Loss 1.1699049472808838 - Training Accuracy '
     '0.6556888818740845, Validation Loss: 1.1343441009521484 - Validation '
     'Accuracy 0.6616665124893188')
    ('EPOCH 152: Training Loss 1.1750684976577759 - Training Accuracy '
     '0.6520601511001587, Validation Loss: 1.1326459646224976 - Validation '
     'Accuracy 0.6643511652946472')
    ('EPOCH 153: Training Loss 1.1713052988052368 - Training Accuracy '
     '0.6532917022705078, Validation Loss: 1.1363555192947388 - Validation '
     'Accuracy 0.6622369885444641')
    ('EPOCH 154: Training Loss 1.1680693626403809 - Training Accuracy '
     '0.653335690498352, Validation Loss: 1.132314682006836 - Validation Accuracy '
     '0.6591160893440247')
    ('EPOCH 155: Training Loss 1.1713037490844727 - Training Accuracy '
     '0.6547761559486389, Validation Loss: 1.1345618963241577 - Validation '
     'Accuracy 0.6611295938491821')
    ('EPOCH 156: Training Loss 1.1652553081512451 - Training Accuracy '
     '0.6553479433059692, Validation Loss: 1.1312551498413086 - Validation '
     'Accuracy 0.6609618067741394')
    ('EPOCH 157: Training Loss 1.1638739109039307 - Training Accuracy '
     '0.6548751592636108, Validation Loss: 1.1297193765640259 - Validation '
     'Accuracy 0.6652236580848694')
    ('EPOCH 158: Training Loss 1.1662589311599731 - Training Accuracy '
     '0.654721200466156, Validation Loss: 1.1293823719024658 - Validation Accuracy '
     '0.6639484763145447')
    ('EPOCH 159: Training Loss 1.1624215841293335 - Training Accuracy '
     '0.6547761559486389, Validation Loss: 1.1311123371124268 - Validation '
     'Accuracy 0.6630759239196777')
    ('EPOCH 160: Training Loss 1.1634809970855713 - Training Accuracy '
     '0.6554689407348633, Validation Loss: 1.1383410692214966 - Validation '
     'Accuracy 0.6610960364341736')
    ('EPOCH 161: Training Loss 1.158238172531128 - Training Accuracy '
     '0.6583279371261597, Validation Loss: 1.1320996284484863 - Validation '
     'Accuracy 0.662539005279541')
    ('EPOCH 162: Training Loss 1.163397192955017 - Training Accuracy '
     '0.6554909348487854, Validation Loss: 1.1318773031234741 - Validation '
     'Accuracy 0.6621363162994385')
    ('EPOCH 163: Training Loss 1.1606827974319458 - Training Accuracy '
     '0.6555458903312683, Validation Loss: 1.1287322044372559 - Validation '
     'Accuracy 0.6639820337295532')
    ('EPOCH 164: Training Loss 1.1537909507751465 - Training Accuracy '
     '0.657943069934845, Validation Loss: 1.1239657402038574 - Validation Accuracy '
     '0.6663311123847961')
    ('EPOCH 165: Training Loss 1.15936279296875 - Training Accuracy '
     '0.6563156247138977, Validation Loss: 1.1321452856063843 - Validation '
     'Accuracy 0.661431610584259')
    ('EPOCH 166: Training Loss 1.1613891124725342 - Training Accuracy '
     '0.6560187339782715, Validation Loss: 1.1330485343933105 - Validation '
     'Accuracy 0.6611295938491821')
    ('EPOCH 167: Training Loss 1.154807209968567 - Training Accuracy '
     '0.6585918068885803, Validation Loss: 1.1328761577606201 - Validation '
     'Accuracy 0.662169873714447')
    ('EPOCH 168: Training Loss 1.1581997871398926 - Training Accuracy '
     '0.6568984389305115, Validation Loss: 1.1286633014678955 - Validation '
     'Accuracy 0.6623376607894897')
    ('EPOCH 169: Training Loss 1.153317928314209 - Training Accuracy '
     '0.6565245389938354, Validation Loss: 1.1244421005249023 - Validation '
     'Accuracy 0.6662975549697876')
    ('EPOCH 170: Training Loss 1.1532480716705322 - Training Accuracy '
     '0.657943069934845, Validation Loss: 1.1305570602416992 - Validation Accuracy '
     '0.6619349718093872')
    ('EPOCH 171: Training Loss 1.150705337524414 - Training Accuracy '
     '0.6560956835746765, Validation Loss: 1.1273351907730103 - Validation '
     'Accuracy 0.6606597304344177')
    ('EPOCH 172: Training Loss 1.153024673461914 - Training Accuracy '
     '0.6581739783287048, Validation Loss: 1.1287212371826172 - Validation '
     'Accuracy 0.6627067923545837')
    ('EPOCH 173: Training Loss 1.1524055004119873 - Training Accuracy '
     '0.659152626991272, Validation Loss: 1.127245545387268 - Validation Accuracy '
     '0.6659954786300659')
    ('EPOCH 174: Training Loss 1.1441601514816284 - Training Accuracy '
     '0.6607140898704529, Validation Loss: 1.1265074014663696 - Validation '
     'Accuracy 0.664149820804596')
    ('EPOCH 175: Training Loss 1.1492916345596313 - Training Accuracy '
     '0.6586028337478638, Validation Loss: 1.1358999013900757 - Validation '
     'Accuracy 0.6612302660942078')
    ('EPOCH 176: Training Loss 1.149064540863037 - Training Accuracy '
     '0.6586797833442688, Validation Loss: 1.1274563074111938 - Validation '
     'Accuracy 0.6627739071846008')
    ('EPOCH 177: Training Loss 1.1477996110916138 - Training Accuracy '
     '0.6585698127746582, Validation Loss: 1.1243929862976074 - Validation '
     'Accuracy 0.664149820804596')
    ('EPOCH 178: Training Loss 1.145031213760376 - Training Accuracy '
     '0.6586138010025024, Validation Loss: 1.1210060119628906 - Validation '
     'Accuracy 0.6657605767250061')
    ('EPOCH 179: Training Loss 1.1456339359283447 - Training Accuracy '
     '0.6586908102035522, Validation Loss: 1.1281075477600098 - Validation '
     'Accuracy 0.6627403497695923')
    ('EPOCH 180: Training Loss 1.141016960144043 - Training Accuracy '
     '0.6599223613739014, Validation Loss: 1.1302917003631592 - Validation '
     'Accuracy 0.6620020866394043')
    ('EPOCH 181: Training Loss 1.1450337171554565 - Training Accuracy '
     '0.6610879302024841, Validation Loss: 1.121757984161377 - Validation Accuracy '
     '0.6669015884399414')
    ('EPOCH 182: Training Loss 1.1433439254760742 - Training Accuracy '
     '0.6589877009391785, Validation Loss: 1.123853087425232 - Validation Accuracy '
     '0.6658948063850403')
    ('EPOCH 183: Training Loss 1.1413177251815796 - Training Accuracy '
     '0.6609340310096741, Validation Loss: 1.122088074684143 - Validation Accuracy '
     '0.6672036051750183')
    ('EPOCH 184: Training Loss 1.1435023546218872 - Training Accuracy '
     '0.6606481075286865, Validation Loss: 1.1228487491607666 - Validation '
     'Accuracy 0.6641162633895874')
    ('EPOCH 185: Training Loss 1.141489863395691 - Training Accuracy '
     '0.6592516303062439, Validation Loss: 1.1198722124099731 - Validation '
     'Accuracy 0.6636800169944763')
    ('EPOCH 186: Training Loss 1.1388636827468872 - Training Accuracy '
     '0.6617587208747864, Validation Loss: 1.1223069429397583 - Validation '
     'Accuracy 0.6651229858398438')
    ('EPOCH 187: Training Loss 1.1419072151184082 - Training Accuracy '
     '0.661241888999939, Validation Loss: 1.130564570426941 - Validation Accuracy '
     '0.6622034311294556')
    ('EPOCH 188: Training Loss 1.137138843536377 - Training Accuracy '
     '0.6632322072982788, Validation Loss: 1.123076319694519 - Validation Accuracy '
     '0.6661968231201172')
    ('EPOCH 189: Training Loss 1.1348074674606323 - Training Accuracy '
     '0.6623964905738831, Validation Loss: 1.1228841543197632 - Validation '
     'Accuracy 0.6661297082901001')
    ('EPOCH 190: Training Loss 1.1368030309677124 - Training Accuracy '
     '0.6620336174964905, Validation Loss: 1.1231878995895386 - Validation '
     'Accuracy 0.6638813614845276')
    ('EPOCH 191: Training Loss 1.137520432472229 - Training Accuracy '
     '0.661241888999939, Validation Loss: 1.116166591644287 - Validation Accuracy '
     '0.6681767702102661')
    ('EPOCH 192: Training Loss 1.13715398311615 - Training Accuracy '
     '0.6602412462234497, Validation Loss: 1.120430588722229 - Validation Accuracy '
     '0.6642504930496216')
    ('EPOCH 193: Training Loss 1.1334517002105713 - Training Accuracy '
     '0.662121593952179, Validation Loss: 1.1168136596679688 - Validation Accuracy '
     '0.6685794591903687')
    ('EPOCH 194: Training Loss 1.1346489191055298 - Training Accuracy '
     '0.6626713871955872, Validation Loss: 1.1209403276443481 - Validation '
     'Accuracy 0.6663311123847961')
    ('EPOCH 195: Training Loss 1.1338471174240112 - Training Accuracy '
     '0.6623855233192444, Validation Loss: 1.117313027381897 - Validation Accuracy '
     '0.6674385070800781')
    ('EPOCH 196: Training Loss 1.1278409957885742 - Training Accuracy '
     '0.6650905609130859, Validation Loss: 1.1224020719528198 - Validation '
     'Accuracy 0.6640827059745789')
    ('EPOCH 197: Training Loss 1.1320390701293945 - Training Accuracy '
     '0.6638370156288147, Validation Loss: 1.1184043884277344 - Validation '
     'Accuracy 0.6669015884399414')
    ('EPOCH 198: Training Loss 1.1309547424316406 - Training Accuracy '
     '0.6626273989677429, Validation Loss: 1.117652416229248 - Validation Accuracy '
     '0.6659283638000488')
    ('EPOCH 199: Training Loss 1.131151556968689 - Training Accuracy '
     '0.6640899181365967, Validation Loss: 1.118740200996399 - Validation Accuracy '
     '0.663847804069519')
    ('EPOCH 200: Training Loss 1.1301138401031494 - Training Accuracy '
     '0.6624075174331665, Validation Loss: 1.1184934377670288 - Validation '
     'Accuracy 0.667371392250061')
    'Test Loss: 1.058790683746338 Test Accuracy: 0.6791555881500244'
    ```

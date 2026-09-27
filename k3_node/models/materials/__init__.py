"""Materials and crystal models (aliased from k3_node.applications.materials)."""

import sys
from k3_node.applications.materials import *
from k3_node.applications.materials import (
    basis,
    core,
    readout,
    wrappers,
    io,
    megnet,
    m3gnet,
    tensornet,
    chgnet,
    so3net,
    grace,
    qet,
)
from k3_node.applications.materials import __all__

# Alias submodules in sys.modules for full backward compatibility
sys.modules[__name__ + ".basis"] = basis
sys.modules[__name__ + ".core"] = core
sys.modules[__name__ + ".readout"] = readout
sys.modules[__name__ + ".wrappers"] = wrappers
sys.modules[__name__ + ".io"] = io
sys.modules[__name__ + ".megnet"] = megnet
sys.modules[__name__ + ".m3gnet"] = m3gnet
sys.modules[__name__ + ".tensornet"] = tensornet
sys.modules[__name__ + ".chgnet"] = chgnet
sys.modules[__name__ + ".so3net"] = so3net
sys.modules[__name__ + ".grace"] = grace
sys.modules[__name__ + ".qet"] = qet

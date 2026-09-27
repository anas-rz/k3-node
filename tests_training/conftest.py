import os

import pytest
import keras


@pytest.fixture(autouse=True)
def _seed_everything():
    # Loss-decrease checks on random data are order-dependent without a fixed seed:
    # each test would otherwise inherit whatever RNG state earlier tests left behind.
    # Set K3_TEST_SEED to check that a result does not hinge on one particular seed.
    keras.utils.set_random_seed(int(os.environ.get("K3_TEST_SEED", "0")))

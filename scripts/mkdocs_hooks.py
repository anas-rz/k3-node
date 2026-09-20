import os
import sys

# Ensure scripts directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_examples_doc import generate_all


def on_pre_build(config):
    """Automatically discover examples/**/*.ipynb and update docs/examples pages."""
    generate_all()


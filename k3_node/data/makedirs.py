import os


def makedirs(path: str):
    """Recursively creates a directory."""
    os.makedirs(path, exist_ok=True)


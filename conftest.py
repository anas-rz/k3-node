import os

# Enable 64-bit precision for JAX to prevent int64/float64 truncation warnings
os.environ.setdefault("JAX_ENABLE_X64", "True")

try:
    import jax
    jax.config.update("jax_enable_x64", True)
except (ImportError, AttributeError):
    pass


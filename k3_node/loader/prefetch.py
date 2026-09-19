import warnings
from contextlib import nullcontext
from functools import partial
from typing import Any, Optional

try:
    import torch
except ImportError:
    torch = None
    DataLoader = object


class DeviceHelper:
    def __init__(self, device: Optional[Any] = None):
        if torch is None:
            self.device = 'cpu'
            self.is_gpu = False
            self.stream = None
            self.stream_context = nullcontext
            self.module = None
            return

        with_cuda = torch.cuda.is_available()
        with_xpu = hasattr(torch, 'xpu') and torch.xpu.is_available()

        if device is None:
            if with_cuda:
                device = 'cuda'
            elif with_xpu:
                device = 'xpu'
            else:
                device = 'cpu'

        self.device = torch.device(device)
        self.is_gpu = self.device.type in ['cuda', 'xpu']

        if ((self.device.type == 'cuda' and not with_cuda)
                or (self.device.type == 'xpu' and not with_xpu)):
            warnings.warn(
                f"Requested device '{self.device.type}' is not available, falling back to CPU",
                stacklevel=2,
            )
            self.device = torch.device('cpu')
            self.is_gpu = False

        self.stream = None
        self.stream_context = nullcontext
        self.module = getattr(torch, self.device.type) if self.is_gpu else None

    def maybe_init_stream(self) -> None:
        if self.is_gpu and self.module is not None:
            self.stream = self.module.Stream()
            self.stream_context = partial(self.module.stream, stream=self.stream)

    def maybe_wait_stream(self) -> None:
        if self.stream is not None and self.module is not None:
            self.module.current_stream().wait_stream(self.stream)


class PrefetchLoader:
    r"""A prefetcher class for asynchronously transferring data of a DataLoader
    from host memory to device memory.

    Args:
        loader (DataLoader): The data loader.
        device (torch.device, optional): The device to load the data to. (default: :obj:`None`)
    """
    def __init__(
        self,
        loader: Any,
        device: Optional[Any] = None,
    ):
        self.loader = loader
        self.device_helper = DeviceHelper(device)

    def non_blocking_transfer(self, batch: Any) -> Any:
        if not self.device_helper.is_gpu:
            return batch
        if isinstance(batch, (list, tuple)):
            return type(batch)(self.non_blocking_transfer(v) for v in batch)
        if isinstance(batch, dict):
            return {k: self.non_blocking_transfer(v) for k, v in batch.items()}

        if hasattr(batch, 'pin_memory'):
            batch = batch.pin_memory()
        if hasattr(batch, 'to'):
            return batch.to(self.device_helper.device, non_blocking=True)
        return batch

    def __iter__(self) -> Any:
        first = True
        self.device_helper.maybe_init_stream()

        batch = None
        for next_batch in self.loader:
            with self.device_helper.stream_context():
                next_batch = self.non_blocking_transfer(next_batch)

            if not first:
                yield batch
            else:
                first = False

            self.device_helper.maybe_wait_stream()
            batch = next_batch

        if batch is not None:
            yield batch

    def __len__(self) -> int:
        return len(self.loader)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.loader})'


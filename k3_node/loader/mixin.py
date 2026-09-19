import glob
import logging
import os
import os.path as osp
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None


def get_numa_nodes_cores() -> Dict[str, Any]:
    """Parses numa nodes information into a dictionary."""
    numa_node_paths = glob.glob('/sys/devices/system/node/node[0-9]*')
    if not numa_node_paths:
        return {}

    nodes = {}
    try:
        for node_path in numa_node_paths:
            numa_node_id = int(osp.basename(node_path)[4:])
            thread_siblings = {}
            for cpu_dir in glob.glob(osp.join(node_path, 'cpu[0-9]*')):
                cpu_id = int(osp.basename(cpu_dir)[3:])
                if cpu_id > 0:
                    with open(osp.join(cpu_dir, 'online')) as core_online_file:
                        core_online = int(core_online_file.read().splitlines()[0])
                else:
                    core_online = 1  # cpu0 is always online
                if core_online == 1:
                    with open(osp.join(cpu_dir, 'topology', 'core_id')) as core_id_file:
                        core_id = int(core_id_file.read().strip())
                        if core_id in thread_siblings:
                            thread_siblings[core_id].append(cpu_id)
                        else:
                            thread_siblings[core_id] = [cpu_id]

            nodes[numa_node_id] = sorted([(k, sorted(v)) for k, v in thread_siblings.items()])
    except (OSError, ValueError, IndexError):
        warnings.warn('Failed to read NUMA info')
        return {}

    return nodes


class WorkerInitWrapper:
    r"""Wraps the :attr:`worker_init_fn` argument for DataLoader workers."""
    def __init__(self, func: Optional[Callable]) -> None:
        self.func = func

    def __call__(self, worker_id: int) -> None:
        if self.func is not None:
            self.func(worker_id)


class LogMemoryMixin:
    r"""A context manager to enable logging of memory consumption in
    DataLoader workers.
    """
    def _mem_init_fn(self, worker_id: int) -> None:
        if psutil is not None:
            proc = psutil.Process(os.getpid())
            memory = proc.memory_info().rss / (1024 * 1024)
            logging.debug(f"Worker {worker_id} @ PID {proc.pid}: {memory:.2f} MB")
        self._old_worker_init_fn(worker_id)

    @contextmanager
    def enable_memory_log(self):
        self._old_worker_init_fn = WorkerInitWrapper(getattr(self, 'worker_init_fn', None))
        try:
            self.worker_init_fn = self._mem_init_fn
            yield
        finally:
            self.worker_init_fn = self._old_worker_init_fn


class MultithreadingMixin:
    r"""A context manager to enable multi-threading in DataLoader workers."""
    def _mt_init_fn(self, worker_id: int) -> None:
        if torch is not None:
            try:
                torch.set_num_threads(int(self._worker_threads))
            except IndexError as e:
                raise ValueError(f"Cannot set {self._worker_threads} threads in worker {worker_id}") from e
        self._old_worker_init_fn(worker_id)

    @contextmanager
    def enable_multithreading(self, worker_threads: Optional[int] = None):
        num_workers = getattr(self, 'num_workers', 0)
        if not num_workers > 0:
            raise ValueError(f"'enable_multithreading' needs to be performed with at least one worker (got {num_workers})")

        if torch is not None:
            if worker_threads is None:
                worker_threads = torch.get_num_threads() // num_workers
            if worker_threads > torch.get_num_threads():
                raise ValueError(
                    f"'worker_threads' should be smaller than total available threads {torch.get_num_threads()} (got {worker_threads})"
                )
            context = torch.multiprocessing.get_context()._name
            if context != 'spawn':
                raise ValueError(f"'enable_multithreading' can only be used with 'spawn' multiprocessing context (got {context})")
        else:
            if worker_threads is None:
                worker_threads = 1

        self._worker_threads = worker_threads
        self._old_worker_init_fn = WorkerInitWrapper(getattr(self, 'worker_init_fn', None))
        try:
            logging.debug(f"Using {worker_threads} threads in each worker")
            self.worker_init_fn = self._mt_init_fn
            yield
        finally:
            self.worker_init_fn = self._old_worker_init_fn


class AffinityMixin:
    r"""A context manager to enable CPU affinity for data loader workers."""
    def _aff_init_fn(self, worker_id: int) -> None:
        try:
            worker_cores = self.loader_cores[worker_id]
            if not isinstance(worker_cores, list):
                worker_cores = [worker_cores]

            if torch is not None and torch.multiprocessing.get_context()._name == 'spawn':
                torch.set_num_threads(len(worker_cores))

            if psutil is not None:
                psutil.Process().cpu_affinity(worker_cores)
        except IndexError as e:
            raise ValueError(f"Cannot use CPU affinity for worker ID {worker_id} on CPU {self.loader_cores}") from e

        self._old_worker_init_fn(worker_id)

    @contextmanager
    def enable_cpu_affinity(self, loader_cores: Optional[Union[List[List[int]], List[int]]] = None):
        num_workers = getattr(self, 'num_workers', 0)
        if not num_workers > 0:
            raise ValueError(f"'enable_cpu_affinity' should be used with at least one worker (got {num_workers})")
        if loader_cores and len(loader_cores) != num_workers:
            raise ValueError(
                f"The number of loader cores ({len(loader_cores)}) in 'enable_cpu_affinity' should match number of workers ({num_workers})"
            )

        from k3_node.data import HeteroData
        if hasattr(self, 'data') and isinstance(self.data, HeteroData):
            warnings.warn(
                "Due to conflicting parallelization methods it is not advised to use affinitization with 'HeteroData' datasets.",
                stacklevel=2,
            )

        self.loader_cores = loader_cores[:] if loader_cores else None
        if self.loader_cores is None:
            numa_info = get_numa_nodes_cores()
            if numa_info and len(numa_info.get(0, [])) > num_workers:
                node0_cores = [cpus[0] for core_id, cpus in numa_info[0]]
                node0_cores.sort()
            elif psutil is not None:
                node0_cores = list(range(psutil.cpu_count(logical=False) or 1))
            else:
                node0_cores = list(range(os.cpu_count() or 1))

            if len(node0_cores) < num_workers:
                raise ValueError(f"More workers ({num_workers}) than available cores ({len(node0_cores)})")

            if torch is not None and torch.multiprocessing.get_context()._name == 'spawn':
                work_thread_pool = int(len(node0_cores) / num_workers)
                self.loader_cores = [
                    list(range(work_thread_pool * i, work_thread_pool * (i + 1)))
                    for i in range(num_workers)
                ]
            else:
                self.loader_cores = node0_cores[:num_workers]

        self._old_worker_init_fn = WorkerInitWrapper(getattr(self, 'worker_init_fn', None))
        try:
            logging.debug(f"{num_workers} data loader workers assigned to CPUs {self.loader_cores}")
            self.worker_init_fn = self._aff_init_fn
            yield
        finally:
            self.worker_init_fn = self._old_worker_init_fn


from collections import OrderedDict
import queue
from contextlib import contextmanager
from threading import Thread
from typing import Dict, Iterable, List, Optional

import torch

from pypipeec import core
from pypipeec.context import NetworkConfig, json_dumps


def get_module_partition(
    module: torch.nn.Sequential,
    rank: int,
    balance: Iterable[int],
) -> torch.nn.Sequential:

    balance = list(balance)

    if len(module) != sum(balance):
        raise ValueError('module and sum of balance have different length '
                         f'(module: {len(module)}, sum of balance: {sum(balance)})')
    if any(x <= 0 for x in balance):
        raise ValueError(
            f'all balance numbers must be positive integer (balance: {balance})')

    j = 0
    layers = OrderedDict()

    for name, layer in module.named_children():
        layers[name] = layer
        if len(layers) == balance[j]:
            # Group buffered layers as a partition.
            if j == rank:
                partition = torch.nn.Sequential(layers)
                return partition
            # Prepare for the next partition.
            layers.clear()
            j += 1

    raise RuntimeError('module and balance mismatch')


def _count_iterable(iterable: Iterable) -> int:
    return sum(
        1 for _ in iterable
    )


def get_base_id(module: torch.nn.Sequential, balance: List[int], rank: int):
    if rank == 0:
        return 0
    new_balance = [sum(balance[0:rank]), sum(balance[rank:])]
    base_id = _count_iterable(
        get_module_partition(module, 0, new_balance).parameters()
    )
    print(f"baseid: {base_id}")
    return base_id


class CheckPointer:
    """Helper class for checkpointing tensors to local/remote shared memory.
    """

    def __init__(
        self,
        checkpointer_type: str = "rs",
    ) -> None:

        self._checkpointer = 0  # pointer
        self._ckpter_type = checkpointer_type
        # N.B. timestamp starts with 1
        self._timestamps: List[int] = []
        self._module: Optional[torch.nn.Module] = None
        self._base_id = 0

        self._streams: Dict[torch.device, torch.cuda.Stream] = {}

        self._to_checkpointer = queue.Queue[bool]()
        self._from_checkpointer = queue.Queue[bool]()

    def _checkpointer_thread(self):
        while True:
            stop = self._to_checkpointer.get()
            if stop:
                return
            result = self.checkpoint_module()
            self._from_checkpointer.put(result)

    def checkpoint_tensor(
        self, parameter: torch.Tensor, tensor_id: int, timestamp: int
    ) -> bool:
        p = parameter.to("cpu")
        succeeded = core.Store(self._checkpointer,
                               p, tensor_id, timestamp)
        return succeeded

    def load_tensor(self, parameter: torch.Tensor, tensor_id: int) -> int:
        return core.Load(self._checkpointer, parameter, tensor_id)

    def checkpoint_module(self) -> bool:
        parameters = []
        # copy tensors to host
        for param in self._module.parameters():
            with torch.cuda.stream(self._streams[param.device]):
                parameters.append(param.to("cpu", non_blocking=True))
        # synchronize
        for stream in self._streams.values():
            if stream is not None:
                stream.synchronize()

        for tensor_id, param in enumerate(parameters):
            if not self.checkpoint_tensor(
                param,
                self._base_id + tensor_id,
                self._timestamps[tensor_id]
            ):
                return False
            self._timestamps[tensor_id] += 1
        return True

    def load_module(self):
        for tensor_id, param in enumerate(self._module.parameters()):
            buffer = torch.zeros_like(param, device="cpu")
            timestamp = self.load_tensor(buffer, self._base_id + tensor_id)
            if timestamp != 0:
                with torch.no_grad():
                    param[:] = buffer[:]
            self._timestamps[tensor_id] = timestamp

    def _start(self, config: NetworkConfig):
        self._checkpointer = core.NewCheckPointer(
            self._ckpter_type, json_dumps(config))

    def checkpoint_module_async(self):
        # copy streams should first sync with the default stream
        for stream in self._streams.values():
            if stream is not None:
                stream.wait_stream(torch.cuda.current_stream())
        # start asynchronous checkpointing
        self._to_checkpointer.put(False)

    def checkpoint_module_wait(self) -> bool:
        return self._from_checkpointer.get()

    def timestamps(self):
        return self._timestamps

    def _shutdown(self, unlink: bool):
        core.Shutdown(self._checkpointer, unlink)

    @contextmanager
    def run_module_context(
        self,
        module: torch.nn.Module,
        base_id: int,
        config: NetworkConfig,
        unlink: bool
    ):
        self._module = module
        self._timestamps = [0 for _ in range(config.BlockNumber)]
        self._base_id = base_id

        # create device copy streams
        for param in module.parameters():
            if param.device not in self._streams:
                device = param.device
                if device.type == "cpu":
                    self._streams[device] = None
                elif device.type == "cuda":
                    self._streams[device] = torch.cuda.Stream(device)
                else:
                    raise ValueError(f"unsupported device type:{device}")

        self._start(config)
        thread = Thread(target=self._checkpointer_thread)
        thread.start()
        yield
        self._to_checkpointer.put(True)
        thread.join()
        self._shutdown(unlink)

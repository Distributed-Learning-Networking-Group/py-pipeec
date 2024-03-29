from contextlib import contextmanager
import queue
from threading import Thread
from time import sleep
from typing import List, Optional
import torch
from viztracer import get_tracer

from pypipeec import core
from pypipeec.context import NetworkConfig, json_dumps


def get_block_number(module: torch.nn.Module):
    return sum(1 for _ in module.parameters())


class CheckPointer:
    """Helper class for checkpointing tensors to local/remote shared memory.
    """

    def __init__(
        self, checkpointer_type: str = "rs"
    ) -> None:

        self._checkpointer = 0  # pointer
        self._ckpter_type = checkpointer_type
        # N.B. timestamp starts with 1
        self._timestamps: List[int] = []
        self._module: Optional[torch.nn.Module] = None
        self._base_id = 0

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
        with get_tracer().log_event("checkpointing"):
            for id, param in enumerate(self._module.parameters()):
                if not self.checkpoint_tensor(param, self._base_id + id, self._timestamps[id]):
                    return False
                else:
                    self._timestamps[id] += 1
            return True

    def load_module(self):
        for id, param in enumerate(self._module.parameters()):
            timestamp = self.load_tensor(param, self._base_id + id)
            self._timestamps[id] = timestamp

    def _start(self, config: NetworkConfig):
        self._checkpointer = core.NewCheckPointer(
            self._ckpter_type, json_dumps(config))

    def checkpoint_module_async(self):
        self._to_checkpointer.put(False)

    def checkpoint_module_wait(self) -> bool:
        return self._from_checkpointer.get()

    def timestamps(self):
        return self._timestamps

    def _shutdown(self, unlink: bool):
        core.Shutdown(self._checkpointer, unlink)

    @contextmanager
    def run_context(self, config: NetworkConfig, unlink: bool):
        self._start(config)
        yield
        self._shutdown(unlink)

    @contextmanager
    def run_module_context(self, module: torch.nn.Module, base_id: int, config: NetworkConfig, unlink: bool):
        self._module = module
        self._timestamps = [0 for _ in range(config.BlockNumber)]
        self._base_id = base_id
        self._start(config)
        thread = Thread(target=self._checkpointer_thread)
        thread.start()
        yield
        self._to_checkpointer.put(True)
        thread.join()
        self._shutdown(unlink)

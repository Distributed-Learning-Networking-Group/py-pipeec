from contextlib import contextmanager
from typing import Optional
import torch

from pypipeec import core
from pypipeec.context import NetworkConfig, json_dumps


class CheckPointer:
    """Helper class for checkpointing tensors to local/remote shared memory.
    """

    def __init__(
        self, checkpointer_type: str = "rs"
    ) -> None:

        self._checkpointer = 0  # pointer
        # N.B. timestamp starts with 1
        self._timestamp = 0
        self._ckpter_type = checkpointer_type

    def checkpoint(self, parameter: torch.Tensor, tensor_id: int) -> bool:
        succeeded = core.Store(self._checkpointer,
                               parameter, tensor_id, self._timestamp)
        return succeeded

    def load(self, parameter: torch.Tensor, tensor_id: int) -> int:
        return core.Load(self._checkpointer, parameter, tensor_id)

    def _start(self, config: NetworkConfig):
        self._checkpointer = core.NewCheckPointer(
            self._ckpter_type, json_dumps(config))

    def _shutdown(self, unlink: bool):
        core.Shutdown(self._checkpointer, unlink)

    @contextmanager
    def run_context(self, config: NetworkConfig, start_timestamp: int, unlink: bool):
        self._timestamp = start_timestamp
        self._start(config)
        yield
        self._shutdown(unlink)

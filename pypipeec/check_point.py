from contextlib import contextmanager
import torch

from pypipeec import core
from pypipeec.context import NetworkConfig, json_dumps


class CheckPointer:
    """Helper class for checkpointing a given module.
        Currently only torch.nn.Sequential is supported.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        id_base: int,
        checkpointer_type: str,
        config: NetworkConfig,
    ) -> None:

        if not isinstance(module, torch.nn.Sequential):
            raise ValueError(
                "module type {} not supported".format(type(module)))

        self._checkpointer = core.NewCheckPointer(
            checkpointer_type, json_dumps(config))
        self._config = config

        self._module = module
        # N.B. timestamp starts with 1
        self._timestamp = 1
        self._id_base = id_base

    def checkpoint(self) -> bool:
        for id_offset, paramter in enumerate(self._module):
            succeeded = core.Store(self._checkpointer, paramter,
                                   self._id_base + id_offset, self._timestamp)
            if not succeeded:
                return False

        self._timestamp += 1
        return True

    @contextmanager
    def context(self):
        yield
        self.shutdown()

    def restore(self) -> bool:
        for id_offset, paramter in enumerate(self._module):
            timestamp = core.Store(self._checkpointer, paramter,
                                   self._id_base + id_offset, self._timestamp)
            # ******************************************
            # Currently allow tensor-wise inconsistency
            # ******************************************
            if timestamp is 0:
                return False
        return True

    def shutdown(self):
        core.Shutdown(self._checkpointer)

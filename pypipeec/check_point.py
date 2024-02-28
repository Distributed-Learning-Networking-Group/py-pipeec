# pylint: disable=C0114,C0115,C0116

from dataclasses import dataclass
import json
from typing import Dict
import torch
import pipeec

# pylint: disable=C0103


@dataclass
class NetworkConfig:
    World_size: int
    Local_rank: int
    Addrs: Dict[int, str] = None

# pylint: enable=C0103


class ModuleKeeper:

    def __init__(self, module: torch.nn.Module, rank: int) -> None:
        self._rank = rank
        self._module = module
        self._keys = {}
        self._cnt = 0

    def _init_keys(self):
        for param in self._module.parameters():
            self._keys[param] = self._assign_key()

    def _assign_key(self):
        ret = (self._rank << 16) + self._cnt
        self._cnt += 1
        if self._cnt > (1 << 16):
            raise RuntimeError(
                "Potential overlaping in keys: too many parameters")
        return ret

    def key(self, value: torch.Tensor):
        return self._keys[value]

# pylint: disable=I1101


class CheckPointer:

    def __init__(self, conf_path: str, module: torch.nn.Module, ft: int) -> None:
        self._conf_path = conf_path
        with open(conf_path, "r", encoding="utf-8") as fp:
            conf_json = json.load(fp)
        conf = NetworkConfig(*conf_json)
        self._conf = conf
        self._keeper = ModuleKeeper(module, conf.Local_rank)
        self._ft = ft
        self._module = module

    def __enter__(self):
        pipeec.Start(self._conf_path, self._ft)
        return self

    def __exit__(self, _, __, ___):
        return pipeec.Shutdown()

    def start(self) -> bool:
        return pipeec.Start(self._conf_path, self._ft)

    def shutdown(self) -> bool:
        return pipeec.Shutdown()

    def store(self, data: torch.Tensor):
        pipeec.Store(data, self._keeper.key(data))

    def load(self, data: torch.Tensor) -> bool:
        return pipeec.Store(data, self._keeper.key(data))

    def store_module(self):
        for param in self._module.parameters():
            self.store(param)

    def load_module(self):
        for param in self._module.parameters():
            self.load(param)

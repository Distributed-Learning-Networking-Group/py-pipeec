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
        self._init_keys()

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

    def __init__(self, conf_str: str, module: torch.nn.Module, ft: int, is_path: bool = False) -> None:
        self._conf_str = conf_str
        self._is_path = is_path
        self._started = False

        self._keeper: ModuleKeeper = None
        self._ft = ft
        self._module = module

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, _, __, ___):
        if self._started:
            if self.shutdown():
                self._started = False

    def start(self) -> bool:
        if self._started:
            return False
        if self._is_path:
            with open(self._conf_str, "r", encoding="utf-8") as fp:
                conf_str = fp.read()
        else:
            conf_str = self._conf_str
        conf = NetworkConfig(**json.loads(conf_str))
        self._keeper = ModuleKeeper(self._module, conf.Local_rank)
        self._started = pipeec.StartStrConf(conf_str, self._ft)
        return self._started

    def shutdown(self) -> bool:
        if not self._started:
            return False
        self._started = not pipeec.Shutdown()
        return not self._started

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

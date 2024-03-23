from dataclasses import dataclass
import json
from typing import Dict


@dataclass
class NetworkConfig:
    World_size: int
    Local_rank: int
    Fault_tolerance: int
    Addrs: Dict[int, str]
    Service: str
    BlockNumber: int


def json_dumps(config: NetworkConfig):
    return json.dumps(config.__dict__)

import json
from dataclasses import dataclass
from typing import Dict


@dataclass
class NetworkConfig:
    WorldSize: int
    LocalRank: int
    FaultTolerance: int
    Addrs: Dict[int, str]
    Names: Dict[int, str]
    Service: str
    BlockNumber: int


def json_dumps(config: NetworkConfig):
    return json.dumps(config.__dict__)

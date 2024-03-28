import torch


def NewCheckPointer(checkpointer_type: str, network_config: str) -> int:
    """Create new checkpointer with type "checkpointer_type", initialize
    the checkpointer with config 'network_config'

    currently supported value of checkpointer types:
        "rs": reedsolomon checkpointer

    network_config is a string presentation of json object, which has fields:

        World_size      int             \n
        Local_rank      int             \n
        Fault_tolerance int             \n
        Addrs           map[int]string  \n
        Service         string          \n
        BlockNumber     int             \n


    Args:
        checkpointer_type (str):  type of the checkpointer 

        network_config (str): json representation of checkpointer configuration 

    Returns:
        int: _description_
    """
    ...


def Load(checkpointer: int, tensor: torch.Tensor, data_id: int) -> int:
    """Load tensor of id 'data_id'

    Args:
        checkpointer (int): pointer to the checkpointer object, created by 'NewCheckPointer' 
        tensor (torch.Tensor): tensor to load from checkpointer 
        data_id (int): id of the given tensor

    Returns:
        int: timestamp of the returned tensor. Return value 0 means
        a failed load operation.
    """
    ...


def Store(checkpointer: int, tensor: torch.Tensor, data_id: int, timestamp: int) -> bool:
    """Store tensor of id 'data_id', with timestamp 'timestamp'

    N.B.: 'timestamp' starts with 1, a zero value means an invalid timestamp

    If this function succeeds, no partial write w.r.t the 'data_id' exists in the system.
    This synchronious behavior in distribtued training setup is on purpose beacause pipeline
    parallelism requires all nodes to be functional during an iteration.

    Args:
        checkpointer (int): pointer to the checkpointer object, created by 'NewCheckPointer' 
        tensor (torch.Tensor): tensor to be checkpointed by the checkpointer 
        data_id (int): id of the given tensor
        timestamp (int): timestamp of the operation

    Returns:
        bool: True if this operation succeeded  
    """
    ...


def Shutdown(checkpointer: int, unlink: bool):
    """Shutdown the checkpointer

    Args:
        checkpointer (int): pointer to the checkpoitner object, created by 'NewCheckPoitner' 
    """
    ...

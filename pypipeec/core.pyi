# pylint: disable=C0103, W0613


def PipeecInitService(local_addr: str) -> int:
    """Init checkpoint service, should be called before any Pipeec* function.  

    Args:
        local_addr (str): tcp address to listen on.

    Returns:
        int: service id representing the launched service. 
    """


def PipeecInitCheckPointContext(svc_id: int, file_name: str, network_config_json: str) -> int:
    """Create new context under service of id svc_id

    Args:
        svd_id (int): service id created by PipeecInitService
        file_name (str): checkpoint file to be transfered/read 
        network_config_json (str): json representation of the context.Networkconfig 

    Returns:
        int: context id specific to this service. 
    """


def PipeecDestroyCheckPointContext(svc_id: int, ctx_id: int):
    """Destroy the context of id ctx_id of service of id svc_id

    Args:
        svc_id (int): service id  
        ctx_id (int): context id to be destroyed 
    """


def PipeecRead(svc_id: int, ctx_id: int):
    """read the file, with service svc_id and context ctx_id

    Args:
        svc_id (int): service id  
        ctx_id (int): context id
    """


def PipeecStartTransfer(svc_id: int, ctx_id: int):
    """encode, transfer service svc_id and context ctx_id

    Args:
        svc_id (int): service id  
        ctx_id (int): context id
    """


def PipeecSuspendTransfer(svc_id: int, ctx_id: int):
    """suspend the transfer task, launched by previous PipeecStartTransfer

    Args:
        svc_id (int): service id  
        ctx_id (int): context id
    """


def PipeecResumeTransfer(svc_id: int, ctx_id: int):
    """resume the transfer task, suspend by previous PipeecSuspendTransfer

    Args:
        svc_id (int): service id  
        ctx_id (int): context id
    """


def PipeecWaitTransfer(svc_id: int, ctx_id: int):
    """wait for the transfer task to finish

    Args:
        svc_id (int): service id  
        ctx_id (int): context id
    """

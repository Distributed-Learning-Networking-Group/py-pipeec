
import torch


class NVTracer:
    """event tracer base on PyTorch's NVIDIA Tools Extension (NVTX)
    """

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda is not available")

    def start_event(self, event_name) -> int:
        return torch.cuda.nvtx.range_push(event_name)

    def end_evnet(self) -> int:
        return torch.cuda.nvtx.range_pop()

    def shutdown():
        """do nothing
        """
        ...

# cpp-extensino 'pypipeec.core' contains dynamic library
# that depends on torch
import torch

torch.multiprocessing.set_start_method(
    "spawn",
    force=True,
)

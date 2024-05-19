from collections import namedtuple
from typing import Callable, Dict, Tuple
import numpy
from pyzfp import compress, decompress  # pylint: disable=E0611,W0611
import torch

NumpyMeta = namedtuple('NumpyMeta', ['shape', 'dtype', 'do_compress'])

_COMPRESS_CHUNK_SIZE = 16 * 1024 * 1024  # 16MB
_COMPRESS_TOLERANCE = 0.0001


def _do_compress(tensor: torch.Tensor):
    if len(tensor.shape) >= 4:
        return False
    return tensor.numel()*tensor.element_size()/1024/1024 >= 64


def _div_ceil(lhs: int, rhs: int):
    return (lhs - 1) // rhs + 1


def _to_cpu_chunks(src: torch.Tensor, wait_for_ready: Callable):
    src = src.detach()
    tensor_nbytes = src.numel() * src.element_size()
    if tensor_nbytes <= _COMPRESS_CHUNK_SIZE:
        return src.to(device="cpu")
    buffer = torch.empty_like(src, device="cpu")
    chunks = _div_ceil(tensor_nbytes, _COMPRESS_CHUNK_SIZE)
    for src_chunk, buffer_chunk in zip(src.chunk(chunks), buffer.chunk(chunks)):
        wait_for_ready()
        buffer_chunk[:] = src_chunk[:]
    return buffer


def _compress_tensor(tensor: torch.Tensor, wait_for_ready: Callable):
    with torch.no_grad():
        do_compress = _do_compress(tensor)
        tensor_numpy: numpy.ndarray = _to_cpu_chunks(
            tensor, wait_for_ready).numpy()
        compressed = tensor_numpy
        if do_compress:
            parallel = True
            compressed = compress(
                tensor_numpy, tolerance=_COMPRESS_TOLERANCE, parallel=parallel)
            compressed = bytes(compressed)
        return compressed, NumpyMeta(tensor_numpy.shape, tensor_numpy.dtype, do_compress)


def compress_state_dict(checkpoint: Dict[str, torch.Tensor], wait_for_ready: Callable):
    checkpoint = checkpoint.copy()
    for key, value in checkpoint.items():
        if isinstance(value, dict):
            value = compress_state_dict(value, wait_for_ready)
        elif isinstance(value, torch.Tensor):
            value = _compress_tensor(value, wait_for_ready)
        checkpoint[key] = value
    return checkpoint


def _decompress_tensor(value: Tuple[numpy.ndarray | bytes, NumpyMeta]):
    tensor_numpy, meta = value
    if meta.do_compress:
        tensor_numpy: numpy.ndarray = decompress(
            tensor_numpy,
            meta.shape,
            meta.dtype,
            _COMPRESS_TOLERANCE
        )
    return torch.from_numpy(tensor_numpy).cuda()


def decompress_state_dict(checkpoint: Dict[str, Tuple[numpy.ndarray, NumpyMeta]]):
    for key, value in checkpoint.items():
        if isinstance(value, dict):
            value = decompress_state_dict(value)
        elif isinstance(value, tuple):
            value = _decompress_tensor(value)
        checkpoint[key] = value
    return checkpoint

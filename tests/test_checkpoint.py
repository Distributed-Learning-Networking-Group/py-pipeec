# pylint: disable=C0113,C0114,C0115,C0116,E1101

import hashlib
import tempfile
import torch
from torchvision.models import resnet50, ResNet50_Weights
from pypipeec.checkpoint import CheckPointer, NetworkConfig


def _test_confs(
    base_port: int,
    num_worker: int,
    fault_tolerance: int,
    service: str,
    block_number: int,
):
    addrs = {
        i: f"localhost:{base_port + i}" for i in range(num_worker)
    }
    confs = {
        i: NetworkConfig(num_worker, i, fault_tolerance, addrs, service, block_number) for i in range(num_worker)
    }
    return confs


def test_checkpoint_tensor_mem():
    confs = _test_confs(0, 8, 3, "mem", 32)
    ckpter = CheckPointer()

    t = torch.randint(0, 2048, (1024,))
    t_test = torch.zeros_like(t)

    with ckpter.run_context(confs[0], 1):
        ckpter.checkpoint(t, 3)
        ckpter.load(t_test, 3)
        assert torch.equal(t, t_test)


def test_checkpoint_resnet50():
    confs = _test_confs(0, 8, 3, "mem", 512)
    ckpter = CheckPointer()

    with ckpter.run_context(confs[0], 1):
        module = resnet50(weights=ResNet50_Weights.DEFAULT)

        # caculate origional module checkpoint sha256 checksum
        with tempfile.TemporaryFile("w+b") as tmp_file:
            torch.save(module, tmp_file)
            digest_origin = hashlib.file_digest(tmp_file, "sha256")

        # checkpoint
        for tensor_id, parameter in enumerate(module.parameters()):
            ckpter.checkpoint(parameter, tensor_id)

        new_module = resnet50()
        # load checkpoint
        for tensor_id, parameter in enumerate(new_module.parameters()):
            ckpter.load(parameter, tensor_id)

        # caculate new module checkpoint sha256 checksum
        with tempfile.TemporaryFile("w+b") as tmp_file:
            torch.save(new_module, tmp_file)
            digest_new = hashlib.file_digest(tmp_file, "sha256")

        assert digest_origin.hexdigest() == digest_new.hexdigest()

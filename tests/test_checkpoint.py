# pylint: disable=C0113,C0114,C0115,C0116,E1101

import hashlib
import tempfile
import torch
from torchvision.models import resnet50, ResNet50_Weights
from pypipeec.checkpoint import CheckPointer, NetworkConfig


def _test_confs(
    name: str,
    base_port: int,
    num_worker: int,
    fault_tolerance: int,
    service: str,
    block_number: int,
):
    names = {
        i: name + f"_{i}" for i in range(num_worker)
    }
    addrs = {
        i: f"localhost:{base_port + i}" for i in range(num_worker)
    }
    confs = {
        i: NetworkConfig(num_worker, i, fault_tolerance, addrs, names, service, block_number) for i in range(num_worker)
    }
    return confs


def test_checkpoint_tensor_mem():
    confs = _test_confs("test_checkpoint_tensor_mem", 0, 8, 3, "mem", 32)
    ckpter = CheckPointer()

    t = torch.randint(0, 2048, (1024,))
    t_test = torch.zeros_like(t)

    with ckpter.run_context(confs[0], True):
        assert ckpter.checkpoint_tensor(t, 3, 4)
        assert ckpter.load_tensor(t_test, 3) == 4
        assert torch.equal(t, t_test)


def test_checkpoint_resnet50():
    confs = _test_confs("test_checkpoint_resnet50", 0, 8, 3, "mem", 1024)
    ckpter = CheckPointer()

    module = resnet50(weights=ResNet50_Weights.DEFAULT)

    with ckpter.run_module_context(module, 0, confs[0], True):

        # caculate origional module checkpoint sha256 checksum
        with tempfile.TemporaryFile("w+b") as tmp_file:
            torch.save(module, tmp_file)
            digest_origin = hashlib.file_digest(tmp_file, "sha256")

        # checkpoint
        ckpter.checkpoint_module()

        # zero_ module parameter
        with torch.no_grad():
            for param in module.parameters():
                param.zero_()

        # load checkpoint
        ckpter.load_module()

        # caculate new module checkpoint sha256 checksum
        with tempfile.TemporaryFile("w+b") as tmp_file:
            torch.save(module, tmp_file)
            digest_new = hashlib.file_digest(tmp_file, "sha256")

        assert digest_origin.hexdigest() == digest_new.hexdigest()

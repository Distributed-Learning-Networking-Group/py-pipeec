# pylint: disable=C0113,C0114,C0115,C0116,E1101

import hashlib
import tempfile
import pytest
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


@pytest.mark.timeout(10)
def test_checkpoint_resnet50_async():
    confs = _test_confs("test_checkpoint_resnet50", 0, 8, 3, "mem", 1024)
    ckpter = CheckPointer()

    module = resnet50(weights=ResNet50_Weights.DEFAULT)

    with ckpter.run_module_context(module, 0, confs[0], True):

        # caculate origional module checkpoint sha256 checksum
        with tempfile.TemporaryFile("w+b") as tmp_file:
            torch.save(module, tmp_file)
            digest_origin = hashlib.file_digest(tmp_file, "sha256")

        # checkpoint
        ckpter.checkpoint_module_async()
        ckpter.checkpoint_module_wait()

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


@pytest.mark.skipif(
    not (torch.cuda.is_available and (torch.cuda.device_count() >= 1)),
    reason="cuda not available"
)
def test_checkpoint_resnet50_cuda():
    confs = _test_confs("test_checkpoint_resnet50", 0, 8, 3, "mem", 1024)
    ckpter = CheckPointer()

    module = resnet50(weights=ResNet50_Weights.DEFAULT).to("cuda")

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

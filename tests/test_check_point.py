# pylint: disable=C0113,C0114,C0115,C0116,E1101

import hashlib
import json
import tempfile
from threading import Event
import threading
import unittest
import torch
from torchvision.models import resnet50, ResNet50_Weights
from pypipeec.check_point import CheckPointer, NetworkConfig

addrs = {
    0: "localhost:7776",
    1: "localhost:7778",
    2: "localhost:7779",
    3: "localhost:7786",
}

confs = {
    0: NetworkConfig(4, 0, addrs),
    1: NetworkConfig(4, 1, addrs),
    2: NetworkConfig(4, 2, addrs),
    3: NetworkConfig(4, 3, addrs),
}


def _get_conf_str(rank: int):
    return json.dumps(confs[rank].__dict__)

# pylint: disable=I1101


def _run_service(rank: int, module: torch.nn.Module, ft: int, stop_event: Event):
    with CheckPointer(_get_conf_str(rank), module, ft):
        stop_event.wait()


# pylint: enable=I1101


class TestCheckPointer(unittest.TestCase):

    def setUp(self):
        self.module = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.ft = 10

    def test_start_stop(self):
        with CheckPointer(_get_conf_str(0), self.module, self.ft) as ckpter:
            for _ in range(3):
                result = ckpter.shutdown()
                if not result:
                    self.fail()
                result = ckpter.start()
                if not result:
                    self.fail()

    def test_store_load(self):
        stop_event = Event()
        for i in range(1, 4):
            t = threading.Thread(target=_run_service, args=(
                i, self.module, self.ft, stop_event), daemon=True)
            t.start()

        with tempfile.TemporaryFile("w+b") as tmp_file:
            torch.save(self.module, tmp_file)
            digest_origin = hashlib.file_digest(tmp_file, "sha256")

        with CheckPointer(_get_conf_str(0), self.module, self.ft) as ckpter:
            ckpter.store_module()

        new_module = resnet50()
        with CheckPointer(_get_conf_str(0), new_module, self.ft) as ckpter:
            ckpter.load_module()
            with tempfile.TemporaryFile("w+b") as tmp_file:
                torch.save(new_module, tmp_file)
                digest = hashlib.file_digest(tmp_file, "sha256")
        self.assertEqual(digest_origin.hexdigest(), digest.hexdigest())
        stop_event.set()
        stop_event.wait()


if __name__ == '__main__':
    unittest.main()

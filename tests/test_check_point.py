# pylint: disable=C0113,C0114,C0115,C0116,E1101

import hashlib
import tempfile
import unittest
import torch
from torchvision.models import resnet50, ResNet50_Weights
from pypipeec.check_point import CheckPointer


class TestCheckPointer(unittest.TestCase):

    def setUp(self):
        self.conf_path = "tests/conf.json"
        self.module = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.ft = 10
        self.checkpointer = CheckPointer(self.conf_path, self.module, self.ft)

    def test_start_stop(self):
        result = self.checkpointer.start()
        if not result:
            self.checkpointer.shutdown()
            self.fail("start")
        result = self.checkpointer.shutdown()
        if not result:
            self.checkpointer.shutdown()
            self.fail("shutdown")
        self.checkpointer.shutdown()

    def test_store_load(self):
        with self.checkpointer:
            self.checkpointer.store_module()
            with tempfile.TemporaryFile("w+b") as tmp_file:
                torch.save(self.module, tmp_file)
                digest_origin = hashlib.file_digest(tmp_file, "sha256")

            new_module = resnet50()
            new_checkpointer = CheckPointer(
                self.conf_path, new_module, self.ft)
            new_checkpointer.load_module()
            with tempfile.TemporaryFile("w+b") as tmp_file:
                torch.save(new_module, tmp_file)
                digest = hashlib.file_digest(tmp_file, "sha256")

            self.assertEqual(digest_origin, digest)


if __name__ == '__main__':
    unittest.main()

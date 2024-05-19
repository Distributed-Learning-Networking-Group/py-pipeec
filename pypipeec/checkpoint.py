import json
from ctypes import c_bool
from multiprocessing import Process
from threading import Condition
from typing import Dict, Optional, Tuple

import torch
from torch import multiprocessing

from pypipeec import config, core
from pypipeec.compress import compress_state_dict, decompress_state_dict


class CheckPointContext:

    def __init__(
            self,
            service_id: int,
            file_name: str,
            network_config: config.NetworkConfig,
            dry_run=False,
    ):
        config_json = json.dumps(network_config.__dict__)
        self.svc = service_id
        self.id = core.PipeecInitCheckPointContext(
            service_id,
            file_name,
            config_json
        )
        self.file_name = file_name
        self.dry_run = dry_run
        self._suspend = multiprocessing.Value(c_bool, False, lock=False)
        self._cv: Optional[Condition] = multiprocessing.Condition()

    def load_cuda(
        self,
        module: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        checkpoint = torch.load(self.file_name)
        module.load_state_dict(decompress_state_dict(checkpoint["parameter"]))
        optimizer.load_state_dict(
            decompress_state_dict(checkpoint["optimizer"]))
        cpu_state, gpu_states = checkpoint["rng"]
        torch.set_rng_state(cpu_state)
        for device, gpu_state in gpu_states.items():
            torch.cuda.set_rng_state(gpu_state, device)

    @staticmethod
    def gather_rng_states():
        cpu_state = torch.get_rng_state()
        gpu_states = {f"cuda:{i}": torch.cuda.get_rng_state(device=i)
                      for i in range(torch.cuda.device_count())}
        return cpu_state, gpu_states

    @staticmethod
    def _do_save_cuda(
        cv_proxy: Condition,
        suspend_proxy: c_bool,
        file_name: str,
        module: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        rng: Tuple[torch.Tensor, Dict[str, torch.Tensor]] = None,
    ):
        def _wait_for_ready():
            with cv_proxy:
                cv_proxy.wait_for(lambda: not suspend_proxy)
        if rng is None:
            rng = CheckPointContext.gather_rng_states()
        checkpoint = {
            "parameter": compress_state_dict(module.state_dict(), _wait_for_ready),
            "optimizer": compress_state_dict(optimizer.state_dict(), _wait_for_ready),
            "rng": rng
        }
        torch.save(checkpoint, file_name,
                   _use_new_zipfile_serialization=False)

    def save_cuda(
        self,
        module: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        rng: Tuple[torch.Tensor, Dict[str, torch.Tensor]] = None,
    ):
        p = Process(target=self._do_save_cuda, args=(
            self._cv,
            self._suspend,
            self.file_name,
            module,
            optimizer,
            rng
        ))
        p.start()
        p.join()

    def check_point(self):
        if not self.dry_run:
            core.PipeecStartTransfer(self.svc, self.id)

    def wait(self):
        if not self.dry_run:
            core.PipeecWaitTransfer(self.svc, self.id)

    def suspend(self):
        with self._cv:
            self._suspend = True
        if not self.dry_run:
            core.PipeecSuspendTransfer(self.svc, self.id)

    def resume(self):
        with self._cv:
            self._suspend = False
            self._cv.notify_all()
        if not self.dry_run:
            core.PipeecResumeTransfer(self.svc, self.id)

    def read(self):
        core.PipeecRead(self.svc, self.id)

    def destroy(self):
        core.PipeecDestroyCheckPointContext(self.svc, self.id)


def init_service(listen_addr: str) -> int:
    return core.PipeecInitService(listen_addr)

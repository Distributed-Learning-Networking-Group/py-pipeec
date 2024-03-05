#include <cstdlib>
#include <iostream>
#include <libpipeec.h>
#include <memory>
#include <string>
#include <torch/extension.h>

namespace pipeec {

void Store(torch::Tensor t, GoInt key) {
  auto &store = t.storage();
  auto src = store.data();
  auto len = store.nbytes();
  ::Store(key, reinterpret_cast<GoUintptr>(src), len);
}

bool Load(torch::Tensor t, GoInt key) {
  auto &store = t.storage();
  auto dst = store.data();
  auto len = store.nbytes();
  auto ret = ::Load(key, reinterpret_cast<GoUintptr>(dst), len);
  return ret == 0;
}

bool StartPath(std::string conf_path, int ft) {
  auto p = const_cast<char *>(conf_path.c_str());
  auto ret = ::StartPath(p, ft);
  return ret == 0;
}

bool StartStrConf(std::string conf_str, int ft) {
  auto p = const_cast<char *>(conf_str.c_str());
  auto ret = ::StartStrConf(p, ft);
  return ret == 0;
}

bool Shutdown() {
  auto ret = ::Shutdown();
  return ret == 0;
}

} // namespace pipeec

PYBIND11_MODULE(pipeec, m) {
  m.def("Store", pipeec::Store, "store tensor");
  m.def("Load", pipeec::Load, "load tensor");
  m.def("StartPath", pipeec::StartPath, "start server");
  m.def("StartStrConf", pipeec::StartStrConf, "start server");
  m.def("Shutdown", pipeec::Shutdown, "start server");
}
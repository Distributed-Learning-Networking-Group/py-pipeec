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

bool Start(std::string conf_path, int ft) {
  auto p = const_cast<char *>(conf_path.c_str());
  auto ret = ::Start(p, ft);
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
  m.def("Start", pipeec::Start, "start server");
  m.def("Shutdown", pipeec::Shutdown, "start server");
}
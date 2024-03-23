#include <cstdint>
#include <libpipeec.h>
#include <torch/extension.h>

namespace pipeec {

auto NewCheckPointer(char *ckpt, char *conf_str) {
  return ::NewCheckPointer(ckpt, conf_str);
}

auto Store(GoUintptr check_pointer, torch::Tensor t, GoInt data_id,
           GoInt timestamp) {
  auto &store = t.storage();
  auto src = store.data();
  auto len = store.nbytes();
  // clang-format off
  return ::Store(check_pointer,
                 data_id, 
                 reinterpret_cast<GoUintptr>(src),
                 static_cast<GoInt64>(len), 
                 timestamp);
  // clang-format on
}

auto Load(GoUintptr check_pointer, torch::Tensor t, GoInt data_id) {
  auto &store = t.storage();
  auto dst = store.data();
  auto len = store.nbytes();
  // clang-format off
  return ::Load(
    check_pointer,
    data_id, 
    reinterpret_cast<GoUintptr>(dst), 
    static_cast<GoInt64>(len)
    );
  // clang-format on 
}

auto Shutdown(std::uintptr_t check_pointer) {
  return ::Shutdown(static_cast<GoUintptr>(check_pointer));
}

} // namespace pipeec

PYBIND11_MODULE(core, m) {
  m.def("NewCheckPointer", pipeec::NewCheckPointer, "store tensor");
  m.def("Store", pipeec::Store, "store tensor");
  m.def("Load", pipeec::Load, "load tensor");
  m.def("Shutdown", pipeec::Shutdown, "start server");
}
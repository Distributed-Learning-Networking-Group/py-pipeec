#include <libpipeec.h>
#include <torch/extension.h>
#include <utility>
#include <type_traits>
#include <pybind11/pybind11.h>

namespace pypipeec {

namespace detail {
template <typename T> struct function_traits {};

template <typename ret_type, typename... arg_types>
struct function_traits<ret_type(arg_types...)> {

  template <typename func> static constexpr auto prepend_op(func &&f) {
    return [f](arg_types... args) {
      pybind11::gil_scoped_release release;
      return f(std::forward<arg_types>(args)...);
    };
  }
};
} // namespace detail

template <typename T> auto no_gil(T&& func) {
  return detail::function_traits<std::remove_reference_t<T>>::prepend_op(func);
}

} // namespace pypipeec


PYBIND11_MODULE(core, m) {
  m.def("PipeecInitService", pypipeec::no_gil(PipeecInitService));
  m.def("PipeecInitCheckPointContext", pypipeec::no_gil(PipeecInitCheckPointContext));
  m.def("PipeecDestroyCheckPointContext", pypipeec::no_gil(PipeecDestroyCheckPointContext));
  m.def("PipeecStartTransfer", pypipeec::no_gil(PipeecStartTransfer));
  m.def("PipeecSuspendTransfer", pypipeec::no_gil(PipeecSuspendTransfer));
  m.def("PipeecResumeTransfer", pypipeec::no_gil(PipeecResumeTransfer));
  m.def("PipeecRead", pypipeec::no_gil(PipeecRead));
  m.def("PipeecWaitTransfer", pypipeec::no_gil(PipeecWaitTransfer));
}
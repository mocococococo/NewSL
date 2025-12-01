// add.cpp
#include <pybind11/pybind11.h>
namespace py = pybind11;

int add(int a, int b) {
    return a + b;
}

int sub(int a, int b) {
    return a - b;
}

int mul(int a, int b) {
    return a * b;
}

int dev(int a, int b) {
    return a / b;
}

PYBIND11_MODULE(culc_module, m) {
    m.def("add", &add, "二つの整数を足す関数");
    m.def("sub", &sub, "二つの整数を引く関数");
    m.def("mul", &mul, "二つの整数を掛ける関数");
    m.def("dev", &dev, "二つの整数を割る関数");
}
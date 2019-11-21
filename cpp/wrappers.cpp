#include <pybind11/pybind11.h>
#include <cstdio>

int add(int i, int j)
{
    printf("C++ being called! %d %d\n", i, j);
    return i + j;
}

PYBIND11_MODULE(kf_cpp, m)
{
    m.doc() = "C++ KF implementation wrappers";  // optional module docstring

    m.def("add", &add, "A function which adds two numbers");
}

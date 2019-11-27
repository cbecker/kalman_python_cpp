#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <cstdio>

#include "kf.h"

int add(int i, int j)
{
    printf("C++ being called! %d %d\n", i, j);
    return i + j;
}

PYBIND11_MODULE(kf_cpp, m)
{
    m.doc() = "C++ KF implementation wrappers";  // optional module docstring

    m.def("add", &add, "A function which adds two numbers");

    pybind11::class_<KF>(m, "KF")
        .def(pybind11::init<double, double, double>())
        .def("predict", &KF::predict)
        .def("update", &KF::update)
        .def_property_readonly("cov", &KF::cov)
        .def_property_readonly("mean", &KF::mean)
        .def_property_readonly("pos", &KF::pos)
        .def_property_readonly("vel", &KF::vel);
}

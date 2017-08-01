#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <lawrap/blas.h>
#include <string>
#include <iostream>

namespace py = pybind11;

py::tuple getJK_np(py::array_t<double> g,
                  py::array_t<double> gg,
                  py::array_t<double> D)
{
    py::buffer_info g_info = g.request();
    py::buffer_info gg_info = gg.request();
    py::buffer_info D_info = D.request();

    if(g_info.ndim != 4)
        throw std::runtime_error("g is not a four-tensor");

    if(D_info.ndim != 2)
        throw std::runtime_error("D is not a matrix");

    if(g_info.shape[0] != D_info.shape[0])
        throw std::runtime_error("Dimensions not match");

    size_t nbas = g_info.shape[0];
    size_t s1 = g_info.strides[0] / sizeof(double);
    size_t s2 = g_info.strides[1] / sizeof(double);
    size_t s3 = g_info.strides[2] / sizeof(double);
    size_t s4 = g_info.strides[3] / sizeof(double);

    const double* g_data = static_cast<double*>(g_info.ptr);
    const double* gg_data = static_cast<double*>(gg_info.ptr);
    const double* D_data = static_cast<double*>(D_info.ptr);
    std::vector<double> J_data(nbas * nbas);
    std::vector<double> K_data(nbas * nbas);

    for(size_t i = 0; i < nbas; i++)
    for(size_t j = 0; j <= i; j++)
    {
        double valJ = LAWrap::dot(nbas * nbas, D_data, 1,
            &g_data[i * s1 + j * s2], 1);
        double valK = LAWrap::dot(nbas * nbas, D_data, 1,
            &gg_data[i * s1 + j * s2], 1);
        J_data[i * nbas + j] = J_data[j * nbas + i] = valJ;
        K_data[i * nbas + j] = K_data[j * nbas + i] = valK;
    }

    py::buffer_info J_buf =
    {
        J_data.data(),
        sizeof(double),
        py::format_descriptor<double>::format(),
        2,
        {nbas, nbas},
        {sizeof(double) * nbas, sizeof(double)}
    };
    py::buffer_info K_buf =
    {
        K_data.data(),
        sizeof(double),
        py::format_descriptor<double>::format(),
        2,
        {nbas, nbas},
        {sizeof(double) * nbas, sizeof(double)}
    };

    return py::make_tuple(py::array_t<double>(J_buf),
                          py::array_t<double>(K_buf));
}

py::tuple getJK_np_Dshift(py::array_t<double> g,
                          py::array_t<double> D)
{
    py::buffer_info g_info = g.request();
    py::buffer_info D_info = D.request();

    if(g_info.ndim != 4)
        throw std::runtime_error("g is not a four-tensor");

    if(D_info.ndim != 2)
        throw std::runtime_error("D is not a matrix");

    if(g_info.shape[0] != D_info.shape[0])
        throw std::runtime_error("Dimensions not match");

    size_t nbas = g_info.shape[0];
    size_t s1 = g_info.strides[0] / sizeof(double);
    size_t s2 = g_info.strides[1] / sizeof(double);
    size_t s3 = g_info.strides[2] / sizeof(double);
    size_t s4 = g_info.strides[3] / sizeof(double);

    const double* g_data = static_cast<double*>(g_info.ptr);
    const double* D_data = static_cast<double*>(D_info.ptr);
    std::vector<double> J_data(nbas * nbas);
    std::vector<double> K_data(nbas * nbas);

#pragma omp parallel for schedule(dynamic)
    for(size_t i = 0; i < nbas; i++)
    for(size_t j = 0; j <= i; j++)
    {
        double valJ = 0., valK = 0.;
        int indJ = i * s1 + j * s2, indK = i * s1 + j * s3;
        for(size_t k = 0; k < nbas; k++)
        for(size_t l = 0; l <= k; l++)
        {
            valJ += 2. * g_data[indJ + k * s3 + l * s4] * D_data[k * nbas + l];
            valK += (g_data[indK + k * s2 + l * s4] +
                g_data[indK + k * s4 + l * s2]) * D_data[k * nbas + l];
        }
        J_data[i * nbas + j] = J_data[j * nbas + i] = valJ;
        K_data[i * nbas + j] = K_data[j * nbas + i] = valK;
    }

    py::buffer_info J_buf =
    {
        J_data.data(),
        sizeof(double),
        py::format_descriptor<double>::format(),
        2,
        {nbas, nbas},
        {sizeof(double) * nbas, sizeof(double)}
    };
    py::buffer_info K_buf =
    {
        K_data.data(),
        sizeof(double),
        py::format_descriptor<double>::format(),
        2,
        {nbas, nbas},
        {sizeof(double) * nbas, sizeof(double)}
    };

    return py::make_tuple(py::array_t<double>(J_buf),
                          py::array_t<double>(K_buf));
}

PYBIND11_PLUGIN(jk)
{
    py::module m("jk", "Hongzhou's basic module");

    m.def("getJK_np", &getJK_np, "homemade get J and K (returned as a tuple)");
    m.def("getJK_np_Dshift", &getJK_np_Dshift,
        "homemade get J and K (returned as a tuple) w/ np.diag(D) shifted");

    return m.ptr();
}

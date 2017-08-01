#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <lawrap/blas.h>
#include <string>
#include <iostream>

namespace py = pybind11;

py::array_t<double> xform_4_np(py::array_t<double> g,
                               py::array_t<double> A)
/*
 * Given a numpy four-tensor g and xform matrix A
 * do the basis set xform
 * [NOTE] A is symmetric.
 */
{
    py::buffer_info g_info = g.request();
    py::buffer_info A_info = A.request();

    if(g_info.ndim != 4)
        throw std::runtime_error("g is not a four-tensor");

    if(A_info.ndim != 2)
        throw std::runtime_error("A is not a matrix");

    if(g_info.shape[0] != A_info.shape[0])
        throw std::runtime_error("Dimensions not match");

    size_t nbas = g_info.shape[0];
    size_t s1 = g_info.strides[0] / sizeof(double);
    size_t s2 = g_info.strides[1] / sizeof(double);
    size_t s3 = g_info.strides[2] / sizeof(double);
    size_t s4 = g_info.strides[3] / sizeof(double);

    const double* g_data = static_cast<double*>(g_info.ptr);
    const double* A_data = static_cast<double*>(A_info.ptr);
    std::vector<double> J_data(nbas * nbas * nbas * nbas);

    std::vector<double> X(nbas * nbas);
    std::vector<double> Y(nbas * nbas);
    std::vector<double> gx(nbas * nbas * nbas * nbas);
    for(size_t p = 0; p < nbas; p++)
    for(size_t q = 0; q <= p; q++)
    {
        const int ind = p * s1 + q * s2, ind2 = q * s1 + p * s2;
        for(size_t r = 0; r < nbas; r++)
        for(size_t s = 0; s <= r; s++)
            // X_rs = V_pqrs
            X[r * nbas + s] = X[s * nbas + r] = g_data[ind + r * s3 + s * s4];
        // Y = X * A
        LAWrap::gemm('N', 'N', nbas, nbas, nbas,
            1., X.data(), nbas, A_data, nbas, 0., Y.data(), nbas);
        // X = A.T * Y
        LAWrap::gemm('T', 'N', nbas, nbas, nbas,
            1., A_data, nbas, Y.data(), nbas, 0., X.data(), nbas);
        for(size_t k = 0; k < nbas; k++)
        for(size_t l = 0; l <= k; l++)
            gx[ind + k * s3 + l * s4] = gx[ind + l * s3 + k * s4] =
            gx[ind2 + k * s3 + l * s4] = gx[ind2 + l * s3 + k * s4] =
                X[k * nbas + l];
    }

    for(size_t k = 0; k < nbas; k++)
    for(size_t l = 0; l <= k; l++)
    {
        const int ind = k * s3 + l * s4, ind2 = l * s3 + k * s4;
        for(size_t p = 0; p < nbas; p++)
        for(size_t q = 0; q <= p; q++)
            // X_pq = gx_pqkl
            X[p * nbas + q] = X[q * nbas + p] = gx[p * s1 + q * s2 + ind];
        // Y = X * A.T
        LAWrap::gemm('N', 'T', nbas, nbas, nbas,
            1., X.data(), nbas, A_data, nbas, 0., Y.data(), nbas);
        // X = A * Y
        LAWrap::gemm('N', 'N', nbas, nbas, nbas,
            1., A_data, nbas, Y.data(), nbas, 0., X.data(), nbas);
        for(size_t i = 0; i < nbas; i++)
        for(size_t j = 0; j <= i; j++)
            gx[i * s1 + j * s2 + ind] = gx[j * s1 + i * s2 + ind] =
            gx[i * s1 + j * s2 + ind2] = gx[j * s1 + i * s2 + ind2] =
            X[i * nbas + j];
    }

    py::buffer_info gx_buf =
    {
        gx.data(),
        sizeof(double),
        py::format_descriptor<double>::format(),
        4,
        {nbas, nbas, nbas, nbas},
        {sizeof(double) * s1, sizeof(double) * s2,
            sizeof(double) * s3, sizeof(double) * s4}
    };

    return py::array_t<double>(gx_buf);
}

PYBIND11_PLUGIN(xform)
{
    py::module m("xform", "Hongzhou's basic module");

    m.def("xform_4_np", &xform_4_np, "O(N^5) version of xform4");

    return m.ptr();
}

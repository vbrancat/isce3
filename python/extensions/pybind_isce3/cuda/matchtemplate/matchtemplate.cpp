#include "matchtemplate.h"

#include "pycuampcor.h"

namespace py = pybind11;

void addsubmodule_cuda_matchtemplate(py::module& m)
{
    py::module m_matchtemplate = m.def_submodule("matchtemplate");

    addbinding_pycuampcor(m_matchtemplate);
}

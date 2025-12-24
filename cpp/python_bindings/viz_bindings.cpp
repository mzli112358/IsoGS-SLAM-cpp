// Python bindings for visualization (using Pybind11)
// Note: This requires Pybind11 to be installed

#ifdef USE_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "viz/online_viz.hpp"
#include "viz/final_viz.hpp"

namespace py = pybind11;

PYBIND11_MODULE(isogs_viz, m) {
    m.doc() = "IsoGS-SLAM Visualization Python Bindings";
    
    // OnlineVisualizer
    py::class_<isogs::OnlineVisualizer>(m, "OnlineVisualizer")
        .def(py::init<>())
        .def("initialize", &isogs::OnlineVisualizer::initialize)
        .def("update_render", &isogs::OnlineVisualizer::updateRender)
        .def("update_loss", &isogs::OnlineVisualizer::updateLoss)
        .def("update_mesh_progress", &isogs::OnlineVisualizer::updateMeshProgress)
        .def("show", &isogs::OnlineVisualizer::show)
        .def("should_close", &isogs::OnlineVisualizer::shouldClose)
        .def("close", &isogs::OnlineVisualizer::close);
    
    // FinalVisualizer
    py::class_<isogs::FinalVisualizer>(m, "FinalVisualizer")
        .def(py::init<>())
        .def("initialize", &isogs::FinalVisualizer::initialize)
        .def("load_and_show", &isogs::FinalVisualizer::loadAndShow)
        .def("show_gaussians", &isogs::FinalVisualizer::showGaussians)
        .def("show_mesh", &isogs::FinalVisualizer::showMesh)
        .def("run", &isogs::FinalVisualizer::run)
        .def("close", &isogs::FinalVisualizer::close);
}
#endif


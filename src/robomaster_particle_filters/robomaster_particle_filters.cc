#include <fast_plate_orbit/init.h>
#include <plate_orbit/init.h>

// pybind11
#include <pybind11/pybind11.h>

PYBIND11_MODULE(robomaster_particle_filters, m) {
  plate_orbit::init(m);
  fast_plate_orbit::init(m);
}

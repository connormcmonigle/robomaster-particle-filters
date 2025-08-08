#include <fast_plate_orbit/init.h>
#include <plate_orbit/init.h>
#include <plate_orbit_v2/init.h>

// pybind11
#include <pybind11/pybind11.h>

PYBIND11_MODULE(robomaster_particle_filters, m) {
  fast_plate_orbit::init(m);
  plate_orbit::init(m);
  plate_orbit_v2::init(m);
}

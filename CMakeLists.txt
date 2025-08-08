cmake_minimum_required (VERSION 3.24)
project (robomaster_particle_filters LANGUAGES CXX CUDA)

include (GNUInstallDirs)

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -ffast-math -Wpedantic -Wall -Wextra -flto -fopenmp -lgomp")

add_subdirectory (deps/pybind11)

pybind11_add_module (
  robomaster_particle_filters
  src/robomaster_particle_filters/robomaster_particle_filters.cc

  src/plate_orbit/init.cc
  src/plate_orbit/particle_filter.cu

  src/fast_plate_orbit/init.cc
  src/fast_plate_orbit/particle_filter.cu

  src/plate_orbit_v2/init.cc
  src/plate_orbit_v2/particle_filter.cu
)

target_compile_features (
  robomaster_particle_filters PRIVATE
  cxx_std_20
)

target_include_directories (
  robomaster_particle_filters PRIVATE
  include
  deps/eigen
  deps/particle-filter
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_compile_definitions (robomaster_particle_filters PRIVATE $<$<COMPILE_LANGUAGE:CXX>: PF_TARGET_OMP>)
target_compile_definitions (robomaster_particle_filters PRIVATE $<$<COMPILE_LANGUAGE:CXX>: THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP>)
target_compile_definitions (robomaster_particle_filters PRIVATE $<$<COMPILE_LANGUAGE:CXX>: THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP>)

target_compile_definitions (robomaster_particle_filters PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: PF_TARGET_CUDA>)
target_compile_definitions (robomaster_particle_filters PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP>)
target_compile_definitions (robomaster_particle_filters PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA>)

set_target_properties (robomaster_particle_filters PROPERTIES CUDA_ARCHITECTURES all-major)
target_compile_options (robomaster_particle_filters PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --expt-relaxed-constexpr
                       --extended-lambda
                       --use_fast_math
                       >)

install (TARGETS robomaster_particle_filters DESTINATION .)

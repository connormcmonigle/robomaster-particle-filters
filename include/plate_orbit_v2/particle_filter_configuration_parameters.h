#pragma once

#include <Eigen/Dense>

namespace plate_orbit_v2 {

struct particle_filter_configuration_parameters {
  float radius_prior;
  float visibility_logit_coefficient;

  float radius_prior_variance_one_plate;
  float radius_prior_variance_two_plates;

  float radius_common_process_variance;
  float radius_offset_process_variance;

  float z_coordinate_common_process_variance;
  float z_coordinate_offset_process_variance;

  float orientation_velocity_prior_variance;
  float orientation_velocity_process_variance;

  Eigen::Vector2f center_velocity_prior_diagonal_covariance;
  Eigen::Vector2f center_velocity_process_diagonal_covariance;
};

}  // namespace plate_orbit_v2

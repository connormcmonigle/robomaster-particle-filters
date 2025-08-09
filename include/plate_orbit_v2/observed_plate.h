#pragma once

#include <thrust/execution_policy.h>

#include <Eigen/Dense>

namespace plate_orbit_v2 {

class observed_plate {
 private:
  Eigen::Vector3f position_;
  Eigen::Vector3f position_diagonal_covariance_;

 public:
  PF_TARGET_ATTRS [[nodiscard]] const Eigen::Vector3f& position() const noexcept { return position_; }

  PF_TARGET_ATTRS [[nodiscard]] const Eigen::Vector3f& position_diagonal_covariance() const noexcept {
    return position_diagonal_covariance_;
  }

  PF_TARGET_ATTRS
  observed_plate(const Eigen::Vector3f& position, const Eigen::Vector3f& position_diagonal_covariance) noexcept
      : position_{position}, position_diagonal_covariance_{position_diagonal_covariance} {}
};

}  // namespace plate_orbit_v2

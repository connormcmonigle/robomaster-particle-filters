#pragma once

#include <pf/config/target_config.h>
#include <thrust/execution_policy.h>

#include <Eigen/Dense>

namespace fast_plate_orbit {

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

}  // namespace fast_plate_orbit

#pragma once

#include <pf/config/target_config.h>
#include <plate_orbit_v2/observed_plate.h>
#include <thrust/optional.h>

#include <Eigen/Dense>

namespace plate_orbit_v2 {

class observation {
 private:
  Eigen::Vector3f observer_position_;
  observed_plate plate_one_;
  thrust::optional<observed_plate> plate_two_;

 public:
  PF_TARGET_ATTRS [[nodiscard]] const Eigen::Vector3f& observer_position() const noexcept { return observer_position_; }
  PF_TARGET_ATTRS [[nodiscard]] const observed_plate& plate_one() const noexcept { return plate_one_; }
  PF_TARGET_ATTRS [[nodiscard]] const thrust::optional<observed_plate>& plate_two() const noexcept { return plate_two_; }

  PF_TARGET_ATTRS observation(const Eigen::Vector3f& observer_position, const observed_plate& plate_one) noexcept
      : observer_position_{observer_position}, plate_one_{plate_one}, plate_two_{thrust::nullopt} {}

  PF_TARGET_ATTRS
  observation(const Eigen::Vector3f& observer_position, const observed_plate& plate_one, const observed_plate& plate_two) noexcept
      : observer_position_{observer_position}, plate_one_{plate_one}, plate_two_{plate_two} {}

  PF_TARGET_ATTRS static observation from_one_plate(
      const Eigen::Vector3f& observer_position,
      const observed_plate& plate_one) noexcept {
    return observation(observer_position, plate_one);
  }

  PF_TARGET_ATTRS static observation from_two_plates(
      const Eigen::Vector3f& observer_position,
      const observed_plate& plate_one,
      const observed_plate& plate_two) noexcept {
    return observation(observer_position, plate_one, plate_two);
  }
};

}  // namespace plate_orbit

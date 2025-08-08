#include <pf/filter/particle_filter.h>
#include <plate_orbit_v2/particle_filter.h>
#include <plate_orbit_v2/particle_filter_configuration.h>

#include <utility>

namespace plate_orbit_v2 {

struct particle_filter::impl : public pf::filter::particle_filter<particle_filter_configuration> {
 public:
  template <typename... Ts>
  impl(Ts&&... ts) : pf::filter::particle_filter<particle_filter_configuration>(std::forward<Ts>(ts)...) {}
};

void particle_filter::impl_deleter::operator()(particle_filter::impl* p_impl) { delete p_impl; }

prediction particle_filter::extrapolate_state(const float& time_offset_seconds) const noexcept {
  return p_impl_->extrapolate_state(time_offset_seconds);
}

void particle_filter::update_state_sans_observation(const float& time_offset_seconds) noexcept {
  p_impl_->update_state_sans_observation(time_offset_seconds);
}

void particle_filter::update_state_with_observation(
    const float& time_offset_seconds,
    const observation& observation_state) noexcept {
  p_impl_->update_state_with_observation(time_offset_seconds, observation_state);
}

particle_filter::particle_filter(
    const std::size_t& number_of_particles,
    const observation& initial_observation,
    const particle_filter_configuration_parameters& params) noexcept
    : p_impl_(new particle_filter::impl(number_of_particles, initial_observation, params)) {}

}  // namespace plate_orbit

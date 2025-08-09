from dataclasses import dataclass
import math
import time
import cv2
import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional, List
from threading import Lock
from robomaster_particle_filters import plate_orbit_v2


ANGULAR_VELOCITY_STEP_SIZE = 4.0
POSITION_DIAGONAL_COVARIANCE = np.array([0.0001, 0.0001, 0.0001])
RADIAL_OFFSETS = [0.05, 0.0, 0.05, 0.0]
Z_OFFSETS = [0.03, -0.03, 0.03, -0.03]
VISIBILITY_ANGLE_THRESHOLD = math.pi / 3.0

@dataclass
class SimulatedRobotState:
    radius: float
    angle: float
    angular_velocity: float
    center: npt.NDArray[np.float64]
    observer: npt.NDArray[np.float64]

    def plate_positions(self) -> List[npt.NDArray[np.float64]]:
        offsets = [(0) * math.pi, (1 / 2) * math.pi,
                   (1) * math.pi, (3 / 2) * math.pi]

        def to_plate(offset, radial_offset, z_offset):
            horizontal_displacement = np.array(
                [math.cos(self.angle + offset), math.sin(self.angle + offset), 0.0])
            vertical_displacement = np.array([0.0, 0.0, z_offset])
            return self.center + (self.radius + radial_offset) * horizontal_displacement + vertical_displacement

        return [to_plate(*args) for args in zip(offsets, RADIAL_OFFSETS, Z_OFFSETS)]

    def visible_plate_positions(self) -> List[npt.NDArray[np.float64]]:
        visible_plates = []
        to_observer = self.observer - self.center
        for plate in self.plate_positions():
            to_plate = plate - self.center
            similarity = np.dot(to_plate, to_observer) / \
                (np.linalg.norm(to_observer) * np.linalg.norm(to_plate))
            if similarity >= math.cos(VISIBILITY_ANGLE_THRESHOLD):
                visible_plates.append(plate)
        return visible_plates

    def step_up_velocity(self):
        self.angular_velocity += ANGULAR_VELOCITY_STEP_SIZE

    def step_down_velocity(self):
        self.angular_velocity -= ANGULAR_VELOCITY_STEP_SIZE

    def update(self, offset_seconds: float):
        self.angle += offset_seconds * self.angular_velocity

    def set_center(self, center: npt.NDArray[np.float64]):
        self.center = center

    def set_observer(self, observer: npt.NDArray[np.float64]):
        self.observer = observer


class InteractivePlateOrbitFilterDemo:
    window_title: str
    h: int
    w: int
    scale: float
    simulated_robot_state: SimulatedRobotState
    number_of_particles: int
    particle_filter_configuration: plate_orbit_v2.ParticleFilterConfigurationParameters
    input_image: npt.NDArray[np.uint8]
    filter_lock: Lock
    latest_update_time: Optional[float]
    filter: Optional[plate_orbit_v2.ParticleFilter]

    def __init__(self, window_title: str, resolution: Tuple[int, int], scale: float, simulated_robot_state: SimulatedRobotState, number_of_particles: int, config: plate_orbit_v2.ParticleFilterConfigurationParameters):
        self.window_title = window_title
        self.h, self.w = resolution
        self.scale = scale
        self.simulated_robot_state = simulated_robot_state
        self.number_of_particles = number_of_particles
        self.particle_filter_configuration = config
        self.input_image = np.zeros((self.h, self.w, 3), np.uint8)
        self.state_lock = Lock()
        self.latest_update_time = None
        self.filter = None
        self.reset = False
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.imshow(window_title, self.input_image)
        cv2.setMouseCallback(window_title, self.on_mouse)

    def _screen_to_world(self, screen_pos: Tuple[int, int]) -> npt.NDArray[np.float64]:
        """Convert screen coordinates to world coordinates"""
        x, y = screen_pos
        x_world = (x / self.w) / self.scale
        y_world = (y / self.h) / self.scale
        return np.array([x_world, y_world, 0.0])

    def _world_to_screen(self, world_pos: npt.NDArray[np.float64]) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        x_world, y_world, _ = world_pos
        x_screen = int(self.w * x_world * self.scale)
        y_screen = int(self.h * y_world * self.scale)
        return x_screen, y_screen

    def _get_observation(self) -> plate_orbit_v2.Observation:
        plates = self.simulated_robot_state.visible_plate_positions()
        assert len(plates) == 1 or len(plates) == 2
        if len(plates) == 1:
            plate_one, = plates
            observation = plate_orbit_v2.Observation.from_one_plate(
                self.simulated_robot_state.observer,
                plate_orbit_v2.ObservedPlate(
                    plate_one, POSITION_DIAGONAL_COVARIANCE),
            )
        elif len(plates) == 2:
            plate_one, plate_two = plates
            observation = plate_orbit_v2.Observation.from_two_plates(
                self.simulated_robot_state.observer,
                plate_orbit_v2.ObservedPlate(
                    plate_one, POSITION_DIAGONAL_COVARIANCE),
                plate_orbit_v2.ObservedPlate(
                    plate_two, POSITION_DIAGONAL_COVARIANCE),
            )
        return observation

    def initalize_filter(self):
        observation = self._get_observation()
        self.filter = plate_orbit_v2.ParticleFilter(
            self.number_of_particles, observation, self.particle_filter_configuration)
        self.latest_update_time = time.monotonic()

    def update_filter(self) -> plate_orbit_v2.Prediction:
        current_time = time.monotonic()
        elapsed = current_time - self.latest_update_time
        self.simulated_robot_state.update(elapsed)
        observation = self._get_observation()
        self.filter.update_state_with_observation(elapsed, observation)
        self.latest_update_time = current_time
        return self.filter.extrapolate_state(0.0)

    def on_mouse(self, event, x: int, y: int, flags, param):
        # Convert screen coordinates to world coordinates
        world_position = self._screen_to_world((x, y))

        with self.state_lock:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.simulated_robot_state.step_up_velocity()
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.simulated_robot_state.step_down_velocity()

            if self.filter is not None and self.latest_update_time is not None:
                self.simulated_robot_state.set_center(world_position)
                _ = self.update_filter()
            else:
                self.simulated_robot_state.set_center(world_position)
                self.initalize_filter()

    def render_loop(self):
        RED = (0, 0, 255)
        BLUE = (255, 0, 0)
        GREEN = (0, 255, 0)
        WHITE = (255, 255, 255)
        CYAN = (255, 255, 0)
        SPACE_KEY = 32

        while True:
            if self.filter is not None and self.latest_update_time is not None:
                with self.state_lock:
                    prediction = self.update_filter()

                self.input_image = np.zeros((self.h, self.w, 3), np.uint8)

                # Render predicted plates
                for plate in prediction.predicted_plates():
                    x, y, _ = plate.position()
                    v_x, v_y, _ = plate.velocity()
                    center = self._world_to_screen(np.array([x, y, 0.0]))
                    end = self._world_to_screen(np.array([x + 0.5 * v_x, y + 0.5 * v_y, 0.0]))
                    cv2.arrowedLine(self.input_image, center,
                                    end, color=GREEN, thickness=5)
                    cv2.circle(self.input_image, center,
                               radius=max(2, int(2 * self.scale)), color=GREEN, thickness=1)

                # Render predicted center and ellipse
                x, y = prediction.center()
                v_x, v_y = prediction.center_velocity()
                center = self._world_to_screen(np.array([x, y, 0.0]))
                end = self._world_to_screen(np.array([x + 0.5 * v_x, y + 0.5 * v_y, 0.0]))
                
                # Scale the radii for display
                radii = (int(self.w * prediction.radius_0() * self.scale),
                        int(self.h * prediction.radius_1() * self.scale))
                
                degrees = (180.0 / math.pi) * prediction.orientation()
                cv2.ellipse(self.input_image, center, radii, degrees,
                            0.0, 360.0, color=WHITE, thickness=3)
                cv2.circle(self.input_image, center, radius=max(8, int(8 * self.scale)),
                           color=WHITE, thickness=-1)
                cv2.arrowedLine(self.input_image, center,
                                end, color=WHITE, thickness=5)

                # Render actual plate positions
                for plate in self.simulated_robot_state.plate_positions():
                    center = self._world_to_screen(plate)
                    cv2.circle(self.input_image, center,
                               radius=max(8, int(8 * self.scale)), color=RED, thickness=-1)

                # Render visible plate positions
                for plate in self.simulated_robot_state.visible_plate_positions():
                    center = self._world_to_screen(plate)
                    cv2.circle(self.input_image, center,
                               radius=max(8, int(8 * self.scale)), color=BLUE, thickness=-1)

                # Render observer position
                center = self._world_to_screen(self.simulated_robot_state.observer)
                cv2.circle(self.input_image, center,
                           radius=max(100, int(100 * self.scale)), color=CYAN, thickness=-1)

            cv2.imshow(self.window_title, self.input_image)

            key = cv2.waitKey(1)
            if key == SPACE_KEY:
                self.input_image = np.zeros((self.h, self.w, 3), np.uint8)

                self.input_image[:, :] = np.array(RED)
                cv2.imshow(self.window_title, self.input_image)
                _ = cv2.waitKey(200)

                with self.state_lock:
                    self.initalize_filter()
                _ = cv2.waitKey(16)


if __name__ == "__main__":
    config = plate_orbit_v2.ParticleFilterConfigurationParameters(
        0.3, # radius_prior
        2.0, # visibility_logit_coefficient
        0.001, # radius_prior_variance_one_plate
        0.005, # radius_prior_variance_two_plate
        0.0005, # radius_common_process_variances
        0.00005, # radius_offset_process_variance
        0.07, # z_coordinate_common_process_variance
        0.007, # z_coordinate_offset_process_variance
        6.0, # orientation_velocity_prior_variance
        3.0, # orientation_velocity_process_variance
        np.array([9.0, 9.0]), # center_velocity_prior_diagonal_covariance
        np.array([6.0, 6.0]), # center_velocity_process_diagonal_covariance
    )

    simulated_robot_state = SimulatedRobotState(
        radius=0.3, angle=0.0, angular_velocity=0.0, center=np.zeros(3), observer=np.zeros(3))
    
    # Scale parameter - values > 1.0 zoom in, values < 1.0 zoom out
    scale = 0.5
    
    demo = InteractivePlateOrbitFilterDemo(
        window_title="demo", 
        resolution=(1024, 1024), 
        scale=scale,
        simulated_robot_state=simulated_robot_state, 
        number_of_particles=(1 << 20), 
        config=config
    )
    demo.render_loop()

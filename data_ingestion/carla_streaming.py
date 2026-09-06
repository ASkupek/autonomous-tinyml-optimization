"""CARLA Telemetry Streaming Engine.

This module connects to the CARLA autonomous driving simulator, spawns autopilot-controlled
vehicles equipped with obstacle sensors, extracts telemetry, and streams state vectors 
in real-time to an Apache Kafka broker topic.
"""

import json
import math
import random
import time
from typing import Any, Dict, List, Optional

import carla
from confluent_kafka import Producer

from config import GLOBAL_CONFIG, GlobalConfig


# Documentation:
# https://docs.confluent.io/platform/current/clients/confluent-kafka-python/html/index.html#pythonclient-producer
# https://github.com/confluentinc/librdkafka/blob/master/CONFIGURATION.md
# https://carla.readthedocs.io/en/latest/python_api/
# https://carla.readthedocs.io/en/latest/ref_sensors/#obstacle-detector

# TODO(v0.2.0): Combine break + throttle
# TODO(v0.2.0): User can select town for testing 
# TODO(v0.2.0): Getting data in a loop is causing below code killing the loop (method: start_streaming) -> At the moment we comment it out and user need to call it
                #finally:
                #self.cleanup()

class CarlaDataStreaming:
    """Handles real-time vehicle telemetry streaming from CARLA to Apache Kafka.

        Manages the lifecycle of CARLA autonomous vehicles, attaches obstacle detection 
        sensors, and streams physical features (speed, acceleration, distance) to a Kafka topic.
    """
    def __init__(self, config: GlobalConfig = GLOBAL_CONFIG, target_map: Optional[str] = None) -> None:
        """Initializes the streaming pipeline with global configuration.

        Args:
            config (GlobalConfig): Object containing network and simulation parameters.
                Defaults to GLOBAL_CONFIG.
            target_map (Optional[str]): Selected world on which we will execute Carla 
                Defaults to None.
        """
        self.config: GlobalConfig = config
        self.target_map: str = target_map
        self.carla_client: Optional[carla.Client] = None
        self.carla_world: Optional[carla.World] = None
        if not self.config.carla.mcu_testing:
            self.kafka_producer: Optional[Producer] = None

        self.vehicles: List[carla.Actor] = []
        self.obstacle_detectors: Dict[int, carla.Actor] = {}
        self.distances: Dict[int, float] = {}

    @staticmethod
    def on_delivery(err: Optional[Exception], msg: Any) -> None:
        """Callback triggered upon Kafka message delivery confirmation.

        Args:
            err (Optional[Exception]): The error object if delivery failed, else None.
            msg (Any): The Kafka message object containing metadata.
        """
        if err is not None:
            print(f"[Kafka Error] Message delivery failed: {err}")

    def connect(self) -> None:
        """Establishes connections to both Apache Kafka Broker and CARLA Simulator.

        Raises:
            Exception: If connection to Kafka or CARLA cannot be established.
        """
        if not self.config.carla.mcu_testing:
            print(f"[Kafka] Connecting to broker at {self.config.kafka.kafka_server}...")
            try:
                producer_config = {
                    'bootstrap.servers': self.config.kafka.kafka_server,
                    'client.id': self.config.kafka.kafka_client_id,
                    'linger.ms': self.config.kafka.kafka_linger_ms,
                    'batch.num.messages': self.config.kafka.kafka_batch_num_messages,
                    'acks': self.config.kafka.kafka_acks,                             
                    'compression.type': self.config.kafka.kafka_compression_type,
                }
                self.kafka_producer = Producer(producer_config)
                print("[Kafka] Producer initialized successfully.")
                
            except Exception as error:
                print(f"[Kafka Error] Failed to initialize Producer: {error}")
                raise error
        
        try:
            self.carla_client = carla.Client(self.config.carla.carla_host, self.config.carla.carla_port)
            self.carla_client.set_timeout(self.config.carla.carla_timeout)
            if not self.config.carla.mcu_testing:
                self.carla_world = self.carla_client.load_world(self.target_map)
            else:
                self.carla_world = self.carla_client.get_world()
            print("[CARLA] Successfully connected to world.")
        except Exception as error:
            print(f"[CARLA Error] Failed to connect to simulator: {error}")
            raise error
    
    def spawn_vehicle(self, model_name: Optional[str]='vehicle.tesla.model3') -> None:
        """Spawns autopilot-controlled vehicles and initializes obstacle sensors.

        Shuffles available map spawn points and configures the CARLA Traffic Manager
        to prevent immediate collisions. Each vehicle gets an attached obstacle sensor.

        Args:
            model_name (str, optional): CARLA blueprint string for the car model. 
                Defaults to 'vehicle.tesla.model3'.
        """

        num_of_cars = self.config.carla.carla_number_vehicles

        carla_blueprint_library = self.carla_world.get_blueprint_library()
        carla_spawn_points = self.carla_world.get_map().get_spawn_points()

        # Shuffle points
        random.shuffle(carla_spawn_points)

        # We set this due to the reason that without the cars were spawn into the map and a lot of them crash
        carla_traffic_manager = self.carla_client.get_trafficmanager(self.config.carla.traffic_manager_port)
        carla_traffic_manager.set_global_distance_to_leading_vehicle(self.config.carla.traffic_manager_distance)

        print(f"[CARLA] Spawning {num_of_cars} autonomous vehicles ({model_name})...")

        while len(self.vehicles) < num_of_cars and len(carla_spawn_points) > 0:
            carla_spawn_point = carla_spawn_points.pop()
            vehicle_bp = carla_blueprint_library.find(model_name)

            try:
                vehicle = self.carla_world.spawn_actor(vehicle_bp, carla_spawn_point)
                if not self.config.carla.mcu_testing:
                    vehicle.set_autopilot(True, carla_traffic_manager.get_port())
                else:
                    vehicle.set_autopilot(False, carla_traffic_manager.get_port())
                self.vehicles.append(vehicle)

                # Set default distance
                self.distances[vehicle.id] = self.config.carla.carla_obstacle_sensor_distance

                # Adding additional sensor
                # Obstacle sensor will look only upfront and not on back/left/right
                obstacle_bp = carla_blueprint_library.find('sensor.other.obstacle')
                obstacle_bp.set_attribute('distance', str(self.config.carla.carla_obstacle_sensor_reach))
                obstacle_bp.set_attribute('hit_radius', str(self.config.carla.carla_obstacle_sensor_hit_radius))
                obstacle_bp.set_attribute('only_dynamics', str(self.config.carla.carla_obstacle_sensor_only_dynamics))

                # Attach the sensor on the vehicle
                sensor_transform = carla.Transform(carla.Location(x=2.0, z=1.0))
                obstacle_sensor = self.carla_world.spawn_actor(obstacle_bp, sensor_transform, attach_to=vehicle)

                obstacle_sensor.listen(lambda event, v_id=vehicle.id: self._on_obstacle_detected(event=event, vehicle_id=v_id))
                self.obstacle_detectors[vehicle.id] = obstacle_sensor

                # Micro-sleep to distribute physics calculations smoothly across server frames
                time.sleep(0.2)

            except Exception as error:
                print(f"[CARLA Warning] Failed to spawn vehicle at point: {error}")
                continue
        print(f"[CARLA] Successfully spawned {len(self.vehicles)}/{num_of_cars} vehicles.")

    def _on_obstacle_detected(self, event: Any, vehicle_id: int) -> None:
        """Asynchronous callback handler capturing distance updates from obstacle sensors.

        Args:
            event (carla.ObstacleDetectionEvent): Physics payload containing distance to collision matrix.
            vehicle_id (int): Unique identifier of the host vehicle.
        """
        self.distances[vehicle_id] = event.distance

    def get_lane_data(self, vehicle) -> tuple[float, float]:
        """Returns the lateral offset from the lane center and the heading error.

        Args:
            vehicle (carla.Vehicle): Vehicle used to determine the current lane
                position and heading.

        Returns:
            tuple[float, float]: A tuple containing:
                - lane_offset (float): Lateral offset from the center of the lane.
                - heading_error (float): Normalized heading error in range [-1.0, 1.0].
        """
        try:
            transform = vehicle.get_transform()
            location = transform.location

            waypoint = self.carla_world.get_map().get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)

            if waypoint is None:
                return 0.0, 0.0

            vehicle_yaw = transform.rotation.yaw
            lane_yaw = waypoint.transform.rotation.yaw

            heading_error = vehicle_yaw - lane_yaw

            while heading_error > 180.0:
                heading_error -= 360.0

            while heading_error < -180.0:
                heading_error += 360.0

            heading_error /= 180.0

            lane_center = waypoint.transform.location
            right_vector = waypoint.transform.get_right_vector()

            dx = location.x - lane_center.x
            dy = location.y - lane_center.y

            lane_offset = (
                dx * right_vector.x +
                dy * right_vector.y
            )

            return lane_offset, heading_error
        except Exception as error:
            print(f"[Streaming Error] Unexpected error during getting lane assist: {error}")
            return 0.0, 0.0

    def get_vehicle_data(self) -> tuple[float, float, float, float, float]:
        """Returns vehicle data required for MCU model testing in the simulator.

        Note:
            Used for close-loop testing.

        Returns:
            tuple[float, float, float, float, float]: State vector tuple containing:
                - speed (float): Magnitude of multi-axis vehicle velocity vector.
                - acceleration (float): Magnitude of acceleration vector.
                - distance (float): Distance to obstacle upfront.
                - lane_offset (float): Lateral offset from lane center.
                - heading_error (float): Normalized vehicle heading error.
        """
        try:
            for vehicle in self.vehicles:
                vel = vehicle.get_velocity()
                acc = vehicle.get_acceleration()
                control = vehicle.get_control()
                
                #calculation of speed and acceleration
                speed = math.hypot(vel.x, vel.y, vel.z)
                acceleration = math.hypot(acc.x, acc.y, acc.z)
                lane_offset, heading_error = self.get_lane_data(vehicle=vehicle)
                return (speed, acceleration, lane_offset, heading_error)
                #return (speed, acceleration, self.distances.get(vehicle.id, self.config.carla.carla_obstacle_sensor_distance), 1.0, lane_offset, heading_error)

        except Exception as error:
            print(f"[Streaming Error] Unexpected error during get vehicle data: {error}")

    def set_carla_data(self, prediction) -> None:
        """Applies MCU model predictions to the vehicle control.
        Note:
            Used for close-loop testing.
        Args:
            input_data (List[float]): Prediction array containing [throttle, brake, steer] values.

        """
        try:
            # At the momet due to the teaching problem just steering is used
            for vehicle in self.vehicles:
                control = carla.VehicleControl()

                raw_steer = float(prediction[0])

                control.steer = raw_steer
              
                control.throttle = 0.30
                #control.brake = raw_brake
                vehicle.apply_control(control)
        except Exception as error:
            print(f"[Streaming Error] Unexpected error during set vehicle data: {error}")
            

    def start_streaming(self) -> None:
        """Enters infinite telemetry extraction loop, dispatching state vectors to Kafka.

        Extracts multi-axis velocity, acceleration, and manual control logs, computes 
        scalar magnitudes, and fires JSON payloads into the pipeline telemetry stream.
        """

        sleep_time = 1.0 / self.config.pipeline.streaming_frequency_hz
        if not self.config.carla.mcu_testing:
            initial_time = time.time()
        try:
            while True:
                if not self.config.carla.mcu_testing:
                    if (time.time() - initial_time) >= self.config.carla.time_for_town_train:
                        print(f"\n[Streaming] Reached target time limit of {self.config.carla.time_for_town_train / 60:.1f} minutes.")
                        break

                timestamp = time.time()
                for vehicle in self.vehicles:
                    if not vehicle.is_alive:
                        continue

                    vel = vehicle.get_velocity()
                    acc = vehicle.get_acceleration()
                    control = vehicle.get_control()

                    #calculation of speed and acceleration
                    speed = math.hypot(vel.x, vel.y, vel.z)
                    acceleration = math.hypot(acc.x, acc.y, acc.z)
                    lane_offset, heading_error = self.get_lane_data(vehicle=vehicle)
                    message = {
                        "timestamp": timestamp,
                        "vehicle_id": vehicle.id,
                        "speed": speed,
                        "acceleration": acceleration,
                        "throttle": control.throttle,
                        "brake": control.brake,
                        "steer": control.steer,
                        "distance": self.distances.get(vehicle.id, self.config.carla.carla_obstacle_sensor_distance),
                        "friction": 1.0,
                        "lane_offset": lane_offset,
                        "heading_error": heading_error                
                    }

                    if self.kafka_producer is not None:
                        payload = json.dumps(message).encode('utf-8')
                        self.kafka_producer.produce(
                            self.config.kafka.kafka_topic, 
                            value=payload,
                            on_delivery = self.on_delivery)
                # Clear network buffer
                if self.kafka_producer is not None:
                    self.kafka_producer.poll(0)

                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n[Streaming] User interruption detected (Ctrl+C). Stopping loop...")
        except Exception as error:
            print(f"[Streaming Error] Unexpected error during streaming: {error}")
        #finally:
            #self.cleanup()

    def cleanup(self) -> None:
        """Gracefully purges execution environment, destroying CARLA actors and flushing Kafka."""
        print("[Cleanup] Terminating sensors and vehicle actors...")

        for sensor in self.obstacle_detectors.values():
            if sensor is not None and sensor.is_alive:
                try:
                    sensor.stop()
                    sensor.destroy()
                except Exception as error:
                    print(f"[Cleanup Warning] Error destroying sensor: {error}")

        for vehicle in self.vehicles:
            if vehicle is not None and vehicle.is_alive:
                try:
                    vehicle.destroy()
                except Exception as error:
                    print(f"[Cleanup Warning] Error destroying vehicle {vehicle.id}: {error}")
                    pass
        self.vehicles.clear()
        self.obstacle_detectors.clear()
        self.distances.clear()
        if not self.config.carla.mcu_testing:
            if self.kafka_producer is not None:
                print("[Kafka] Flushing remaining telemetry messages...")
                self.kafka_producer.flush()
                
                #Old versions didnt have close
                if hasattr(self.kafka_producer, "close"):
                    self.kafka_producer.close()
        print("[Cleanup] Cleanup complete.")


"""MCU Hardware-in-the-Loop (HIL) Execution & Closed-Loop Test Engine.

This module manages real-time serial communication with an NXP microcontroller (MCU),
sending scaled vehicle telemetry vectors and receiving 32-bit floating-point control 
predictions. Supports both offline dataset batch evaluation and live CARLA closed-loop control.
"""

from collections import deque
import time
from typing import Optional, Tuple
import joblib
from nicegui.html import data
import numpy as np
import pandas as pd
import torch
from config import GlobalConfig, GLOBAL_CONFIG
import serial
import struct
import os
import matplotlib.pyplot as plt

from ml_pipeline.models import AutonomousDriving
from data_ingestion.carla_streaming import CarlaDataStreaming


# TODO(v0.2.0): When sending to the MCU we are using fixed values also checking in receiving we hardcoded 12, this need to be updated.
# TODO(v0.2.0): Bottle neck is sendng the data to MCU with such approach. Need to be check if this we can speed up.
# TODO(v0.2.0): test_on_dataset need to be tested and rewritten to output better data.
class ExecuteOnMCU:
    """Executes and evaluates a trained TinyML model directly on the MCU.

        The class supports two evaluation modes:

        1. Dataset-based testing:
        Sends a predefined test dataset through the MCU model and compares
        the MCU predictions against the expected target values.

        2. Closed-loop CARLA testing:
        Continuously retrieves vehicle telemetry from CARLA, sends the input
        features to the MCU, receives the model prediction, and applies the
        prediction back to the simulated vehicle.
        """
    def __init__(self, config: GlobalConfig = GLOBAL_CONFIG, model_rank: int = 1, close_loop_test: bool = False, carla_streaming = None) -> None:
        """Initializes the MCU execution environment.

        Args:
            config (GlobalConfig, optional):
                Global project configuration.
            model_rank (int, optional):
                Rank of the exported model to evaluate.
                Defaults to 1.
            carla_streaming (Optional[CarlaDataStreaming], optional): Injected CARLA streaming instance 
                for closed-loop testing. If None, instantiates CarlaDataStreaming on demand.

        Raises:
            Exception:
                If dataset, scaler, CARLA, or other initialization fails.
        """
        try:
            self.config: GlobalConfig = config
            self.model_rank = model_rank

            # Load the dataset and  return separated values for x and y
            (self.X_test_scaled, self.y_test, self.scaler) = self.load_dataset()

            if close_loop_test:
                self.carla: Optional[CarlaDataStreaming] = carla_streaming or CarlaDataStreaming(config=self.config, target_map="Town02")
                self.carla.connect()
                self.carla.spawn_vehicle()

                
        except Exception as error:
            print(f"[Execute on MCU Error] Failed during initialization: {error}")
            raise error

    def close_loop_testing_CARLA(self) -> None:
        """Runs closed-loop MCU model evaluation inside CARLA.

        CARLA provides the current vehicle telemetry. The telemetry is
        transformed using the same scaler used during training and sent
        to the MCU. The MCU prediction is then applied directly to the
        simulated vehicle.

        Raises:
            Exception: Re-raises any execution failure occurring during serial streaming or vehicle control.
        """
        try:

            _, _, scaler = self.load_dataset()
            
            window_size = self.config.pipeline.window_size
            input_window = deque(maxlen=10)
            model = AutonomousDriving(layer_structure=[82,105,57,29], activation_type="relu")
            model.load_state_dict(torch.load("ml_pipeline/exported_models/model_rank_1.pt"))
            model.eval()
            
            with serial.Serial(self.config.port_configuration.port_name, baudrate=self.config.port_configuration.baud_rate, timeout=self.config.port_configuration.timeout) as connection:
                while True:
                    t0 = time.perf_counter()
                    carla_data = self.carla.get_vehicle_data()
                    t1 = time.perf_counter()

                    input_window.append(carla_data)
                    if len(input_window) < window_size:
                        steering = 0.0
                        continue

                    # Construct window dataframe from historical records
                    window = np.array(input_window, dtype=np.float32)
                    
                    # Flatten the window features into a single continuous input vector row

                    X_scaled = scaler.transform(window)
                    t2 = time.perf_counter()
                    #X_scaled = np.clip(X_scaled, -3.0, 3.0)
                    model_input = torch.from_numpy(X_scaled).unsqueeze(0)
                    with torch.no_grad():
                        output = model(model_input)
                    t3 = time.perf_counter()


                    current_input = X_scaled[-1]
                    
                    received = self._send_to_mcu(connection=connection, inputs=current_input)
                    print(f"[Execute on MCU] Received from MCU: {received} | PT model prediction: {output}")
                    t4 = time.perf_counter()

                    #print(
                    #    f"CARLA={(t1-t0)*1000:.2f} ms | "
                    #    f"PREP={(t2-t1)*1000:.2f} ms | "
                    #    f"PT={(t3-t2)*1000:.2f} ms | "
                    #    f"MCU={(t4-t3)*1000:.2f} ms"
                    #)
                    if received is not None:
                        self.carla.set_carla_data(received)

        except KeyboardInterrupt:
            print("\n[Execute on MCU] Closed-loop CARLA testing stopped.")

        except Exception as error:
            print(f"[Execute on MCU Error] Closed-loop CARLA testing failed: {error}")
            raise error

    def _send_to_mcu(self, connection: serial.Serial, inputs: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Sends input features to the MCU and receives model predictions.

        The input features are serialized as 32-bit floating-point values
        and transmitted over the configured serial connection. The MCU
        performs inference and returns three 32-bit floating-point values
        corresponding to the configured target outputs.

        Args:
            connection (serial.Serial):
                Active serial connection to the MCU.
            inputs (np.ndarray):
                Scaled input feature vector.

        Returns:
            Optional[tuple[float, float, float]]:
                MCU predictions as (throttle, brake, steer), or None if
                the expected response was not received.

        Raises:
            Exception:
                If serial communication fails.
        """
        try:
            header = bytes([0xAA, 0xBB])
            payload = struct.pack("<4f",*inputs.astype(np.float32))
            packet = header + payload
            connection.write(packet)
            connection.flush()

            response = connection.read(4)

            if len(response) != 4:
                print(f"[Execute on MCU Warning] Expected 4 bytes, received {len(response)}.")
                return None

            return struct.unpack("<1f", response)

        except Exception as error:
            print(f"[Execute on MCU Error] MCU communication failed: {error}")
            raise error

    def test_on_dataset(self) -> None:
        """Evaluates MCU model predictions against the dataset test set.

        Each scaled test sample is transmitted to the MCU. The MCU performs
        inference and returns three floating-point output values corresponding
        to the configured target outputs.

        Raises:
            Exception: Re-raises any error encountered during dataset iteration or metric computation.
        """  
        try:  
            X_test_scaled, y_test, _ = self.load_dataset()
            window_size = self.config.pipeline.window_size
            mcu_outputs = []
            with serial.Serial(self.config.port_configuration.port_name, baudrate=self.config.port_configuration.baud_rate, timeout=self.config.port_configuration.timeout) as connection:
                for i in range(0, len(y_test)):
                    received = self._send_to_mcu(connection=connection,inputs=X_test_scaled[i])

                    if received is not None:
                        mcu_outputs.append(received)

            mcu_preds_matrix = np.array(mcu_outputs)
            mse = float(np.mean((mcu_preds_matrix - y_test) ** 2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(mcu_preds_matrix - y_test)))
            print(f"[Execute on MCU] Evaluation Results -> MSE: {mse:.6f} | RMSE: {rmse:.6f} | MAE: {mae:.6f}")
        except Exception as error:
            print(f"[Execute on MCU Error] Dataset evaluation failed: {error}")
            raise error

       
    def load_dataset(self) -> Tuple[np.ndarray, pd.DataFrame, object]:
        """Loads, splits, and scales the dataset for MCU evaluation.

        The same input feature ordering defined by ``input_features`` is used
        when constructing the model input matrix. The scaler previously fitted
        during training is reused so that the MCU receives data using the
        exact same scaling parameters.

        Returns:
            Tuple[np.ndarray, pd.DataFrame, object]:
                Scaled test inputs, unscaled test targets, and fitted scaler.

        Raises:
            FileNotFoundError:
                If the dataset or scaler cannot be found.
            ValueError:
                If required dataset columns are missing.
        """
        try:
            if not os.path.isfile(self.config.pipeline.export_test_data):
                raise FileNotFoundError(f"Dataset file does not exist at: {self.config.pipeline.export_test_data}")
            # Load the dataset
            test_data = np.load(self.config.pipeline.export_test_data)
            X_test_scaled = test_data["X_test_scaled"]
            y_test = test_data["y_test"]
            scaler = joblib.load(self.config.pipeline.scaler_path)

            print(f"[Execute on MCU] Raw records loaded.")
            return (X_test_scaled, y_test, scaler)
        except Exception as error:
            print(f"[Execute on MCU] following error is raised during the processing of dataset {error}")
            raise
"""Export Engine for NAS Top Models.
This module processes pymoo optimization results, identifies the Top N non-dominated
models from the Pareto front, trains/refines them, and exports both metadata (.json), PyTorch weights (.pt), tfLite and .h files for deployment on MCXN947.
Note: Due to a lot of problems with the onnx2ft library, we are using a custom approach to convert PyTorch models to TensorFlow and then to TensorFlow Lite and from there to a C header file.
In future we will try to use onnx2ft library to achiteve this, but for now we will use a custom approach.
Due to that we add a check function that will use testdata and compare the outputs of the PyTorch model, TensorFlow model and TensorFlow Lite model to make sure that the conversion is correct and that the models are equivalent.
"""

# IMPORTANT NOTE:
# Due to mixing 2 libraries: torch and tensorflow we got semantic error -> known issue in AI community
# The fastest solution to continue our development was to separate the for loop where we are saving the models and 
# perform silent include of tensorflow library
# Better solution is to use multiprocessing and we perform all tensorflow actions there
# To fully avoid this problem, we would need to use onnx etc.
# Option number 2 is the one we will most probably update in next releases

import json
import os
from typing import Any, Callable, Dict, Generator, List
import numpy as np
from ml_pipeline.models import AutonomousDriving
import torch
from torch.utils.data import DataLoader
from config import GlobalConfig, GLOBAL_CONFIG
from .train import  ModelsTraining

# TODO(v0.2.0): act_map = {0: "relu", 1: "tanh", 2: "sigmoid"} -> This need to be updated or taken from the conf file
# TODO(v0.2.0): Isolate Keras/TensorFlow model conversion into a separate sub-process 
#               (e.g., via Python multiprocessing) to resolve PyTorch and TensorFlow 
#               C++ runtime (libgomp/OpenMP) import conflicts cleanly.
# TODO(v0.2.0): Implementation of this class need to be generic and we need to take the .pt files diferently than as we do now (res_F) this means that we cannot test this function without running complete stack!
# TODO(v0.2.0):  def verify_models need to be updated, since at the moment is not correct.
# TODO(v0.2.0): Copy weights function need to be updates especially for layer norm need to be added there
# TODO(v0.2.0): Keras is always having first layer as GRU, if we change this, also here need to be changed
class ExportEngine:

    """Engine to process and export top models from NAS optimization results.

    ```
    This class handles the extraction of top-performing models from the Pareto front,
    trains/refines them, and exports their metadata and weights for deployment.
    """

    def __init__(self, config: GlobalConfig = GLOBAL_CONFIG, export_dir: str = "ml_pipeline/exported_models", trainer_factory: Callable[[List[int], str, bool], ModelsTraining] = None) -> None:
        """Initializes the ExportEngine with the given configuration.

        Args:
            config (GlobalConfig): Configuration object containing paths and parameters. Defaults to GLOBAL_CONFIG.
            export_dir (str): Directory where exported models will be saved. Defaults to "ml_pipeline/exported_models".
            trainer_factory (Callable): Factory function returning an initialized ModelsTraining object.
        """
        self.config: GlobalConfig = config
        self.export_dir: str = export_dir

        self.trainer_factory: Callable[[List[int], str, bool], ModelsTraining] = trainer_factory
        # Create the export directory if it doesn't exist
        os.makedirs(self.export_dir, exist_ok=True)

    def verify_models(self, pytorch_model: torch.nn.Module, tf_model: Any, tflite_path: str, validation_loader: DataLoader) -> Dict[str, float]:
        '''Verifies the equivalence of PyTorch, TensorFlow, and TFLite models using test data.
        Args:
            pytorch_model (torch.nn.Module): The PyTorch model to verify.
            tf_model (tf.keras.Model): The TensorFlow model to verify.
            tflite_path (str): Path to the TFLite model file.
            validation_loader (DataLoader): DataLoader for validation data. Defaults to None.

        Returns:
            dict: A dictionary containing verification results.
        '''
        import tensorflow as tf
        try:
            # We switch the model to evaluation mode to disable dropout and batch normalization layers
            pytorch_model.eval()

            # Load of the TFLite model and allocate tensors
            # Documentation: [https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter)
            interpreter = tf.lite.Interpreter(model_path=tflite_path, experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
            interpreter.allocate_tensors()

            # We get metadata about the input and output tensors of the TFLite model
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()


            pt_tf_errors = []
            tf_tflite_errors = []
            pt_outputs = []
            tf_outputs = []
            tflite_outputs = []
            max_batches = 200
            for batch_idx, (inputs, _) in enumerate(validation_loader):
                if batch_idx >= max_batches:
                    break
                with torch.no_grad():

                    # We send the inputs the the model, load results into cpu and convert to numpy array
                    pt_output = pytorch_model(inputs).cpu().numpy()
                    pt_outputs.append(pt_output)
                    # We load the inputs to the TensorFlow model, and get the output as numpy array
                    tf_output = tf_model(inputs.cpu().numpy(),training=False,).numpy()
                    tf_outputs.append(tf_output)
                    # We set the input tensor for the TFLite model
                    # It expect the input at the index of the input tensor, and the input data as a numpy array

                    interpreter.resize_tensor_input(
                        input_details[0]["index"],
                        inputs.cpu().numpy().shape,
                    )

                    interpreter.allocate_tensors()

                    # Refresh tensor metadata after allocation
                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()

                    # Quantization is a technique to reduce the model size and inference time by converting floating point numbers to integers. The scale and zero point are used to map the integer values back to floating point values.
                    # Quantization is like this:
                    # Exampe: input_scale = 0.1, input_zero = 128
                    input_scale, input_zero = input_details[0]["quantization"]
                    output_scale, output_zero = output_details[0]["quantization"]

                    # We check if scale is grater than 0, in this case we calculate the quantized input for TFLite model, otherwise we use the original input
                    if input_scale > 0:
                        tflite_input = np.round(
                            inputs.cpu().numpy() / input_scale + input_zero
                        )

                        tflite_input = np.clip(
                            tflite_input,
                            -128,
                            127
                        ).astype(np.int8)
                    else:
                        tflite_input = inputs.cpu().numpy().astype(np.float32)

                    interpreter.set_tensor(
                        input_details[0]["index"],
                        tflite_input,
                    )
                    # Start the model
                    interpreter.invoke()
                    # Invoke will run the model and we can get the output tensor at the index of the output tensor
                    tflite_output = interpreter.get_tensor(output_details[0]["index"])

                    # We check if we were using int8 quantization and in this case, we transform the output back to the float32
                    if output_scale > 0:
                        tflite_output = (tflite_output.astype(np.float32) - output_zero) * output_scale
                    tflite_outputs.append(tflite_output)

            pt_output = np.concatenate(pt_outputs, axis=0)
            tf_output = np.concatenate(tf_outputs, axis=0)
            tflite_output = np.concatenate(tflite_outputs, axis=0)       
            # We calculate the absolute errors between the outputs of the different models
            pt_tf_errors = np.abs(pt_output - tf_output)
            # We calculate the absolute errors between the outputs of the different models
            tf_tflite_errors = np.abs(tf_output - tflite_output)

            # We return a dictionary with the mean absolute error, root mean square error and max error for both comparisons
            results = {
                "pt_tf_mae": float(np.mean(pt_tf_errors)),
                "pt_tf_rmse": float(np.sqrt(np.mean(pt_tf_errors ** 2))),
                "pt_tf_max": float(np.max(pt_tf_errors)),
                "tf_tflite_mae": float(np.mean(tf_tflite_errors)),
                "tf_tflite_rmse": float(np.sqrt(np.mean(tf_tflite_errors ** 2))),
                "tf_tflite_max": float(np.max(tf_tflite_errors)),
            }
            return results
        except Exception as error:
            print(f"[Export Engine] Verification failed: {error}")
            raise error


    def representative_data_gen(self, validation_loader:DataLoader) -> Generator[List[np.ndarray], None, None]:
        """This generator yields representative samples from the validation loader for TFLite quantization.
        It is used to calibrate the quantization parameters of the TFLite model. Quantization is a technique to reduce the model size and inference time by converting floating point numbers to integers.&#x20;
        The representative dataset is used to determine the range of values for each layer in the model, which is then used to calculate the scale and zero point for quantization.
        Args:
            validation loader (DataLoader): DataLoader for validation data. Defaults to None.
        Yields:&#x20;
            Generator[List[np.ndarray], None, None]: Generator yielding representative samples for TFLite quantization.
        """

        # We are using the steering ranges to select representative samples for TFLite quantization. The ranges are defined as follows below.
        # This is useful because we want to make sure that the model is well calibrated for all steering angles, especially for sharp turns and recovery maneuvers.

        try:
            # Collect the complete validation dataset
            all_inputs = []
            all_targets = []

            for inputs, targets in validation_loader:
                all_inputs.append(inputs.cpu())
                all_targets.append(targets.cpu())

            all_inputs = torch.cat(all_inputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            # Absolute steering values
            abs_targets = torch.abs(all_targets)

            # Define steering ranges
            masks = [
                abs_targets <= 0.01,
                (abs_targets > 0.01) & (abs_targets <= 0.05),
                (abs_targets > 0.05) & (abs_targets <= 0.10),
                abs_targets > 0.10,
            ]

            # Number of calibration samples per group
            samples_per_group = [
                5000,
                2000,
                2000,
                1000,
            ]

            selected_indices = []

            for mask, num_samples in zip(masks, samples_per_group):

                indices = torch.where(mask)[0]

                if len(indices) == 0:
                    continue

                # Do not request more samples than available
                num_samples = min(num_samples, len(indices))

                # Randomly select samples from this steering range
                permutation = torch.randperm(len(indices))[:num_samples]

                selected_indices.append(indices[permutation])

            # Combine all selected samples
            selected_indices = torch.cat(selected_indices)

            # Shuffle the final representative dataset
            selected_indices = selected_indices[
                torch.randperm(len(selected_indices))
            ]

            # Yield samples
            for index in selected_indices:

                sample = all_inputs[index]

                yield [
                    sample.unsqueeze(0)
                        .numpy()
                        .astype(np.float32)
                ]

        except Exception as error:
            print(f"[Export Engine] representative_data_gen failed: {error}")
            raise error

    def copy_weights(self, pytorch_model: torch.nn.Module, tf_model: Any) -> None:
        """Copies the weights from a PyTorch model to a TensorFlow model.
        Args:
            pytorch_model (torch.nn.Module): The source PyTorch model.
            tf_model (Any): The target TensorFlow model.
        """
        import tensorflow as tf
        try:
            pt_gru = pytorch_model.gru
            tf_gru = tf_model.get_layer("gru")

            weight_ih = pt_gru.weight_ih_l0.detach().cpu().numpy()
            weight_hh = pt_gru.weight_hh_l0.detach().cpu().numpy()

            bias_ih = pt_gru.bias_ih_l0.detach().cpu().numpy()
            bias_hh = pt_gru.bias_hh_l0.detach().cpu().numpy()

            # PyTorch gate order: reset, update, new
            # TensorFlow gate order: update, reset, new

            ih_r, ih_z, ih_n = np.split(weight_ih, 3, axis=0)
            hh_r, hh_z, hh_n = np.split(weight_hh, 3, axis=0)

            b_ih_r, b_ih_z, b_ih_n = np.split(bias_ih, 3)
            b_hh_r, b_hh_z, b_hh_n = np.split(bias_hh, 3)

            kernel = np.concatenate(
                [ih_z.T, ih_r.T, ih_n.T],
                axis=1
            )

            recurrent_kernel = np.concatenate(
                [hh_z.T, hh_r.T, hh_n.T],
                axis=1
            )

            bias = np.stack(
                [
                    np.concatenate([b_ih_z, b_ih_r, b_ih_n]),
                    np.concatenate([b_hh_z, b_hh_r, b_hh_n])
                ],
                axis=0
            )

            tf_gru.set_weights([
                kernel,
                recurrent_kernel,
                bias
            ])
            # Get only the hidden Dense layers, excluding the output layer.
            tf_hidden_dense = [
                layer for layer in tf_model.layers
                if isinstance(layer, tf.keras.layers.Dense) and layer.name != "steer_dense"
            ]
            # We get all nn.Linear
            pt_linear_layers = [
                        module for module in pytorch_model.module_list
                        if isinstance(module, torch.nn.Linear)
            ]

            # Hidden layers are all - the last one
            pt_hidden_linear = pt_linear_layers[:-1]
            # Get only the last layer - output one
            pt_last_layer = pt_linear_layers[-1]

            # Copy weights and biases from PyTorch to TensorFlow.
            for pt_layer, tf_layer in zip(pt_hidden_linear, tf_hidden_dense):
                weights = pt_layer.weight.detach().cpu().numpy().T
                bias = pt_layer.bias.detach().cpu().numpy()
                tf_layer.set_weights([weights, bias])

            # We just copy the weights for last layer
            pt_w = pt_last_layer.weight.detach().cpu().numpy().T
            pt_b = pt_last_layer.bias.detach().cpu().numpy()

            tf_model.get_layer("steer_dense").set_weights([pt_w, pt_b])
        except Exception as error:
            print(f"[Export Engine] copying of weights failed: {error}")
            raise error

    def export_tflite(self, tf_model: Any, rank: int, validation_loader: DataLoader) -> str:
        """Converts a Keras TensorFlow model to a Full Integer (INT8) TFLite format
        and saves it to disk.

        Args:
            tf_model (Any): Trained TensorFlow Keras model.
            rank (int): Rank/index of the model architecture.
            validation_loader (Optional[DataLoader]): PyTorch DataLoader providing calibration samples.

        Returns:
            str: Path to the saved .tflite model file.
        """
        try:
            import tensorflow as tf
            # Build the target path
            tflite_path = os.path.join(
                self.export_dir,
                f"model_rank_{rank}.tflite",
            )

            # Documentation: [https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter#from_keras_model](https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter#from_keras_model)
            converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)

            # Documentation: [https://www.tensorflow.org/api_docs/python/tf/lite/Optimize](https://www.tensorflow.org/api_docs/python/tf/lite/Optimize)
            converter.optimizations = [
                tf.lite.Optimize.DEFAULT
            ]

            # Needed for quantization. When we start quantization it will use our data to calculate optimal scale and zero point
            converter.representative_dataset = (
                lambda: self.representative_data_gen(validation_loader=validation_loader)
            )

            # Below three calls are needed to have fully supported INT8 quantization
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            # Let's keep input and output as float and see if we will get better results in steering
            converter.inference_input_type = tf.float32
            converter.inference_output_type = tf.float32

            # Convert model and save it
            tflite_model = converter.convert()

            with open(tflite_path, "wb") as f:
                f.write(tflite_model)

            print(f"[Export Engine] INT8 TFLite saved to: {tflite_path}")

            return tflite_path
        except Exception as error:
            print(f"[Export Engine] Export to tfLite failed: {error}")
            raise error

    def export_c_header(self, tflite_path: str, rank: int) -> str:
        """Exports a binary TensorFlow Lite model as a C header (.h) file containing a&#x20;
            byte array for embedded deployment on microcontrollers.

            Args:
                tflite_path (str): Path to the .tflite binary file.
                rank (int): Model architecture rank.

            Returns:
                str: Path to the generated C header file.
        """
        try:
            # We prepare the path to the file
            header_path = os.path.join(
                self.export_dir,
                f"model_rank_{rank}.h",
            )

            # Open the tfLite file and read bytes
            with open(tflite_path, "rb") as f:
                model_bytes = f.read()


            variable_name = f"model_rank_{rank}_tflite"

            # Open the .h file to write into it
            with open(header_path, "w") as h:

                # Write into some comments
                h.write("#pragma once\n\n")

                h.write(
                    f"const unsigned char {variable_name}[] = {{\n"
                )

                # Going through all the bytes
                for i, byte in enumerate(model_bytes):

                    # Add spaces at the beginning of the line
                    if i % 12 == 0:
                        h.write("    ")

                    # Write inside the byte
                    h.write(f"0x{byte:02x},")
                    # Every 12 characters split the line
                    if i % 12 == 11:
                        h.write("\n")
                    else:
                        h.write(" ")

                h.write("\n};\n\n")
                # At the end also write the num of bytes
                h.write(
                    f"const unsigned int {variable_name}_len = {len(model_bytes)};\n"
                )

            print(f"[Export Engine] C header saved to: {header_path}")

            return header_path
        except Exception as error:
            print(f"[Export Engine] Export to .h failed: {error}")
            raise error
  
    def export_top_models(self, res_X: np.ndarray, res_F: np.ndarray, train_loader: DataLoader, test_loader: DataLoader, validation_loader:DataLoader, top_n: int = 5) -> List[Dict[str, Any]]:
        """Processes the optimization results, identifies top models, trains them, and exports their metadata and weights.

        Args:
            res_X (np.ndarray): Array of model architectures from the optimization results.
            res_F (np.ndarray): Array of objective function values corresponding to res_X.
            train_loader: DataLoader for training data.
            test_loader: DataLoader for testing data.
            validation_loader: DataLoader for validation data.
            top_n (int): Number of top models to export.

        Returns:
            List[Dict[str, Any]]: List of metadata dictionaries (profiles) for exported models.   &#x20;
        """

        if res_X is None or len(res_X) == 0:
            print("[Export Engine Warning] No valid Pareto candidates found to export.")
            return []

        #If less than top_n models are avaiable, adjust top_n accordingly
        top_n = min(top_n, len(res_X))
        sorted = np.argsort(res_F[:, 0])[:top_n]  # Sort by the first objective (e.g., validation loss)

        exported_profiles = []
        #Due to combination of TEnsorflow and pytorch
        exported_models = []
        print(f"\n[Export Engine] Starting export process for Top {len(sorted)} Pareto models...")

        for rank, idx in enumerate(sorted, start=1):
            try:

                x_candidate = res_X[idx]
                f_candidate = res_F[idx]

                x_rounded = np.round(x_candidate).astype(int)
                num_hidden_layers = self.config.ai.num_of_hidden_layers

                # Create a structured layer structure based on the rounded candidate architecture
                layer_structure = [int(x_rounded[i]) for i in range(num_hidden_layers)]

                act_idx = int(x_rounded[num_hidden_layers])  # Activation function index

                act_map = {0: "relu", 1: "tanh", 2: "sigmoid"}
                activation_type = act_map.get(act_idx, "relu")

                layer_norm = bool(x_rounded[num_hidden_layers + 1])  # Layer normalization flag

                validation_loss = float(f_candidate[0])  # Validation loss for the candidate model
                tinyML_cost = float(f_candidate[1])  # TinyML cost for the candidate model
                print(f"[Export Engine] Exporting Rank #{rank}: Layers={layer_structure}, Act={activation_type}, Loss={validation_loss:.4f}")

                self.config.ai.num_of_epochs = 20
                trainer = self.trainer_factory(
                    layer_structure=layer_structure,
                    activation_type=activation_type,
                    layer_norm=layer_norm,
                    config=self.config
                )

                # Lets train the model
                trainer.run_training()

                # Get the model
                model = trainer.model
                model.eval()
                exported_models.append(model)
                # Get the final validation loss after training
                final_loss = trainer.epoch_validation_loss

                # Calculate the number of trainable parameters and model size in kilobytes
                param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
                model_size_kb = (param_count * 4) / 1024.0

                # Prepare metadata for export
                weights_filename = f"model_rank_{rank}.pt"
                weights_path = os.path.join(self.export_dir, weights_filename)
                torch.save(model.state_dict(), weights_path)

                profile = {
                    "rank": rank,
                    "layer_structure": layer_structure,
                    "activation_type": activation_type,
                    "layer_norm": layer_norm,
                    "parameter_count": param_count,
                    "model_size_kb": round(model_size_kb, 2),
                    "tinyml_cost": round(tinyML_cost, 4),
                    "validation_loss": round(final_loss, 6),
                    "is_valid": bool(final_loss < 999.0 and not np.isnan(final_loss)),
                    "weights_path": weights_path,
                }

                exported_profiles.append(profile)

            except Exception as e:
                print(f"[Export Engine ERROR] Failed to export model Rank #{rank}. Error details: {str(e)}")
                continue
        try:
            import tensorflow as tf
            for profile, model in zip(exported_profiles, exported_models):


                rank = profile["rank"]
                layer_structure = profile["layer_structure"]
                activation_type = profile["activation_type"]
                layer_norm = profile["layer_norm"]
                weights_path = profile["weights_path"]
                # Create model, ideally in the function, but due to Linux problem with tensorflow and torch we add it here
                #tf_model = tf.keras.Sequential()

                inputs = tf.keras.layers.Input(
                    shape=(
                        self.config.pipeline.window_size,
                        len(self.config.pipeline.input_features)
                    )
                )
                # First layer is always GRU, but need to be changed if we ever change this!
                initial_state = tf.zeros([1, layer_structure[0]], dtype=tf.float32)
                x = tf.keras.layers.GRU(
                    layer_structure[0],
                    return_sequences=False,
                    reset_after=True,
                    unroll=True,
                    name="gru"
                )(inputs, initial_state=initial_state)

                # Define input shape based on dataset features
                #tf_model.add(tf.keras.layers.Input(shape=(len(self.config.pipeline.input_features),)))
                # Dynamically build hidden layers from NAS structure
                for neurons in layer_structure[1:]:
                    # Skip inactive/disabled layers (0 or negative neurons)
                    if neurons <= 0:
                        continue
                    # Add fully connected layer
                    #tf_model.add(tf.keras.layers.Dense(neurons))
                    x = tf.keras.layers.Dense(neurons)(x)
                    # Optionally add normalization
                    if layer_norm:
                        #tf_model.add(tf.keras.layers.LayerNormalization())
                        x = tf.keras.layers.LayerNormalization()(x)
                    # Apply activation function
                    if activation_type == "relu":
                        #tf_model.add(tf.keras.layers.ReLU())
                        x = tf.keras.layers.ReLU()(x)

                    elif activation_type == "tanh":
                        #tf_model.add(tf.keras.layers.Activation("tanh"))
                        x = tf.keras.layers.Activation("tanh")(x)

                    elif activation_type == "sigmoid":
                        #tf_model.add(tf.keras.layers.Activation("sigmoid"))
                        x = tf.keras.layers.Activation("sigmoid")(x)

                    else:
                        raise ValueError(f"Unsupported activation: {activation_type}")

                steer_dense = tf.keras.layers.Dense(1, activation='tanh', name='steer_dense')(x)

                tf_model = tf.keras.Model(inputs=inputs, outputs=steer_dense)

                self.copy_weights(model, tf_model)

                tflite_path = self.export_tflite(tf_model=tf_model, rank=rank, validation_loader=validation_loader)

                c_header_path = self.export_c_header(tflite_path=tflite_path, rank=rank)
                verification = self.verify_models(pytorch_model=model, tf_model=tf_model, tflite_path=tflite_path, validation_loader=validation_loader)

                tflite_bytes = os.path.getsize(tflite_path)
                tflite_size_kb = round(tflite_bytes / 1024.0, 2)

                profile["tflite_path"]= tflite_path
                profile["c_header_path"]= c_header_path
                profile["tflite_size_kb"] = tflite_size_kb
                profile["verification"]= verification
                json_filename = f"model_rank_{rank}_profile.json"
                json_path = os.path.join(self.export_dir, json_filename)

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(profile, f, indent=4)

            print(f"[Export Engine] Successfully exported {len(exported_profiles)} models to '{self.export_dir}'.\n")
            return exported_profiles
        except Exception as e:
            print(f"[Export Engine ERROR] Fail in exporting other filefo")